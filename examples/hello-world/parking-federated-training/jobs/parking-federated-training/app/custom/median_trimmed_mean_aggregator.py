from typing import List, Dict, Any
import numpy as np

from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.apis.shareable import Shareable
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLMetaKey


def _to_numpy(x):
    return x if isinstance(x, np.ndarray) else np.asarray(x)


def _stack_param(param_list):
    return np.stack([_to_numpy(p) for p in param_list], axis=0)  # (n_clients, ...)


def _coord_median(stacked: np.ndarray) -> np.ndarray:
    return np.median(stacked, axis=0)


def _coord_trimmed_mean(stacked: np.ndarray, trim_ratio: float) -> np.ndarray:
    n = stacked.shape[0]
    if n <= 2:
        return stacked.mean(axis=0)
    flat = stacked.reshape(n, -1)
    k = int(np.floor(trim_ratio * n))
    if k == 0 or 2 * k >= n:
        return stacked.mean(axis=0)
    sort_idx = np.argsort(flat, axis=0)
    keep_idx = sort_idx[k:n - k, np.arange(flat.shape[1])]
    kept = flat[keep_idx, np.arange(flat.shape[1])]
    out = kept.mean(axis=0)
    return out.reshape(stacked.shape[1:])


class MedianTrimmedMeanAggregator(Aggregator):
    """
    Robust aggregator compatible with ScatterAndGather:
      - call order: accept(result, fl_ctx) ... repeated, then aggregate(fl_ctx)
      - mode: 'median' or 'trimmed_mean' (with trim_ratio)
    """

    def __init__(self, mode: str = "median", trim_ratio: float = 0.1, num_malicious_clients_simulated: int = 0):
        super().__init__()
        if mode not in ("median", "trimmed_mean"):
            raise ValueError("mode must be 'median' or 'trimmed_mean'")
        if not (0.0 <= trim_ratio < 0.5):
            raise ValueError("trim_ratio must be in [0.0, 0.5)")
        self.mode = mode
        self.trim_ratio = float(trim_ratio)
        self._buffer: List[Shareable] = []  # stash accepted client results each round
        self._num_malicious_clients_simulated = num_malicious_clients_simulated

    # Called per client result. Return True to accept into this round’s aggregation buffer.
    # def accept(self, shareable: Shareable, fl_ctx) -> bool:
    #     try:
    #         dxo = DXO.from_shareable(shareable)
    #     except Exception:
    #         return False
    #     if dxo.data_kind != DataKind.WEIGHTS:
    #         return False
    #     self._buffer.append(shareable)
    #     return True
    # def accept(self, shareable: Shareable, fl_ctx) -> bool:
    #     # Inspect the incoming result
    #     try:
    #         dxo = from_shareable(shareable)
    #     except Exception as e:
    #         self.log_warning(fl_ctx, f"[robust_agg] accept(): not a DXO ({e})")
    #         return False

    #     # Normalize data_kind to a string for wider version coverage
    #     kind = getattr(dxo, "data_kind", None)
    #     kind_str = str(kind) if not isinstance(kind, str) else kind
    #     # NVFLARE versions use either FLMODEL or FL_MODEL
    #     is_weights   = kind == DataKind.WEIGHTS or kind_str.endswith("WEIGHTS")
    #     is_fl_model  = (hasattr(DataKind, "FLMODEL") and kind == DataKind.FLMODEL) or \
    #                    kind_str.endswith("FLMODEL") or kind_str.endswith("FL_MODEL")

    #     if not (is_weights or is_fl_model):
    #         self.log_warning(fl_ctx, f"[robust_agg] accept(): ignoring result with data_kind={kind}")
    #         return False

    #     # Buffer the raw shareable; we'll parse on aggregate()
    #     self._buffer.append(shareable)

    #     # (optional) accumulate steps/examples
    #     try:
    #         steps = int(shareable.get_header(FLMetaKey.NUM_STEPS) or shareable.get_header("num_examples") or 0)
    #         if not hasattr(self, "_acc_steps"):
    #             self._acc_steps = 0
    #         self._acc_steps += steps
    #     except Exception:
    #         pass

    #     self.log_info(fl_ctx, f"[robust_agg] accept(): buffered {kind_str}")
    #     return True
    def accept(self, shareable: Shareable, fl_ctx) -> bool:
        try:
            dxo = from_shareable(shareable)   # your build exposes this helper
        except Exception as e:
            self.log_warning(fl_ctx, f"[robust_agg] accept(): not a DXO ({e})")
            return False

        kind = getattr(dxo, "data_kind", None)
        kind_str = str(kind) if not isinstance(kind, str) else kind

        is_weights     = (kind == DataKind.WEIGHTS) \
                        or kind_str.endswith("WEIGHTS")
        is_fl_model    = (hasattr(DataKind, "FLMODEL") and kind == DataKind.FLMODEL) \
                        or kind_str.endswith("FLMODEL") or kind_str.endswith("FL_MODEL")
        is_weight_diff = (hasattr(DataKind, "WEIGHT_DIFF") and kind == DataKind.WEIGHT_DIFF) \
                        or kind_str.endswith("WEIGHT_DIFF") or kind_str.endswith("WEIGHT-DIFF")

        if not (is_weights or is_fl_model or is_weight_diff):
            self.log_warning(fl_ctx, f"[robust_agg] accept(): ignoring result with data_kind={kind}")
            return False

        self._buffer.append(shareable)

        try:
            steps = int(shareable.get_header(FLMetaKey.NUM_STEPS) or shareable.get_header("num_examples") or 0)
            self._acc_steps = getattr(self, "_acc_steps", 0) + steps
        except Exception:
            pass

        self.log_info(fl_ctx, f"[robust_agg] accept(): buffered {kind_str}")
        return True



    # Called once per round after all accepts. Must consume self._buffer.
    def aggregate(self, fl_ctx) -> Shareable:
        shareables = self._buffer
        self._buffer = []  # clear buffer for the next round

        if not shareables:
            self.log_info(fl_ctx, "[robust_agg] no shareables to aggregate")
            return Shareable()

        # Extract WEIGHT_DIFF dicts
        client_diffs: List[Dict[str, Any]] = []
        num_malicious_weights_applied = 0
        for sh in shareables:
            dxo = from_shareable(sh)
            if self._num_malicious_clients_simulated > 0 and num_malicious_weights_applied < self._num_malicious_clients_simulated:
                num_malicious_weights_applied += 1
                # Simulate some malicious clients by uploading random weights
                if dxo.data_kind == DataKind.WEIGHT_DIFF:
                    if np.random.rand() < (self._num_malicious_clients_simulated / len(shareables)):
                        dxo.data = {k: np.random.randn(*np.array(v).shape).astype(np.array(v).dtype) for k, v in dxo.data.items()}
                        self.log_info(fl_ctx, "[robust_agg] Simulated a malicious client by uploading random WEIGHT_DIFF")
            if dxo.data_kind == DataKind.WEIGHT_DIFF:
                client_diffs.append(dxo.data)

        if not client_diffs:
            self.log_info(fl_ctx, f"[robust_agg] no client WEIGHT_DIFF found in {len(shareables)} shareables")
            return Shareable()

        # intersection of parameter names
        names = set(client_diffs[0].keys())
        for w in client_diffs[1:]:
            names &= set(w.keys())
        if not names:
            raise RuntimeError("No common parameter names across client updates.")

        agg_diffs: Dict[str, Any] = {}
        for name in names:
            per_client = [cw[name] for cw in client_diffs]
            stacked = _stack_param(per_client)
            merged = (
                _coord_median(stacked)
                if self.mode == "median"
                else _coord_trimmed_mean(stacked, self.trim_ratio)
            )
            agg_diffs[name] = merged

        # Wrap result as WEIGHT_DIFF
        out_dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=agg_diffs)

        # add meta if available
        total_steps = 0
        for sh in shareables:
            total_steps += int(sh.get_header(FLMetaKey.NUM_STEPS_CURRENT_ROUND) or sh.get_header("num_examples") or 0)
        out_dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, total_steps)

        out = out_dxo.to_shareable()
        try:
            out.set_header(FLMetaKey.CONTENT_TYPE, "DXO")   # works only if defined
        except AttributeError:
            pass
        out.set_header("content_type", "DXO")

        rnd = shareables[0].get_header(FLMetaKey.CURRENT_ROUND)
        if rnd is not None:
            out.set_header(FLMetaKey.CURRENT_ROUND, rnd)

        self.log_info(fl_ctx, f"[robust_agg] returning WEIGHT_DIFF with {len(agg_diffs)} tensors")
        return out

