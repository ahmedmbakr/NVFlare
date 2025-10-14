# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, List, Optional

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class GaussianPrivacy(DXOFilter):
    """DXO filter that adds calibrated Gaussian noise to outgoing model updates.

    This provides record-agnostic differential privacy at the update level by
    perturbing each tensor with Gaussian noise whose sigma is computed as a
    fraction (`sigma0`) of the 95th percentile (configurable) magnitude of the
    current update values.

    Args:
        sigma0 (float): non-negative fraction used to scale the noise sigma.
        max_percentile (float): percentile in [0, 100] to estimate the scale of values.
        data_kinds (list[str] | None): DXO data kinds to filter. Defaults to
            [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS].
    """

    def __init__(
        self,
        sigma0: float = 0.1,
        max_percentile: float = 95.0,
        data_kinds: Optional[List[str]] = None,
        clip_norm: Optional[float] = None,
    ):
        if not data_kinds:
            data_kinds = [DataKind.WEIGHT_DIFF, DataKind.WEIGHTS]

        super().__init__(
            supported_data_kinds=[DataKind.WEIGHTS, DataKind.WEIGHT_DIFF],
            data_kinds_to_filter=data_kinds,
        )

        if not np.isscalar(sigma0) or sigma0 < 0.0:
            raise ValueError(f"Expected non-negative scalar for sigma0, got: {sigma0}")
        if not np.isscalar(max_percentile) or max_percentile < 0.0 or max_percentile > 100.0:
            raise ValueError(f"max_percentile must be within [0, 100], got: {max_percentile}")

        self.sigma0 = float(sigma0)
        self.max_percentile = float(max_percentile)
        self.clip_norm = float(clip_norm) if clip_norm is not None else None

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        if self.sigma0 <= 0.0:
            self.log_warning(fl_ctx, "GaussianPrivacy: sigma0 is 0. No noise added.")
            return dxo

        weights = dxo.data

        # Optional global L2 clipping across all variables to bound sensitivity
        if self.clip_norm is not None and self.clip_norm > 0.0:
            # Compute global L2 norm of the update vector
            sq_sum = 0.0
            for name in weights:
                w = weights[name]
                sq_sum += float(np.sum(np.square(w)))
            global_l2 = float(np.sqrt(sq_sum))
            if global_l2 > 0.0:
                scale = min(1.0, self.clip_norm / global_l2)
                if scale < 1.0:
                    for name in weights:
                        weights[name] = weights[name] * scale
                    self.log_info(
                        fl_ctx,
                        f"GaussianPrivacy: applied global L2 clipping to norm={global_l2:.6g} with cap={self.clip_norm:.6g} (scale={scale:.6g}).",
                    )
        # Flatten and concatenate abs values to compute scale robustly
        all_abs_values = np.concatenate([np.abs(weights[name].ravel()) for name in weights])
        all_abs_nonzero = all_abs_values[all_abs_values > 0.0]
        if all_abs_nonzero.size == 0:
            self.log_warning(fl_ctx, "GaussianPrivacy: all update values are zero. Skipping noise addition.")
            return dxo

        max_value = np.percentile(a=all_abs_nonzero, q=self.max_percentile, overwrite_input=False)
        noise_sigma = self.sigma0 * max_value

        n_vars = len(weights)
        for var_name in weights:
            w = weights[var_name]
            weights[var_name] = w + np.random.normal(0.0, noise_sigma, np.shape(w))

        self.log_info(
            fl_ctx,
            f"GaussianPrivacy: added noise to {n_vars} vars with sigma={noise_sigma:.6g} "
            f"(sigma0={self.sigma0}, p{self.max_percentile}={max_value:.6g}).",
        )

        dxo.data = weights
        return dxo
