# (ParkFL) Federated Learning for Parking Space Detection

ParkFL is a federated-learning framework for camera‑based parking‑space detection and classification. Built on NVIDIA FLARE (NVFlare), it allows multiple parking sites to train a shared model without exchanging raw images, thereby protecting data privacy. ParkFL supports several state‑of‑the‑art FL algorithms, provides tools for scalability and robustness studies, and includes scripts to reproduce the experiments reported in the accompanying research paper. The framework has been used to benchmark algorithms such as FedAvg, FedProx, FedOpt, SCAFFOLD and a Trimmed‑Mean robust aggregator. Empirical studies show that SCAFFOLD remains stable under severe data imbalance and scales well to 20 clients, while Trimmed‑Mean defends against malicious clients.

This repository is a fork of the [NVFlare](https://github.com/NVIDIA/NVFlare) repository with modifications to support parking space detection across multiple sites while preserving data privacy.

## Table of contents

<!-- toc -->


## Feature overview by branch

| Branch                                            | Purpose/Features                                                                                                                                                                                                                                                                       | Notes                                                                                                                                                                                                                            |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **develop-AB / trimmed\_mean\_with\_scalability** | Implements the **Trimmed‑Mean** aggregation algorithm as a robust alternative to SCAFFOLD/FedAvg.  The trimmed‑mean aggregator discards the highest and lowest model updates from each round to tolerate malicious clients.  A version with 20 clients demonstrates scalability.       | Use this branch when studying adversarial robustness.  The trimming ratio `ρ` (0 ≤ ρ < 0.5) determines how many updates are removed from each end of the sorted update list. |
| **imbalance-expr**                                | Contains scripts and configuration files to reproduce **data‑imbalance experiments**.  Clients are assigned drastically different numbers of samples (2×, 5×, 10× and 20×) to test the resilience of FL algorithms.  SCAFFOLD shows less than 3 % drop in mAP for up to 10× imbalance. | Modify the dataset splits in this branch to create desired imbalance ratios.  The FL aggregator is typically set to SCAFFOLD.                                                                                                    |
| **albumentation**                                 | Demonstrates **data augmentation** using the Albumentations library.  Training images are augmented with rain, snow, fog, sun‑glare, contrast and night effects.  Augmentation improves generalization and the recommended probability is 30 %.                                        | Adjust augmentation probability and transformations in the dataset loader to experiment with robustness against adverse weather conditions.                                                                                      |
| **scalability-expr**                              | Evaluates the **scalability** of ParkFL.  Each of the four datasets is split into five non‑overlapping subsets to create **20 clients**.  Using SCAFFOLD, the system achieves ≈97.18 % mAP with 20 clients, only 0.27 % lower than the 4‑client baseline.                              | Edit the number of clients (`NUM_CLIENTS=20`) in the scripts and update data paths accordingly.                                                                                                                                  |
| **FedOpt**                                        | Implements the **FedOpt** (adaptive server optimization) algorithm using an SGD/Adam‑style update on the server.  Includes a detailed example in `examples/hello-world/parking-federated-training` demonstrating PoC and simulator modes.                                              | Use this branch to study server‑side optimization.  Modify optimizer hyper‑parameters (`lr`, `momentum`, `β₁`, `β₂`) and learning‑rate scheduler in the server configuration.                                                    |
| **scaffold**                                      | Provides a stand‑alone implementation of the **SCAFFOLD** algorithm.  SCAFFOLD reduces client drift via control variates and performs weighted aggregation; it shows stable accuracy under heterogeneous data.                                                                         | Use this branch to benchmark SCAFFOLD without the additional experiments.                                                                                                                                                        |
### How to switch branches

```bash
git checkout <branch_name>
```

## Repository structure

Most experiments are located under examples/hello-world/parking-federated-training. The key components are:

- `jobs/parking-federated-training/` – NVFlare job directory containing the client and server configuration files (`config_fed_client.json` and `config_fed_server.json`), custom trainer (`parkingFL_trainer.py`) and tester (`parkingFL_Tester.py`), and any extra data loaders or utilities.

- `admin_automation.py` – an automation script that submits and monitors PoC jobs using the NVFlare Admin API. Edit this script to adjust the number of clients or select the model architecture.

- `ab-parking-poc.sh` – shell script to prepare a multi‑client PoC workspace, link data folders, assign GPUs, and start the federated training process.

- `ab-parking-resnet-simulator.sh` – quick script to run a single‑client simulator for development and debugging.

- `nvflare_env.yaml` – conda environment specification containing all Python packages required to run NVFlare and the parking‑FL experiments.

The high‑level workflow mirrors Figure 2 of the paper: each client trains locally and sends model updates to the Aggregating Server, which aggregates them using the selected algorithm. When malicious clients are expected, the Trimmed‑Mean aggregator can discard extreme updates; for stable convergence under heterogeneous data and large numbers of clients, SCAFFOLD is recommended.

## Environment setup

All experiments depend on NVFlare and several deep‑learning libraries. A Conda environment file, `nvflare_env.yaml`, is provided in the root of the repository. To create and activate the environment:
```bash
conda env create -f nvflare_env.yaml
conda activate nvflare
```

After activation, install dependencies for Albumentations or other augmentations as needed:
```bash
pip install albumentations
```

## Running the project

ParkFL experiments can be executed in Simulator mode (single client on a single machine) or PoC mode (multiple clients on a local machine or cluster). Before running any experiment, ensure that the NVFlare conda environment is created and activated (see Environment setup below) and that a pretrained model checkpoint is available if required by the server configuration (explained below).

### Simulator mode

1. Save or export a pretrained model (e.g., using the COCO‑pretrained ResNet/SSD weights) to the path specified in `config_fed_server.json`. This file provides a strong starting point; COCO pretraining improves convergence and final mAP compared with random initialization.

2. Run the simulator script from the repository root:
    ```bash
    cd examples/hello-world/parking-federated-training
    bash ab-parking-resnet-simulator.sh
    ```
    This will create a workspace under `simulator-example/`, link a dataset (by default `~/CNR-EXT/`), and start the NVFlare simulator with one client for quick testing.

### Proof‑of‑Concept (PoC) mode

1. Prepare the multi‑client workspace and run the training session:
    ```bash
    cd examples/hello-world/parking-federated-training
    bash ab-parking-poc.sh
    ```
    The script performs the following steps:

    - Sets `NUM_CLIENTS` and `GPU_ASSIGN_PER_CLIENT` to determine the number of clients and which GPUs are assigned (one per client).
    - Creates or resets the NVFlare PoC workspace at `/tmp/bakr-nvflare/poc`.
    - Copies the job directory into the admin transfer folder.
    - Links each dataset directory to the corresponding client folder (by default: `PUCPR`, `UFPR04`, `UFPR05` and `CNR‑EXT`).
    - Starts the NVFlare PoC with debug mode.
2. In a separate terminal, submit and monitor the job using the admin automation script:
    ```bash
    cd examples/hello-world/parking-federated-training
    python admin_automation.py
    ```
    The script submits the job, monitors its progress, retrieves the global model and logs, produces performance visualizations, and optionally runs the tester on the aggregated model. Edit `admin_automation.py` to adjust `NUM_CLIENTS`, the workspace location (`POC_WORKSPACE`), or the model (`MODEL_NAME`).
3. At the end of the training, `admin_automation.py` saves the final global model checkpoint and training logs under `poc_output.zip` in the current directory. It also generates plots of training/validation loss and mAP over communication rounds inside the same zip file.

## Adjustable parameters

The following table summarizes the key parameters you can tune for different experiments. Modify these values in the indicated files to explore algorithmic variants, dataset setups and system scales.

| Parameter                                                   | Description                                                                                                                                                                                                                                                                                                                                                                                   | File/location                                                                     |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `NUM_CLIENTS`                                               | Total number of participating clients in PoC mode; set to 4 by default but can be increased to 20 for scalability experiments.                                                                                                                                                                                                                                                                | `ab-parking-poc.sh` (shell variable), `admin_automation.py` (Python variable)     |
| `GPU_ASSIGN_PER_CLIENT`                                     | Space‑separated list of GPU IDs assigned to each client.  Must have one entry per client.                                                                                                                                                                                                                                                                                                     | `ab-parking-poc.sh`                                                               |
| Data paths                                                  | Symbolic links mapping each client to its dataset (e.g., PUCPR, UFPR04, UFPR05, CNR‑EXT).  Adjust these paths to use your own datasets or to create imbalance by giving clients different numbers of samples.                                                                                                                                                                                 | `ab-parking-poc.sh` (lines starting with `ln -sf`)                                |
| `datasetpath`                                               | Relative path to the training/validation/test data used by a client.  The default is `../../../data`, which points to the linked dataset inside each client's folder.                                                                                                                                                                                                                         | `config_fed_client.json`                                                          |
| `num_classes`                                               | Number of classes (2 for empty/occupied plus background).                                                                                                                                                                                                                                                                                                                                     | `config_fed_client.json`                                                          |
| `batch_size`, `num_workers_dl`                              | Batch size and number of DataLoader workers for local training and validation.                                                                                                                                                                                                                                                                                                                | `config_fed_client.json`                                                          |
| `valid_detection_threshold`                                 | Confidence threshold used to filter detections during validation.                                                                                                                                                                                                                                                                                                                             | `config_fed_client.json`, `ParkingFL_Trainer` and `ParkingFL_Tester` constructors |
| `lr` (learning rate)                                        | Learning rate for the optimizer on each client.                                                                                                                                                                                                                                                                                                                                               | `config_fed_client.json` (`"lr"` under `parkingFL-learner.args`)                  |
| `fedproxloss_mu`                                            | FedProx hyperparameter µ.  Set to 0 for FedAvg; set to a small positive value (e.g., 1e‑5) to enable FedProx.                                                                                                                                                                                                                                                                                 | `config_fed_client.json`                                                          |
| `epochs`                                                    | Number of local epochs each client performs before aggregation.  Varying this parameter controls communication frequency; doubling local epochs halves communication but may hurt accuracy for certain algorithms.                                                                                                                                                                            | `config_fed_client.json`                                                          |
| `model_name`                                                | Choice of DL architecture: `resnet` (ResNet50 + FasterRCNN) or `ssdnet` (SSDNet16).  SSDNet16 yields \~10× faster inference and lower memory usage on edge devices.                                                                                                                                                                                                                           | `config_fed_client.json`, `admin_automation.py`                                   |
| `heart_beat_timeout`                                        | Time in seconds the server waits for a client heartbeat before considering it dropped.  Increase this for slow networks.                                                                                                                                                                                                                                                                      | `config_fed_server.json`                                                          |
| `source_ckpt_file_full_name`                                | Path to the pretrained checkpoint used to initialize the global model.  Must be updated if you change the model or its location.                                                                                                                                                                                                                                                              | `config_fed_server.json` (under `persistor.args`)                                 |
| `aggregator` & `shareable_generator`                        | Select the aggregation algorithm.  For FedOpt, use `InTimeAccumulateWeightedAggregator` and `PTFedOptModelShareableGenerator` (default in FedOpt branch).  For SCAFFOLD, set the aggregator to `ScaffoldAggregator` and the shareable generator to `PTWeightedModelShareableGenerator`.  For Trimmed‑Mean, set the aggregator to `TrimmedMeanAggregator` and adjust the trimming ratio `rho`. | `config_fed_server.json` → `components` and `workflows`                           |
| `min_clients`, `num_rounds`, `wait_time_after_min_received` | Control the federated training schedule: minimum number of clients required to aggregate each round, total number of communication rounds, and timeout after the minimum number of clients submit.  For scalability experiments, set `min_clients` equal to the total number of clients (e.g., 20) or a smaller quorum.                                                                       | `config_fed_server.json` (inside `workflows[0].args`)                             |
| Server optimizer & scheduler                                | When using FedOpt, tune `optimizer_args` (e.g., learning rate, momentum) and `lr_scheduler_args` (e.g., cosine‑annealing parameters) under the shareable generator to adjust server‑side updates.                                                                                                                                                                                             | `config_fed_server.json`                                                          |
| Trimmed‑Mean ratio `ρ`                                      | Fraction of client updates removed from each end of the sorted list when using Trimmed‑Mean.  A 30 % ratio tolerates up to 35 % malicious clients, maintaining ≈96 % mAP.  Set this value in the trimmed‑mean aggregator’s arguments.                                                                                                                                                         | `median_trimmed_mean_aggregator.py` (in the `trimmed_mean` branches)              |
| Augmentation probability & transforms                       | Probability of applying weather‑based augmentations and the list of transformations (rain, snow, fog, glare, contrast, night).  A 30 % augmentation probability provided the best performance.                                                                                                                                                                                                | Data loader or augmentation module in the `albumentation` branch                  |

## Experiment‑specific tips

- **Trimmed‑Mean robustness experiments**: Switch the aggregator to Trimmed‑Mean and set the trimming ratio (`rho`) in the aggregator’s configuration. Increase `NUM_CLIENTS` and add artificially malicious clients to test resilience; according to our experiments, a 30 % trimming ratio handles up to 5 malicious clients out of 20.

- **Albumentation experiments**: In the `albumentation` branch, edit the augmentation pipeline to enable/disable specific weather conditions and adjust the `p` parameter controlling how often an augmentation is applied. The revised manuscript reports that training with augmentation consistently improves accuracy and recommends a 30 % probability.
- **Scalability experiments**: Increase `NUM_CLIENTS` to 20, split each dataset into five subsets, and update the data links accordingly. Use SCAFFOLD and set `min_clients` to the desired number (e.g., all 20 clients).
- **FedOpt experiments**: Use the `FedOpt` branch and leave the aggregator as `InTimeAccumulateWeightedAggregator` and the shareable generator as `PTFedOptModelShareableGenerator`. Tune the server optimizer (`lr`, momentum, `β1`, `β2`) and scheduler (`T_max`, `eta_min`) in `config_fed_server.json`. 
- **SCAFFOLD experiments**: Set the aggregator to SCAFFOLD and use the default shareable generator. SCAFFOLD is robust to heterogeneous data and performs well with 2 or more local epochs.

## Extending ParkFL

Below is a file-by-file checklist to integrate a new model architecture (illustrated with a `deit` model name). This mirrors the structure you used when adding `trimmed_mean`, but focused on model integration rather than a new aggregator.

### Files to add

- `examples/hello-world/parking-federated-training/jobs/parking-federated-training/app/custom/DeiT.py`.
    - Implement a wrapper class similar to `Resnet.py` / `SSDnet.py` exposing:
        - `get_pretrained_model(num_classes: int)` (trainer path)
        - `get_model(num_classes: int)` (tester path)
        - `get_transform()` (train/val/test transforms)
      - Internally load the `DeiT` backbone, attach a detector/classifier head compatible with your training loop, and return a `torch.nn.Module`.

### Files to modify

- `examples/hello-world/parking-federated-training/jobs/parking-federated-training/app/custom/parkingFL_trainer.py`
    - Add `deit` option wherever model_name is checked (alongside `resnet` / `ssdnet`):
        - Import `DeiT` wrapper.
        - In `initialize()` function do the following:
          ```python
          if self.model_name == "deit":
            from DeiT import DeiT
            self.model = DeiT.get_pretrained_model(self.num_classes)
            transforms = DeiT.get_transform()
            ```
        - Ensure optimizer and loss remain valid for the DeiT head (adjust if needed).
        - In `examples/hello-world/parking-federated-training/jobs/parking-federated-training/app/custom/parkingFL_Tester.py`, add "deit" path:
            ```python
            if model_name == "deit":
                from DeiT import DeiT
                self.model = DeiT.get_model(num_classes)
                transforms = DeiT.get_transform()
            ```
        - In `examples/hello-world/parking-federated-training/jobs/parking-federated-training/app/custom/PkLotDataLoader.py`, if `DeiT` requires specific input size or normalization, update or route through `DeiT.get_transform()` to ensure consistent transforms (train/val/test).
        - In `examples/hello-world/parking-federated-training/jobs/parking-federated-training/app/config/config_fed_client.json` expose the new model in the client args:
            ```json
            "components": [
                {
                "id": "parkingFL-learner",
                "path": "parkingFL_trainer.ParkingFL_Trainer",
                "args": {
                    "model_name": "deit",
                    "num_classes": 2,
                    "lr": 0.001,
                    "epochs": 1,
                    "batch_size": 6,
                    "num_workers_dl": 4,
                    "valid_detection_threshold": 0.5,
                    "data_path": "{datasetpath}"
                }
                }
            ]
            ```
        - In `examples/hello-world/parking-federated-training/jobs/parking-federated-training/app/config/config_fed_server.json` update the initial `DeiT` checkpoint path in the persitor arg:
            ```json
            "components": [
                {
                "id": "persistor",
                "name": "PTFileModelPersistor",
                "args": {
                    "source_ckpt_file_full_name": "/abs/path/to/deit_pretrained_model.pt"
                }
                },
                {
                "id": "model",
                "path": "DeiT.DeiT",  
                "args": { }
                }
            ],
            ...
            ``` 
        - In `examples/hello-world/parking-federated-training/admin_automation.py`, set the `MODEL_NAME` variable to `"deit"` to run PoC with the new model:
            ```python
            MODEL_NAME = "deit"  # Options: "resnet", "ssdnet", "deit"
            ```

## Support

For questions or issues, please contact the project maintainers at ambakr@crimson.ua.edu
 or open an issue on GitHub.

## License

ParkFL is released under an [Apache 2.0 license](https://github.com/NVIDIA/NVFlare/blob/main/LICENSE).
