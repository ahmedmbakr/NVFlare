# Federated Learning for Parking Space Detection

This repository implements a federated learning solution for parking space detection using NVIDIA FLARE (NVFlare) framework. The system uses a SSDNet model architecture for detecting parking spaces across multiple sites while preserving data privacy.

## Project Structure

The project implements two main execution modes:

1. Proof of Concept (PoC) mode for actual multi-client deployment
2. Simulator mode for testing and development

## Configuration Files

### Client Configuration (`config_fed_client.json`)

Located at `jobs/parking-federated-training/app/config/config_fed_client.json`:

- Dataset path configuration: Set in `datasetpath` field
- Model parameters:
  - Number of classes: 3
  - Batch size: 6
  - Number of workers: 4
  - Detection threshold: 0.5
- Learning parameters:
  - Learning rate: 0.001
  - FedProx loss mu: 0
  - Epochs per round: 1
- Model architecture: SSDNet

### Server Configuration (`config_fed_server.json`)

Located at `jobs/parking-federated-training/app/config/config_fed_server.json`:

- Server heartbeat timeout: 600 seconds
- Pretrained model path configuration
- Federation parameters:
  - Minimum clients required: 4
  - Number of rounds: 100
- Cross-site validation workflow setup

## Running the Project

### 1. Simulator Mode

To run the project in simulator mode (for testing and development):

```bash
./ab-parking-resnet-simulator.sh
```

This script:

- Creates a simulator workspace
- Links the data directory
- Runs one client in simulation mode

### 2. PoC Mode (Multi-Client Deployment)

For actual federated learning deployment across multiple clients:

```bash
./ab-parking-poc.sh
```

This script:

- Supports multiple clients (default: 4)
- Configures GPU assignments per client
- Sets up workspace at `/tmp/bakr-nvflare/poc`
- Links different parking datasets to different clients:
  - Site-1: PUCPR dataset
  - Site-2: UFPR04 dataset
  - Site-3: UFPR05 dataset
  - Site-4: CNR-EXT dataset

## Prerequisites

1. Install the required environment:

```bash
conda activate nvflare
```

2. Ensure the pretrained model is saved and its path matches the one specified in `config_fed_server.json`
3. Data paths are properly set up for each client (in PoC mode) or linked correctly (in simulator mode)

## Important Notes

- Make sure to run the pretrained model saving script before executing either mode
- Verify that the pretrained model path in `config_fed_server.json` matches the actual file location
- For PoC mode, ensure all data directories exist and are accessible to their respective clients
- GPU assignments can be modified in the PoC script based on your hardware configuration

## Contact Information
For any issues or contributions, please contact the project maintainers (ambakr@crimson.ua.edu) or open an issue in the repository.
