# GTSRB Federated Learning Training

In this example, we change the original example `hello-pt` (Resides in the same folder) to train a new model on a new dataset.
The dataset used is the German Traffic Sign Recognition Benchmark (GTSRB).
More details about the details and download information of the dataset can be found [here](https://benchmark.ini.rub.de/)
Moreover, we utilized `AlexNet` network from [this link](https://mailto-surajk.medium.com/a-tutorial-on-traffic-sign-classification-using-pytorch-dabc428909d7).

## Before you Run

In this section, we will discuss some assumptions before you run this example.
First, it is assumed that this NVFlare repository resides in the home folder, as follows: `~/NVFlare`.
Second, it is assumed that the virtual environment exists in the following path: `NVFlare/examples/hello-world/nvflare_example`.
If this is not the case, please consider changing this directory.
For more information about creating the virtual environment, please perform the following steps, which were inspired from [this link](https://nvflare.readthedocs.io/en/main/example_applications_algorithms.html) to create a virtual environment and install the required libraries when not using Cuda:

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
cd ~/NVFlare/examples/hello-world/
python3 -m venv nvflare_example
source nvflare_example/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install jupyterlab
```

To run NVFlare with CUDA (if you have a GPU), please consider the following steps.
Those steps are tested on Ubuntu 18.04.

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh # Install Anaconda
bash Anaconda3-2021.05-Linux-x86_64.sh
sudo apt-get install build-essential
sudo apt install python3-pip
sudo apt install gcc

# To remove previous installed cuda vesrion, if you do not have CUDA driver 11.1
sudo apt-get remove --purge nvidia\*
sudo apt-get autoremove
sudo apt-get install cuda-11.1

# Check NVIDIA installtion
cat /proc/driver/nvidia/version # Will output NVIDIA Kernel model

sudo apt-get remove nvidia-cuda-toolkit
sudo apt install nvidia-cuda-toolkit
nvcc -V # Will Display CUDA compilation tools version

# Create Anaconda environment with CUDA called nvflare.
conda env create -f nvflare_env.yaml # This file exists in the root folder of the NVFlare repository.
conda activate nvflare
```

The previous steps were developed by Ahmed Bakr.
Please refer to him for any questions.
For day to day usage, please consider using the following command to activate the virtual environment.

```bash
conda activate nvflare
... # Use the virtual environment
conda deactivate # When you are done using the virtual environment
```

Finally, it is assumed that the dataset is downloaded in the following path: `~/data/gtsrb/GTSRB`.
Now, you are ready to jump to the next section and run the example.

## How to Run

To be able to run the program, the dataset has to be downloaded by executing the following script from the CMD.
In the first line of this bash script, you will find the download folder location, which is set to be `~/data` by default.

```bash
bash prepare_dataset.sh
```

Second, the program can be run in the simulator mode or the Proof of Concept (POC) mode.
It is easier to run it first from a simulator to check that everything works by executing the following script.
In this script, it is assumed that the NVFlare folder is in the home directory `~/NVFlare`.
If this is not the case, please consider changing it.
Furthermore, it activates the virtual environment that is necessary to run NVFlare libraries in the second line.

```bash
bash ab-alex-net-gtsrb-simulator.sh
```

The same program can be run in POC mode, which is more realistic, as it depends on independent processes rather than threads that represent different clients.
To run in POC mode, execute the following command:

```bash
bash ab-alex-net-gtsrb-poc.sh # You might be asked to press `Y` in the CMD.
```

An admin panel will be displayed.
Use it to submit the task, as follows:

```bash
submit_job ab-alex-net-gtsrb
```

After the completion of the task (will be displayed in the CMD), close both the clients and the admin panels by executing the following commands from the admin panel.

```bash
shutdown client # You will be asked to type the admin's username: admin@nvidia.com
shutdown server # You will be asked to type the admin's username: admin@nvidia.com
```

The output of running the task in POC mode is inside the following path: `/tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com`.
Note that this folder will be deleted upon the machine's restart.

## Things to Consider

Upon the change of the dataset folder from the assumed location, please consider changing it from the following files:

- Change the first line of the file `prepare_dataset.sh`.
- Change the following lines in the client's configuration file: `jobs/ab-alex-net-gtsrb/app/config/config_fed_client.json`:

```json
"datasetpath": "~/data/gtsrb/GTSRB",
"num_classes": 43,
"train_val_split" : 0.8,
"random_seed": 42,
```

## Running without NVFlare

In this section, we will discuss how to run the program without NVFlare.
The python-script: `normal_training.py` is the main script that trains the model.

Now, we will discuss the parameters that can be changed in the script, as shown below in the following code snippet:

```python
gtsrb = GTSRB(  lr=0.01,
                    epochs=100,
                    batch_size = 128,
                    train_val_split = 0.8,
                    load_model_from_disk = False, # If True, the model will be loaded from the disk
                    model_load_path = os.path.abspath(os.path.join(dir_path, "model_99.pth"))) # The path of the model that should be loaded from the disk if load_model_from_disk is True
```

The model will be trained, the statistics will be printed in the console, and the graphs will be saved in the `ab-alex-net-gtsrb` folder.

```python
gtsrb.local_train(validate=True, save_graphs_after_each_epoch=True) # AB: Training the model
    gtsrb.display_train_trackers() # AB: Display the training trackers
```

The training loss is displayed below.

![alt text](image.png)

Moreover, the validation accuracy is displayed below.

![alt text](image-1.png)

Finally, the model's performance will be tested on the test dataset and the final accuracy is displayed in the console.

```python
    validation_on_test_accuracy = gtsrb.validate(gtsrb.test_loader) * 100 # AB: Final validation on the test data
    print(f"Validation accuracy on test data: {validation_on_test_accuracy:.2f}%")
```

The final achieved accuracy is displayed in the console, as shown below.

![alt text](image-2.png)
