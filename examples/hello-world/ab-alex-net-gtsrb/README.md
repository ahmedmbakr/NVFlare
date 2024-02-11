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
For more information about creating the virtual environment, please perform the following steps, which were inspired from [this link](https://nvflare.readthedocs.io/en/main/example_applications_algorithms.html):

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
