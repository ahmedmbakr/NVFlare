NUM_CLIENTS=3
GPU_ASSIGN_PER_CLIENT="0 1 2"

# source ~/NVFlare/examples/hello-world/nvflare_example/bin/activate
source ~/anaconda3/bin/activate nvflare
# conda activate nvflare
sleep 2
rm -rf /tmp/nvflare/poc
nvflare poc prepare -n ${NUM_CLIENTS} # Prepare the environment for n clients
cd ~/NVFlare/examples/hello-world
cp -r parking-federated-training/jobs/parking-federated-training/ /tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com/transfer/
cd ~/NVFlare
# python examples/hello-world/ab-alex-net-gtsrb/jobs/split_data.py ${NUM_CLIENTS} # Split the data between n clients: Update: AB: The data has been previously split. Replace this with a symbolic link.
cd ~/NVFlare/examples/hello-world
nvflare poc start -gpu ${GPU_ASSIGN_PER_CLIENT} -debug # AB: Start the clients with those GPUs. Each client will have one of the GPUs.

# Now, it goes to the admin panel

# ********************************************************************************
# Current best automated method
# Open another terminal and execute the following:
# cd ~/NVFlare/examples/hello-world/parking-federated-training
# python admin_automation.py
# ********************************************************************************

