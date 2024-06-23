NUM_CLIENTS=3
GPU_ASSIGN_PER_CLIENT="0 1 2"
export NVFLARE_POC_WORKSPACE="/tmp/bakr-nvflare/poc" # Set the workspace for the NVFlare PoC. You cannot change this variable name as it is used by the NVFlare PoC scripts.
PROJECT_WORKSPACE_NAME="example_project/prod_00"

source ~/anaconda3/bin/activate nvflare
# conda activate nvflare
sleep 1
rm -rf ${NVFLARE_POC_WORKSPACE}

nvflare poc prepare -n ${NUM_CLIENTS} # Prepare the environment for n clients
cd ~/NVFlare/examples/hello-world
cp -r parking-federated-training/jobs/parking-federated-training/ ${NVFLARE_POC_WORKSPACE}/${PROJECT_WORKSPACE_NAME}/admin@nvidia.com/transfer/

# Prepare the data for all clients. The fastest way is to create symbolic links instead of copying the data to each client's directory.
ln -sf ~/pklot/PUCPR/ ${NVFLARE_POC_WORKSPACE}/${PROJECT_WORKSPACE_NAME}/site-1/data
ln -sf ~/pklot/UFPR04/ ${NVFLARE_POC_WORKSPACE}/${PROJECT_WORKSPACE_NAME}/site-2/data
ln -sf ~/pklot/UFPR05/ ${NVFLARE_POC_WORKSPACE}/${PROJECT_WORKSPACE_NAME}/site-3/data

cd ~/NVFlare
cd ~/NVFlare/examples/hello-world
nvflare poc start -gpu ${GPU_ASSIGN_PER_CLIENT} -debug # AB: Start the clients with those GPUs. Each client will have one of the GPUs.

# Now, it goes to the admin panel

# ********************************************************************************
# Current best automated method
# Open another terminal and execute the following:
# cd ~/NVFlare/examples/hello-world/parking-federated-training
# python admin_automation.py
# ********************************************************************************

