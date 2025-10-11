NUM_CLIENTS=4
GPU_ASSIGN_PER_CLIENT="0 1 2 3"

# source ~/NVFlare/examples/hello-world/nvflare_example/bin/activate
source ~/anaconda3/bin/activate nvflare
# conda activate nvflare
sleep 2
rm -rf /tmp/nvflare/poc
nvflare poc prepare -n ${NUM_CLIENTS} # Prepare the environment for n clients
cd ~/NVFlare/examples/hello-world
cp -r ab-alex-net-gtsrb/jobs/ab-alex-net-gtsrb/ /tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com/transfer/
cd ~/NVFlare
python examples/hello-world/ab-alex-net-gtsrb/jobs/split_data.py ${NUM_CLIENTS} # Split the data between n clients
cd ~/NVFlare/examples/hello-world
nvflare poc start -gpu ${GPU_ASSIGN_PER_CLIENT} -debug # AB: Start the clients with those GPUs. Each client will have one of the GPUs.

# Now, it goes to the admin panel

#sleep 20 # Sleep for 10 seconds to let the server start

# ********************************************************************************
# Current best automated method
# Open another terminal and execute the following:
# cd ~/NVFlare/examples/hello-world/ab-alex-net-gtsrb
# python admin_automation.py
# ********************************************************************************
# # Run the following from the admin panel.
# # TODO: AB: I do not know how to execute this command in the admin panel, yet.
# submit_job ab-alex-net-gtsrb 

# while true
#     do
#         check_status client # Check the status of the client. If the status is "started", then it is working fine.
#         sleep 30 # Sleep for 30 seconds
#     done

# # Check the rest of admin panel commands in the following link: https://nvflare.readthedocs.io/en/main/real_world_fl/operation.html

# # After the job finishes, the following commands are run in the adming panel:
# shutdown client
# admin@nvidia.com
# shutdown server
# admin@nvidia.com

# ********************************************************************************
# Alternative Method
# Open a new terminal and run the following commands:
# cd ~/NVFlare/examples/hello-world/ab-alex-net-gtsrb/jobs
# nvflare job submit -j ab-alex-net-gtsrb/ -debug
# Wait for the job to complete
# nvflare poc stop # You can also call this command if you wnat to stop the execution of the current tasks at any time.
# 
