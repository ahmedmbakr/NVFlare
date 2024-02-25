source ~/NVFlare/examples/hello-world/nvflare_example/bin/activate
rm -rf /tmp/nvflare/poc
nvflare poc prepare -n 2
cd ~/NVFlare/examples/hello-world
cp -r ab-alex-net-gtsrb/jobs/ab-alex-net-gtsrb/ /tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com/transfer/
cd ~/NVFlare
python examples/hello-world/ab-alex-net-gtsrb/jobs/split_data.py # Split the data between 2 clients
cd ~/NVFlare/examples/hello-world
nvflare poc start

sleep 20 # Sleep for 10 seconds to let the server start

# Run the following from the admin panel
submit_job ab-alex-net-gtsrb

# Check the rest of admin panel commands in the following link: https://nvflare.readthedocs.io/en/main/real_world_fl/operation.html

# After the job finishes, the following commands are run in the adming panel:
# shutdown client
# admin@nvidia.com
# shutdown server
# admin@nvidia.com
