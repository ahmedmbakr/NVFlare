source ~/NVFlare/examples/hello-world/nvflare_example/bin/activate
rm -rf /tmp/nvflare/poc
nvflare poc prepare -n 2
cd ~/NVFlare/examples/hello-world
cp -r ab-alex-net-gtsrb/jobs/ab-alex-net-gtsrb/ /tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com/transfer/
cd ~/NVFlare
python examples/hello-world/ab-alex-net-gtsrb/jobs/split_data.py # Split the data between 2 clients
cd ~/NVFlare/examples/hello-world
nvflare poc start

# Run the following from the admin panel
# submit_job ab-alex-net-gtsrb

# After the job finishes, the following commands are run in the adming panel:
# shutdown client
# type the username: admin@nvidia.com
# shutdown server
# type the username: admin@nvidia.com
