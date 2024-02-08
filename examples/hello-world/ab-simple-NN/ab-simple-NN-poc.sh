source ~/NVFlare/examples/hello-world/nvflare_example/bin/activate
nvflare poc prepare -n 2
cd ~/NVFlare/examples/hello-world
cp -r ab-simple-NN/jobs/ab-simple-NN/ /tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com/transfer/
nvflare poc start

# Run the following from the admin panel
# submit_job ab-simple-NN

# After the job finishes, the following commands are run in the adming panel:
# shutdown client
# type the username: admin@nvidia.com
# shutdown server
# type the username: admin@nvidia.com
