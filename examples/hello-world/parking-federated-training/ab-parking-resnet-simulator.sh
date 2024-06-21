cd ~/NVFlare
source ~/NVFlare/examples/hello-world/nvflare_example/bin/activate
rm -rf simulator-example
mkdir -p simulator-example/workspace
cp -r examples/hello-world/parking-federated-training/ simulator-example/
nvflare simulator -w simulator-example/workspace/ -n 3 -t 3 simulator-example/parking-federated-training/jobs/parking-federated-training/
