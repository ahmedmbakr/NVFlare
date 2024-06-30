cd ~/NVFlare
source ~/anaconda3/bin/activate nvflare
rm -rf simulator-example
mkdir -p simulator-example/workspace
cp -r examples/hello-world/parking-federated-training/ simulator-example/
mkdir -p simulator-example/workspace/
ln -sf ~/CNR-EXT/ simulator-example/workspace/data
nvflare simulator -w simulator-example/workspace/ -n 4 -t 4 simulator-example/parking-federated-training/jobs/parking-federated-training/
