# Make sure to run the save_pretrained_file scrpt before running this file. In addition, make sure that the output path of the pretrained model is the same as the path provided in the config_fed_server.json file.
cd ~/NVFlare
source ~/anaconda3/bin/activate nvflare
rm -rf simulator-example
mkdir -p simulator-example/workspace
cp -r examples/hello-world/parking-federated-training/ simulator-example/
mkdir -p simulator-example/workspace/
ln -sf ~/CNR-EXT/ simulator-example/workspace/data
nvflare simulator -w simulator-example/workspace/ -n 1 -t 1 simulator-example/parking-federated-training/jobs/parking-federated-training/
