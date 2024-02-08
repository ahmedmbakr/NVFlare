cd ~/NVFlare
source ~/NVFlare/examples/hello-world/nvflare_example/bin/activate
rm -rf simulator-example
mkdir -p simulator-example/workspace
cp -r examples/hello-world/ab-alex-net-gtsrb/ simulator-example/
nvflare simulator -w simulator-example/workspace/ -n 2 -t 2 simulator-example/ab-alex-net-gtsrb/jobs/ab-alex-net-gtsrb/
