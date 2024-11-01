docker run --rm -it --device=/dev/kfd --device=/dev/dri -v ~/github/DeepLearningModels:/DeepLearningModels -v /var/run/docker.sock:/var/run/docker.sock --name sriram_test --network=host 126dc70ede3
 
cd /DeepLearningModels/
#python3 -m venv .frameworks_env
source .frameworks_env/bin/activate

pip install jsondiff gitpython
sh install.sh


python3 tools/run_models.py --liveOutput --tags migx_inference_resnet_benchmarks

docker stop container_migx_inference_resnet_benchmarks_migx_onnxrt.ubuntu.amd