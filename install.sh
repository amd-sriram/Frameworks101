#python3 -m venv .frameworks_env
source .frameworks_env/bin/activate

#sudo apt update
#sudo apt install libc6
#onnx runtime for radeon gpu
#https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-onnx.html
sudo apt -y install migraphx
sudo apt -y install half
python3 -m pip install -r requirements.txt
python3 -m pip install onnxruntime-rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/
