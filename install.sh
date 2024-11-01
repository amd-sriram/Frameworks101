#python3 -m venv .frameworks_env
source .frameworks_env/bin/activate

#setup docker gpg keys
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get -f install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo service docker start

#sudo apt update
#sudo apt install libc6
#onnx runtime for radeon gpu
#https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-onnx.html
sudo apt -y install migraphx half git-all
python3 -m pip install -r requirements.txt
python3 -m pip install onnxruntime-rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/
python3 -m torch_ort.configure

#wget https://download.zetane.com/zetane/Zetane-1.7.4.deb 
#sudo apt-get install -f ./Zetane-1.7.4.deb 
