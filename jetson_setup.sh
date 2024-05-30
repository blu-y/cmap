echo initial setting...
sudo apt update
sudo apt install pv vim -y
sudo vi /etc/apt/sources.list +%s/ports.ubuntu.com/ftp.kaist.ac.kr +wq!
sudo apt update
sudo apt install python3-pip libopenblas-dev axel pv -y
pip install --upgrade pip
sudo -H pip install -U jetson-stats
pip install numpy=='1.26.1'
echo "Follow this link to install pytorch on jetson"
echo "https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
echo "export PATH=/home/jetson/.local/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
wget https://github.com/rustdesk/rustdesk/releases/download/1.2.3-2/rustdesk-1.2.3-2-aarch64.deb
sudo dpkg -i rustdesk-1.2.3-2-aarch64.deb

# sudo systemctl enable rustdesk

echo cloning cmap repository...
git clone https://github.com/blu-y/cmap.git
cd cmap

echo downloading datasets
mkdir -p dataset/files
cd dataset
axel https://zenodo.org/records/7811795/files/Robot%40Home2_files.tgz
axel https://zenodo.org/records/7811795/files/Robot%40Home2_db.tgz
md5sum Robot@Home2_db.tgz Robot@Home2_files.tgz
echo d34fb44c01f31c87be8ab14e5ecd0767  Robot@Home2_db.tgz
echo c55465536738ec3470c75e1671bab5f2  Robot@Home2_files.tgz

echo unzipping datasets
pv Robot@Home2_db.tgz | tar -xzf - -C .
pv Robot@Home2_files.tgz | tar -xzf - -C ./files

wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo add-apt-repository "deb [arch=arm64] https://packages.microsoft.com/repos/vscode stable main" -y
sudo apt install code

echo installing python packages
cd ..
pip install -r ./requirements.txt
# pandas==2.2.2
# matplotlib==3.8.4
# tqdm==4.66.4
# open_clip_torch==2.24.0
# robotathome==1.1.9
# tramsformers==4.40.2

echo downloading open clip models
wget https://huggingface.co/timm/ViT-B-16-SigLIP/resolve/main/open_clip_pytorch_model.bin -O ./ViT-B-16-SigLIP/open_clip_pytorch_model.bin
wget https://huggingface.co/laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K/resolve/main/open_clip_pytorch_model.bin -O ./ViT-B-32-256/open_clip_pytorch_model.bin
wget https://huggingface.co/laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K/resolve/main/open_clip_pytorch_model.bin -O ./ViT-B-32/open_clip_pytorch_model.bin
wget https://huggingface.co/apple/DFN2B-CLIP-ViT-L-14/resolve/main/open_clip_pytorch_model.bin -O ./ViT-L-14-quickgelu/open_clip_pytorch_model.bin
# wget https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14/resolve/main/open_clip_pytorch_model.bin -O ./ViT-H-14-quickgelu/open_clip_pytorch_model.bin
