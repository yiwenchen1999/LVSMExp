ssh ubuntu@147.185.41.15
sudo apt install nfs-common
sudo mkdir /data
sudo nano /etc/fstab
# add the following line:
# 10.15.69.83:/data /data nfs rw,nconnect=16,nfsvers=3 0 0
sudo mount -a
apt install python3.10-venv
python3.10 -m venv /data/venv/lvsmexp
source /data/venv/lvsmexp/bin/activate
git clone https://github.com/yiwenchen1999/LVSMExp.git
cd LVSMExp
pip install -r requirements.txt

https://huggingface.co/coast01/LVSM/resolve/main/scene_encoder_decoder_256.pt?download=true