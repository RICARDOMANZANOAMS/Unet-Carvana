# Unet-Carvana
This repository does image segmentation using UNET network


conda create --name unet python=3.10.12

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

conda install ipykernel

python -m ipykernel install --user --name=unet

pip install jupyter