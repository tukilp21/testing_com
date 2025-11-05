
for transferring command line while implementing [FROSS]([url](https://github.com/Howardkhh/FROSS?tab=readme-ov-file#3-run-orb-slam3-on-replicassg))

https://chatgpt.com/share/690adfa4-5648-8005-b49e-a7a68402671b

```
# fresh venv recommended
`python -m pip install --upgrade pip wheel setuptools`

# A) Install Torch 1.9.1 CUDA 11.1 wheels
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html
# (These are the official cu111 binaries.) :contentReference[oaicite:2]{index=2}

# B) Install your CPU-safe deps - with lower pip version to install pytorch-lightning==1.7.7 
python -m pip install "pip==23.2.1"
pip install -r cpu_safe_requirement.txt

# C) Install PyG extensions from the matching cu111 wheel index
pip install -f https://data.pyg.org/whl/torch-1.9.1+cu111.html \
  torch-scatter==2.0.8 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1

# D) Install PyG core
pip install torch-geometric==2.0.4
# (PyG ships versioned wheel indices per Torch+CUDA; using the right index avoids source builds.) :contentReference[oaicite:3]{index=3}
```

Quick validation

```
print("PyG OK:", torch_geometric.__version__)
```

Install TensorRT 8.x for CUDA 11.x

Download a CUDA-11.x TensorRT tarball (x86_64) from NVIDIA archives — TensorRT 8.2.5 is a good fit and explicitly supports CUDA 11.0–11.5 (incl. 11.2). [NVIDIA Docs](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-825/install-guide/index.html)

1. Install NVIDIA’s repo keyring (fixes NO_PUBKEY & lets apt fetch deps):
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
```
1. Ask apt to auto-fix the broken install (will pull missing CUDA/cuDNN bits):
```
sudo apt-get -f install
```
1. 

```
# 1) Add NVIDIA repo for the desired TensorRT 8.x (CUDA 11.x)  — follow the doc page you select
# 2) Install TensorRT runtime + Python bindings
sudo apt-get update
sudo apt-get install -y \
  tensorrt \
  libnvinfer8 libnvinfer-plugin8 libnvonnxparsers8 libnvparsers8 \
  python3-libnvinfer python3-libnvinfer-dev

# (optional) utilities
sudo apt-get install -y uff-converter-tf
```
