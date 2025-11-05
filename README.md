
for transferring command line while implementing [FROSS]([url](https://github.com/Howardkhh/FROSS?tab=readme-ov-file#3-run-orb-slam3-on-replicassg))

https://chatgpt.com/share/690adfa4-5648-8005-b49e-a7a68402671b

## Note
- to deal with [root running out of memory]([url](https://askubuntu.com/questions/57994/root-drive-is-running-out-of-disk-space-how-can-i-free-up-space))

---

## Dataset
[download script]([url](https://gist.github.com/WaldJohannaU/55f5e35992ea91157b789b15eac4d432)) for 3RScan dataset
https://gist.github.com/WaldJohannaU/55f5e35992ea91157b789b15eac4d432

## Env / Dir setup

dep install
```
# fresh venv recommended
python -m pip install --upgrade wheel setuptools
# required for older packages - install on python 3.8
python -m pip install "pip==23.2.1"
```
Install Torch 1.9.1 CUDA 11.1 wheels
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# (These are the official cu111 binaries.) :contentReference[oaicite:2]{index=2}
```

Install PyG extensions from the matching cu111 wheel index
```
pip install -f https://data.pyg.org/whl/torch-1.9.1+cu111.html torch-scatter==2.0.8 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1
# Install PyG core
pip install torch-geometric==2.0.4
# (PyG ships versioned wheel indices per Torch+CUDA; using the right index avoids source builds.) :contentReference[oaicite:3]{index=3}
```

Install reaming deps - with lower pip version to install pytorch-lightning<=1.7.7 
```
python -m pip install "pip==23.2.1"
pip install -r custom_requirement.txt
```

<details>
  <summary> if there are onnx, protobuf conflict with older python version  </summary>
  
  ```
  # 1) clean up the conflicting bits
  pip uninstall -y onnx protobuf
  
  # 2) install the compatible pair
  pip install "protobuf==3.20.1" "onnx==1.12.0"
  
  # 3) (re)install onnxruntime without changing protobuf
  pip install --no-deps "onnxruntime==1.19.0"
  # (ORT doesn’t need to upgrade protobuf for your use; --no-deps prevents bumps.) :contentReference[oaicite:3]{index=3}
  ```
</details>
<details>
  <summary> if fail to install pycocotools  </summary>

  ```
  pip install "pycocotools==2.0.7" --only-binary=:all:
  ```
</details>

_Quick validation_
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

1. Download the TensorRT 8.2.5.1 (CUDA 11.x) tar from NVIDIA (x86_64).

1. Install the Python 3.8 wheel from inside the tar:
```
 # adjust the filename/path to your actual tar
tar -xzf TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-11.*.tar.gz
cd TensorRT-8.2.5.1
python -m pip install python/tensorrt-8.2.5.1-cp38-none-linux_x86_64.whl
 ```
Make sure the loader can find the runtime libs from your apt install:
`export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"`
   

1. Install TensorRT runtime + Python bindings
```
sudo apt-get update
sudo apt-get install -y \
  tensorrt \
  libnvinfer8 libnvinfer-plugin8 libnvonnxparsers8 libnvparsers8 \
  python3-libnvinfer python3-libnvinfer-dev

# (optional) utilities
sudo apt-get install -y uff-converter-tf
```
