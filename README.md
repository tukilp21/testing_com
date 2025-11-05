
for transferring command line while implementing [FROSS](https://github.com/Howardkhh/FROSS?tab=readme-ov-file#3-run-orb-slam3-on-replicassg)

[ChatGPT instruction](https://chatgpt.com/share/690adfa4-5648-8005-b49e-a7a68402671b)

## Note
- to deal with [root running out of memory](https://askubuntu.com/questions/57994/root-drive-is-running-out-of-disk-space-how-can-i-free-up-space)

---

## Dataset
[download script](https://gist.github.com/WaldJohannaU/55f5e35992ea91157b789b15eac4d432) for 3RScan dataset

## Env / Dir setup

fresh install
```
python -m pip install --upgrade wheel setuptools
# required for installing older packages - install on python 3.8
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

1. Install the **Python 3.8 wheel** from inside the tar:
```
# adjust the filename/path to your actual tar
tar -xzf TensorRT-8.2.5.1.Linux.x86_64-gnu.cuda-11.*.tar.gz
cd TensorRT-8.2.5.1
python -m pip install python/tensorrt-8.2.5.1-cp38-none-linux_x86_64.whl
```

*CHECKING FIRST*
```
export TRT_DIR="$HOME/opt/TensorRT-8.2.5.1"
echo "--- libnvinfer_plugin.so deps ---"
ldd "$TRT_DIR/lib/libnvinfer_plugin.so" | egrep -i 'cudart|cublas|cublasLt|cudnn|cufft|not found' || true
```
where
- `ldd` Runs the ldd command to list all shared libraries required by libnvinfer.so.
- `egrep` = grep - e --> allow grep with native support for extended regular expressions such as | (OR), ...

Make sure the loader can find the runtime libs from your apt install:
```
# point to TensorRT's lib directory from the tar
export TRT_DIR="$HOME/opt/TensorRT-8.2.5.1"
export LD_LIBRARY_PATH="$TRT_DIR/lib:${LD_LIBRARY_PATH}"
# optional: add trtexec to PATH if you want the CLI tool
export PATH="$TRT_DIR/bin:${PATH}"
```
- e.g., `PATH="_path1_:_path2_:_path3_`

<details>
  <summary> If `ImportError: libcublas.so.XX cannot open shared object file` </summary>
  
Install CUDA XX (compatible version) runtime
```
cd ~/opt
wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run
chmod +x cuda_11.4.4_470.82.01_linux.run
```

install libraries
```
sudo ./cuda_11.4.4_470.82.01_linux.run --silent --toolkit --no-drm --override
```

add to environments
```
#export TRT_DIR=$HOME/opt/TensorRT-8.2.5.1
export CUDA11_DIR=/usr/local/cuda-11.4
# export LD_LIBRARY_PATH=$CUDA11_DIR/lib64:$TRT_DIR/lib:/usr/lib/x86_64-linux-gnu:/opt/ros/melodic/lib
export LD_LIBRARY_PATH=$CUDA11_DIR/lib64:${LD_LIBRARY_PATH}
```
</details>


If unsure which CUDA libs are missing:
```
export TRT_DIR="$HOME/opt/TensorRT-8.2.5.1"
ldd "$TRT_DIR/lib/libnvinfer.so" | egrep 'cudart|cublas|cudnn|cufft|not found'
# or
ldd "$TRT_DIR/lib/libnvinfer_plugin.so" | egrep 'cudart|cublas|cudnn|cufft|not found'
```

validation
```
python - <<'PY'
import os, tensorrt as trt
print("TRT:", trt.__version__)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH","")[:200], "...")
PY
# optional: trtexec
command -v trtexec && trtexec --version
```

!!!!!!!!!!!!!!!!
Permanently setup env PATH
```
mkdir -p "$(conda info --base)/envs/fross-tst38/etc/conda/activate.d"
mkdir -p "$(conda info --base)/envs/fross-tst38/etc/conda/deactivate.d"

# activation hook
cat > "$(conda info --base)/envs/fross-tst38/etc/conda/activate.d/env_vars.sh" <<'EOF'
export TRT_DIR="$HOME/opt/TensorRT-8.2.5.1"
export LD_LIBRARY_PATH="$TRT_DIR/lib:${LD_LIBRARY_PATH}"
export PATH="$TRT_DIR/bin:${PATH}"
EOF

# deactivation hook
cat > "$(conda info --base)/envs/fross-tst38/etc/conda/deactivate.d/env_vars.sh" <<'EOF'
unset TRT_DIR
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed "s|$HOME/opt/TensorRT-8.2.5.1/lib:||")
export PATH=$(echo "$PATH" | sed "s|$HOME/opt/TensorRT-8.2.5.1/bin:||")
EOF
```
   

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

---
### Install new driver
```
# blacklist Nouveau (open-source driver) so it won't grab the GPU at boot
echo -e "blacklist nouveau\noptions nouveau modeset=0" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u

```
Press Ctrl+Alt+F3 to get to TTY, log in, then:
```
sudo systemctl stop gdm 2>/dev/null || sudo systemctl stop lightdm 2>/dev/null || true
```
- `systemctl stop gdm`: Stop the GNOME Display Manager (used by Ubuntu GNOME / GNOME Shell). If your system uses GNOME, this will end the login screen and desktop session.
- `2>/dev/null`: Redirect error messages (“stderr”) to `/dev/null` (i.e., discard them).
- `systemctl stop lightdm`: Stop the LightDM service (used by older Ubuntu and Xfce variants).
- `systemctl stop sddm`: Stop the Simple Desktop Display Manager (used by **KDE** Plasma).

From the directory where you downloaded NVIDIA-Linux-x86_64-580.105.08.run:
```
chmod +x NVIDIA-Linux-x86_64-580.105.08.run
sudo ./NVIDIA-Linux-x86_64-580.105.08.run --dkms --no-cc-version-check
```
- `--dkms` lets the module rebuild with kernel updates.
- If the installer asks to disable Nouveau, let it write the blacklist (you already did) and proceed.
- If Secure Boot is enabled, the installer will warn that the kernel module can’t load unsigned; either enroll a MOK/sign or disable Secure Boot in BIOS.

final step
```
sudo reboot
nvidia-smi
```
