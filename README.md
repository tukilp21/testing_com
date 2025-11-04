# testing_com
for transferring command line :)

```
# fresh venv recommended
`python -m pip install --upgrade pip wheel setuptools`

# A) Install Torch 1.9.1 CUDA 11.1 wheels
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html
# (These are the official cu111 binaries.) :contentReference[oaicite:2]{index=2}

# B) Install your CPU-safe deps
pip install -r requirements-cuda11_legacy.txt

# C) Install PyG extensions from the matching cu111 wheel index
pip install -f https://data.pyg.org/whl/torch-1.9.1+cu111.html \
  torch-scatter==2.0.8 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1

# D) Install PyG core
pip install torch-geometric==2.0.4
# (PyG ships versioned wheel indices per Torch+CUDA; using the right index avoids source builds.) :contentReference[oaicite:3]{index=3}
```

Quick validation

```
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
import torch_geometric, torch_scatter, torch_sparse, torch_cluster
print("PyG OK:", torch_geometric.__version__)
PY
```

Install TensorRT 8.x for CUDA 11.x

attempt 2
```
# example — adjust to your exact file name & location
tar -xzf TensorRT-8.2.5.*.Linux.x86_64-gnu.cuda-11.*.cudnn8.*.tar.gz
cd TensorRT-8.2.5.*

# Install the Python bindings (pick the wheel that matches your Python):
python -m pip install python/tensorrt-8.2.5.*-cp39-none-linux_x86_64.whl
# (if you also need parsers / extras)
python -m pip install python/onnx_graphsurgeon-*.whl  # optional

#Expose the shared libs at runtime:
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/lib"
# add that line to ~/.bashrc so it persists

```

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
