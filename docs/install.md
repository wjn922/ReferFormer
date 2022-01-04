# Installation

We provide the instructions to install the dependency packages.

## Requirements

We test the code in the following environments, other versions may also be compatible:

- CUDA 11.1
- Python 3.7
- Pytorch 1.8.1



## Setup

First, clone the repository locally.

```
git clone https://github.com/wjn922/ReferFormer.git
```

Then, install Pytorch 1.8.1 using the conda environment.
```
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
```

Install the necessary packages and pycocotools.

```
pip install -r requirements.txt 
pip install 'git+https://github.com/facebookresearch/fvcore' 
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Finally, compile CUDA operators.

```
cd models/ops
python setup.py build install
cd ../..
```