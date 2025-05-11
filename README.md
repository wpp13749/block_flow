# Block Flow

Block Flow is a deep learning project for image generation and training, supporting datasets like CIFAR-10 and FFHQ.

## Project Structure
- Image Generation and Training (CIFAR-10 and FFHQ)
- FFHQ Dataset Clustering (for training label generation)

## Environment Requirements
- PyTorch 1.12.0 / 1.11.0
- Python 3.8.5
- CUDA 11.3

## Installation
```bash
pip install -r .\requirements.txt
```

## Usage

### 1. CIFAR-10 Training
```bash
python train_reverse_img_ddp.py --gpu 0 --dir ./runs/cifar10-beta1/ --weight_prior 1 --learning_rate 2e-4 --dataset cifar10 --warmup_steps 5000 --optimizer adam --batchsize 128 --iterations 500000 --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json
```

### 2. CIFAR-10 Image Generation
```bash
python generate.py --gpu 0,1 --dir runs/cifa10_HABR_rk45 --solver RK45  --res 32 --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-beta1/flow_model_360000_ema.pth --config_de  ./configs/cifar10_de.json --batchsize 1024   --config_en  ./configs/cifar10_en.json  --encoder ./runs/cifar10-beta1/forward_model_360000_ema.pth --atol 2e-5
```

### 3. FFHQ Training Preparation

#### 3.1 FFHQ Dataset Clustering
The FFHQ dataset clustering module is used to provide necessary label information for training. This module implements a memory-efficient clustering solution based on the K-means algorithm.

##### Features
- Loads images from the FFHQ-256 dataset
- Extracts image features
- Performs clustering using MiniBatchKMeans
- Organizes images into different folders based on clustering results

##### Run Clustering Script
```bash
python caption_ffhq_256.py
```

##### Configuration Parameters
- `n_clusters`: Number of clusters (default: 10)
- `sample_size`: Number of images to sample (default: 70,000)
- `image_size`: Image dimensions (default: 64x64)
- `batch_size`: Batch processing size (default: 1,000)
- `output_dir`: Output directory (default: "clustered_images")

##### Output Structure
```
clustered_images/
├── cluster_0/
│   ├── image_123.png
│   └── ...
├── cluster_1/
│   └── ...
...
└── cluster_9/
    └── ...
```

#### 3.2 FFHQ Model Training
After completing dataset clustering, you can start training the FFHQ model:
```bash
python train_reverse_img_ddp.py --gpu 0,1,2,3,4,5,6,7 --dir runs/ffhq-beta20 --weight_prior 1 --learning_rate 2e-4 --dataset ffhq --warmup_steps 40000 --batchsize 256 --iterations 800000 --config_en configs/ffhq_en.json --config_de configs/ffhq_de.json
```

#### 3.3 FFHQ Image Generation
For FFHQ image generation, you can use the following command:
```bash
python generate.py  --gpu 0,1,2,3 --dir runs/ffhq_FABR_10_N100  --res 64 --input_nc 3 --num_samples 50000  --ckpt ./runs/ffhq-10/flow_model_800000_ema.pth  --config_de  ./configs/ffhq_de.json --batchsize 512    --config_en ./configs/ffhq_en.json  --encoder ./runs/ffhq-10/forward_model_800000_ema.pth    --N 100
```

Note: When generating images, you need to modify the sampling strategy in `generate.py`. Replace the uniform sampling code:
```python
y = torch.randint(0, 10, (arg.batchsize,), device=device)
```
with non-uniform sampling code based on the class distribution from clustering results:
```python
# When clustering methods are used to generate synthetic labels (0–9), 
# the distribution of p(y) can be defined according to the percentage composition of each class.
probabilities = torch.tensor([
    0.073549515, 0.133344607, 0.065136205,
    0.081554137, 0.101340211, 0.106004876,
    0.146605786, 0.11002128, 0.094984428,
    0.087458955
], device=device)

probs = probabilities / probabilities.sum()
y = torch.multinomial(
    input = probs,              
    num_samples = arg.batchsize, 
    replacement = True          
)
```
These probabilities can be adjusted based on your specific requirements.

## Acknowledgements
Thanks to [fast_ode](https://github.com/sangyun884/fast-ode) and [EDM](https://github.com/nvlabs/edm) for providing their implementations, which have significantly contributed to this codebase. 
