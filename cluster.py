import os
import numpy as np
from datasets import load_dataset
from datasets import load_from_disk

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import transforms
from PIL import Image
import tqdm

output_dir = "clustered_images"  # 输出目录
# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载数据集
print("Loading dataset...")
dataset = load_dataset("bitmind/ffhq-256")['train']
#dataset = load_from_disk("path/to/your/local_dataset")['train']
# 打印数据集信息
print("\nDataset Info:")
print(dataset)

# 设置参数

n_clusters = 10  # 聚类数量
sample_size = 70000  # 使用的样本数量
image_size = (64, 64)  # 图像尺寸
batch_size = 1000  # 新增：处理批次大小

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
for i in range(n_clusters):
    os.makedirs(os.path.join(output_dir, f"cluster_{i}"), exist_ok=True)

# 预处理函数（保持不变）
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)

# 修改后的特征提取：分批处理
def extract_features_batch(images, batch_size):
    features = []
    for i in tqdm.tqdm(range(0, len(images), batch_size), desc="Extracting features"):
        batch = images[i:i+batch_size]
        batch_features = [np.array(img).reshape(-1) for img in batch]
        features.extend(batch_features)
        # 立即释放内存
        del batch, batch_features
    return np.array(features)

# 随机采样（保持不变）
print(f"\nSampling {sample_size} images...")
sampled_indices = np.random.choice(len(dataset), size=sample_size, replace=False)
sampled_indices = [int(i) for i in sampled_indices]

# 分批处理主逻辑
for batch_start in tqdm.tqdm(range(0, sample_size, batch_size), desc="Processing batches"):
    batch_end = min(batch_start + batch_size, sample_size)
    batch_indices = sampled_indices[batch_start:batch_end]
    
    # 1. 加载当前批次的图像
    batch_images = [dataset[i]['image'] for i in batch_indices]
    
    # 2. 提取特征
    batch_features = extract_features_batch(batch_images, batch_size//10)  # 子批次
    
    # 3. 标准化（注意需要预先拟合scaler或使用在线学习方法）
    if batch_start == 0:
        scaler = StandardScaler()
        scaler.partial_fit(batch_features)  # 首次拟合
    batch_features = scaler.transform(batch_features)
    
    # 4. 增量式K-means训练
    if batch_start == 0:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        kmeans.partial_fit(batch_features)  # 首次拟合
    else:
        kmeans.partial_fit(batch_features)
    
    # 5. 预测并立即保存
    batch_labels = kmeans.predict(batch_features)
    for idx, (label, original_idx) in enumerate(zip(batch_labels, batch_indices)):
        try:
            image = batch_images[idx]
            save_path = os.path.join(output_dir, f"cluster_{label}", f"image_{original_idx}.png")
            image.save(save_path)
        except Exception as e:
            print(f"Error saving image {original_idx}: {str(e)}")
    
    # 释放内存
    del batch_images, batch_features, batch_labels

print(f"\nClustering completed! Images are saved in {output_dir} directory.")