import torch 
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from torchvision import transforms

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm # 데이터포인트에 색상을 입힘
import numpy as np
import os
import sys
# Autoencoder 모델 불러오기
from model import Autoencoder

# hyperparameter
EPOCH = 30
batch_size = 32
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("Using device is", device)


# 가우시안 노이즈 추가함수
def gaussian_noise(img):
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    sigma = 0.01

    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

# data load (가우시안 노이즈 추가)
download_root = './MNIST'
transform = transforms.Compose([
    transforms.ToTensor(),
])


train_dataset = MNIST(download_root, transform=transform, train=True, download=True)
valid_dataset = MNIST(download_root, transform=transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

# 원본 이미지 시각화 sample 추출
view_data = train_dataset.data[:5].view(-1, 28*28).float() / 255.0

# 모델 선언
model = Autoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) 
criterion = nn.MSELoss()

# scheduler 사용
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 학습하기 (noise가 들어간 이미지에서 노이즈를 제거하도록 학습)
def train(autoencoder, train_loader):
    autoencoder.train()
    total_loss = 0
    for step, (x, label) in enumerate(train_loader):
        noisy_x = gaussian_noise(x)  # 입력에 노이즈 더하기
        noisy_x = noisy_x.view(-1, 28*28).to(device)
        y = x.view(-1, 28*28).to(device)  # Auto Encoder이므로 입력과 출력 모두 x로 동일
        label = label.to(device)  

        encoded, decoded = autoencoder(noisy_x)

        loss = criterion(decoded, y)  # 입력과 출력의 MSE 구하기
        optimizer.zero_grad()  # 기울기 정보 초기화
        loss.backward()  # back propagation을 위한 기울기 추출
        optimizer.step()  # Adam을 이용한 최적화
        total_loss += loss.item()

    # scheduler.step()  # 스케줄러 스텝 이동
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# 이미지 저장 경로 설정
output_dir = 'Data_analysis/Denoising-AutoEncoder/output_img(add_noise)/'
os.makedirs(output_dir, exist_ok=True)

# 특정 epoch마다 저장
save_epochs = [1, 5, 10, 20, 30]


# 입력 이미지(노이즈 이미지) 시각화
# for step, (x, label) in enumerate(train_loader):
#     noisy_x = gaussian_noise(x)  
#     noisy_x_ = noisy_x.clone().cpu().numpy()

#     fig, a = plt.subplots(1, 5, figsize=(15, 3))
    
#     for i in range(5):
#         img = np.transpose(noisy_x_[i], (1, 2, 0))  # 이미지를 (height, width, channels)로 변환
#         img = (img * 255).astype(np.uint8)  # 이미지 정규화
#         a[i].imshow(img, cmap='gray')
#         a[i].set_xticks(()); a[i].set_yticks(())
        
#     plt.show()
#     break  # 첫 번째 배치에 대해서만 시각화 실행

for epoch in range(1, EPOCH+1):
    avg_loss = train(model, train_loader)
    print(f"Epoch [{epoch}/{EPOCH}], Loss: {avg_loss:.4f}")

    test_x = view_data.to(device)
    _, decoded_data = model(test_x)

    if epoch in save_epochs:
        # 원본과 디코딩 결과 비교해보기
        f, a = plt.subplots(2, 5, figsize=(5, 2))
        for i in range(5):
            img = np.reshape(view_data.cpu().numpy()[i], (28, 28))  # 파이토치 텐서를 넘파이로 변환합니다.
            a[0][i].imshow(img, cmap='gray')
            a[0][i].set_xticks(()); a[0][i].set_yticks(())

        for i in range(5):
            img = np.reshape(decoded_data.cpu().data.numpy()[i], (28, 28)) 
            a[1][i].imshow(img, cmap='gray')
            a[1][i].set_xticks(()); a[1][i].set_yticks(())
        
        # 이미지 저장
        plt.savefig(f"{output_dir}epoch_{epoch}.png")
        plt.close()

# 3차원 축소 후 시각화
def visualize_3d(autoencoder, data_loader):
    autoencoder.eval()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color_map = cm.get_cmap('Spectral')

    with torch.no_grad():
        for step, (x, label) in enumerate(data_loader):
            x = x.view(-1, 28*28).to(device)
            encoded, _ = autoencoder(x)
            encoded = encoded.cpu().numpy()
            label = label.cpu().numpy()
            
            scatter = ax.scatter(encoded[:, 0], encoded[:, 1], encoded[:, 2], c=label, cmap=color_map)
            if step == 0:
                fig.colorbar(scatter, ax=ax)
            if step > 20:  # 데이터 양을 조절
                break
    plt.show()

# 3차원 시각화 실행
visualize_3d(model, valid_loader)

