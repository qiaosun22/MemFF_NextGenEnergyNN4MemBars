# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:14:28 2021

@author: Wang Ze
"""

import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
# Imports
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_utils import *

# ======================================== #
# Settings
# ======================================== #
# device
#device = torch.device("cuda")
device = torch.device("cpu")
num_workers = 0

# Set print precision
torch.set_printoptions(precision = 8)

# Model load/save flags
load_model_flag = 0
save_model_flag = 1
# Save model
PATH = './FashionMNIST.pth'

# Quantize input images
img_quant_flag = 1

check_accuracy_flag = 1

plot_flag = 0

# ======================================== #
# Hyperparameters
# ======================================== #
# network
in_channels = 1
num_classes = 10

# LR
LR_Adam = 1e-3
LR_Adam_min = LR_Adam / 100
# LR Scheduler
factor = 0.5
patience = 1e3
cooldown = 0

batch_size = 128
num_epochs_Adam = 10

drop_p = 0

min_LR_iter = patience * 10

# ======================================== #
# Noise-Quantization Training Parameters
# ======================================== #
img_half_level = 4
weight_bit = 4
output_bit = 6
isint = 0
clamp_std = 0
noise_scale = 0.05

# ======================================== #
# CNN for FashionMNIST Quant+Noise
# ======================================== #
class CNN(nn.Module):
    def __init__(self, in_channels = in_channels, num_classes = num_classes):
        super().__init__()
        self.conv1 = my.Conv2d_quant_noise(
            in_channels = in_channels,
            out_channels = 32,
            kernel_size = 3,
            stride = 1,
            padding = 0,
            bias = False,
            weight_bit = weight_bit,
            output_bit = output_bit,
            isint = isint,
            clamp_std = clamp_std,
            noise_scale = noise_scale,
        )
        # 26*26

        self.pool_s2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        # 20 * 13 * 13

        self.conv2 = my.Conv2d_quant_noise(
            in_channels = 32,
            out_channels = 32,
            kernel_size = 3,
            stride = 2,
            padding = 0,
            bias = False,
            weight_bit = weight_bit,
            output_bit = output_bit,
            isint = isint,
            clamp_std = clamp_std,
            noise_scale = noise_scale,
        )
        # 30 * 6 * 6
        # 30 * 3 * 3

        self.fc1 = my.Linear_quant_noise(288, 48,
                                         weight_bit = weight_bit,
                                         output_bit = output_bit,
                                         isint = isint,
                                         clamp_std = clamp_std,
                                         noise_scale = noise_scale,
                                         bias = False)

        self.fc2 = my.Linear_quant_noise(48, 10,
                                         weight_bit = weight_bit,
                                         output_bit = output_bit,
                                         isint = isint,
                                         clamp_std = clamp_std,
                                         noise_scale = noise_scale,
                                         bias = False)

        # weight = 100 * 10
        self.dropout = nn.Dropout2d(p = drop_p)
        self.dropout_1d = nn.Dropout(p = drop_p)

    def forward(self, x):
        # 第1层
        x = self.conv1(x)
        x = self.dropout(x)
        x = F.relu(x)

        # 第2层pooling
        x = self.pool_s2(x)

        # 第3层
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 第4层pooling
        x = self.pool_s2(x)

        # 第5层
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


# ======================================== #
# Load Dataset
# ======================================== #
train_dataset = datasets.FashionMNIST(root = "dataset/", train = True,
                                      transform = transforms.Compose([transforms.ToTensor()]),
                                      download = True)
test_dataset = datasets.FashionMNIST(root = "dataset/", train = False,
                                     transform = transforms.Compose([transforms.ToTensor()]), download = True)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = 0,
                          num_workers = num_workers)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = 0,
                         num_workers = num_workers)
#show_img(train_loader, img_half_level)

# ======================================== #
# Training Function
# ======================================== #
# Initialize network
model = CNN(in_channels = in_channels, num_classes = num_classes).to(device)

if load_model_flag == 1:
    load_model(model, PATH)


def train_net(model, train_loader, criterion, optimizer,
              epoch_num, scheduler = None, min_LR_iter = min_LR_iter):
    # Train Network
    model_max = copy.deepcopy(model.state_dict())
    accuracy_max = 0
    accuracy = 0
    LR = []
    loss_plt = []
    for epoch in range(epoch_num):
        print(f'Epoch = {epoch}')
        loop = tqdm(train_loader, leave = True)
        if min_LR_iter <= 0:
            break
        for batch_idx, (data, targets) in enumerate(loop):
            # Get data to cuda if possible
            data = data.to(device = device)
            targets = targets.to(device = device)

            # convert to 1bit image
            if img_quant_flag == 1:
                data, _ = my.data_quantization_sym(data, half_level = img_half_level)

            scores = model(data)

            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
            if scheduler != None:
                # 如果使用ReduceLROnPlateau，执行下面一行代码
                scheduler.step(loss)

                # 如果使用CosineAnnealingLR，执行下面一行代码
                # scheduler.step()
                pass
            accuracy = (sum(scores.argmax(dim = 1) == targets) / len(targets)).item() * 100
            lr_current = optimizer.param_groups[0]['lr']
            loop.set_postfix(
                accuracy = accuracy,
                loss = loss.item(),
                LR = lr_current
            )
            LR.append(lr_current)
            loss_plt.append(loss.item())
            if lr_current == LR_Adam_min:
                min_LR_iter -= 1
                if min_LR_iter <= 0:
                    break
            if accuracy >= accuracy_max:
                accuracy_max = accuracy
                model_max = copy.deepcopy(model)
        if plot_flag == 1:
            # if epoch % 10 == 0:
            plt.plot(LR)
            plt.show()
            plt.plot(loss_plt)
            plt.show()

        if save_model_flag == 1:
            print('model saved')
            save_model(model_max, PATH)

    model = model_max


# ======================================== #
# Train Model
# ======================================== #
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_adam = optim.Adam(model.parameters(), lr = LR_Adam)
# scheduler_adam = lr_scheduler.CosineAnnealingLR(optimizer_adam, eta_min = LR_Adam_min, T_max = T_max)
scheduler_adam = lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode = 'min', factor = factor,
                                                patience = patience, cooldown = cooldown,
                                                min_lr = LR_Adam_min)
# Train Network
train_net(model, train_loader, criterion, optimizer_adam, num_epochs_Adam, scheduler_adam)

# ======================================== #
# Check accuracy and stuff
# ======================================== #
# Check accuracy on training & test to see how good our model
if check_accuracy_flag == 1:
    accuracy_train = check_accuracy(train_loader, model, img_quant_flag, img_half_level)
    print(f"Accuracy on training set: {accuracy_train:.2f}")
    accuracy_test = check_accuracy(test_loader, model, img_quant_flag, img_half_level)
    print(f"Accuracy on test set: {accuracy_test:.2f}")

# Save quantized weight
model.eval()
checkpoint = torch.load(PATH, map_location='cpu')
half_level = 2 ** weight_bit / 2 - 1
for i in checkpoint:
    checkpoint[i], _ = my.data_quantization_sym(checkpoint[i],
                                                half_level = half_level,
                                                isint = 1)
    layer_name = i
    file_name = fr'{i}_quantized.npy'
    np.save(file_name, checkpoint[i].detach().cpu().numpy())
