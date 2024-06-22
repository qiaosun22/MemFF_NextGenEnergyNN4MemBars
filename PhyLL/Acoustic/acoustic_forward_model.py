import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd
from random import choice
import os
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import io
from google.colab import files
import matplotlib.pyplot as plt


# upload the exprimental data

data_to_load = files.upload()
df_x = pd.read_csv(io.BytesIO(data_to_load['x_data_10k.csv']))
data_to_load = files.upload()
df_y = pd.read_csv(io.BytesIO(data_to_load['y_data_40_10k.csv']))


x_data=np.array((df_x), dtype=np.float32)
y_data=np.array((df_y), dtype=np.float32)
x_data=x_data[:,0:20]
y_data=y_data[:,0:20]
Train_data=x_data[0:9000,:]
Train_label=y_data[0:9000,:]
Val_data=x_data[9000:9300,:]
Val_label=y_data[9000:9300,:]
Test_data=x_data[9300:10000,:]
Test_label=y_data[9300:10000,:]


train_batch_size = 100
val_batch_size = 100
test_batch_size = 100
train_data = []
for i in range(Train_data.shape[0]):
   train_data.append([Train_data[i], Train_label[i]])

val_data = []
for i in range(Val_data.shape[0]):
   val_data.append([Val_data[i], Val_label[i]])

test_data = []
for i in range(Test_data.shape[0]):
   test_data.append([Test_data[i], Test_label[i]])


train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=val_batch_size)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=test_batch_size)



torch.set_default_dtype(torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 100),
            #nn.LayerNorm(100, eps=1e-05, elementwise_affine=True),
            nn.SiLU(),
            nn.Linear(100, 200),
            #nn.LayerNorm(200, eps=1e-05, elementwise_affine=True),
            nn.SiLU(),
            nn.Linear(200, 100),
            #nn.LayerNorm(100, eps=1e-05, elementwise_affine=True),
            nn.SiLU(),
            nn.Linear(100, 20)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()



mean_train_losses = []
mean_valid_losses = []
valid_acc_list = []
epochs =300

for epoch in range(epochs):
    model.train()

    train_losses = []
    valid_losses = []
    for i, (images, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        outputs = model(images.to(device))
        loss = torch.sqrt(loss_fn(outputs, labels.to(device)))
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (i * 128) % (128 * 100) == 0:
            print(f'{i * 128} / 50000')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            outputs = model(images.to(device))

            loss = torch.sqrt(loss_fn(outputs, labels.to(device)))
            valid_losses.append(loss.item())



    mean_train_losses.append(np.mean(train_losses))
    mean_valid_losses.append(np.mean(valid_losses))

 
    print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}%'\
         .format(epoch+1, np.mean(train_losses), np.mean(valid_losses)))



plt.plot(mean_train_losses, label='train')
plt.plot(mean_valid_losses, label='valid')



model.eval()

Test_losses = []
for i, (images, labels) in enumerate(test_loader):
    outputs = model(images.to(device))
    loss = torch.sqrt(loss_fn(outputs, labels.to(device)))
    Test_losses.append(loss.item())
print((Test_losses))


model.eval()
images,labels=next(iter(test_loader))

outputs = model(images.to(device)).cpu()
outputs=(outputs.detach().numpy())

plt.rcParams['font.size'] = '24'
fig=  plt.figure(figsize=(6*2,4*2), dpi=100)
plt.plot(outputs[3],'o:r',ms = 10, label='Forward model output')
plt.plot(labels[3],'o:b',ms = 10, label='Physical output')
plt.xlabel("Vector index")
plt.ylabel("Amplitude")
plt.xlim([-0.25,20])
plt.ylim([-0.25,1.25])
plt.legend()
plt.show()
fig.savefig("TA.png", dpi=800)
