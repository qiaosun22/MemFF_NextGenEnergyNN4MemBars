import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
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
import sys
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision
from train_utils import *

device = torch.device('cuda')
batchsize = 100
#================================
# Quantization levels
#================================
img_half_level = 4
weight_bit = 4
output_bit = 6
isint = 0
clamp_std = 0
noise_scale = 0.05


def F_Mnist(batch_size_train, batch_size_test)
  compress_factor = 1
  reshape_f = lambda x: torch.reshape(x[0, ::compress_factor, ::compress_factor], (-1, ))

  transforms_train = Compose([
    #transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
    #transforms.RandomRotation(5),     #Rotates the image to a specified angel
    transforms.RandomAffine(0, translate= (.025,.025),shear=0), #Performs actions like zooms, change shear angles.
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # Set the color params
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),Lambda(lambda x: torch.flatten(x)) ])
  transforms_val = transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.1307,), (0.3081,)),Lambda(lambda x: torch.flatten(x))])

  train_dataset = datasets.FashionMNIST("~/.pytorch/F_MNIST_data/", train=True, download=True, transform=transforms_train)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)

  val_dataset = datasets.FashionMNIST("~/.pytorch/F_MNIST_data/", train=False, download=True, transform=transforms_val)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False)
  return train_loader,val_loader



def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:,:10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_



p_vector0=torch.normal(0, 1, size=(1, 1000))
p_vector0=p_vector0/ (p_vector0.norm(2, 1, keepdim=True) + 1e-4)
p_vector1=torch.normal(0, 1, size=(1, 1000))
p_vector1=p_vector1/ (p_vector1.norm(2, 1, keepdim=True) + 1e-4)

p_vector0=(p_vector0.repeat(batchsize, 1)).to(device=device)
p_vector1=(p_vector1.repeat(batchsize, 1)).to(device=device)

#p_vector0=(p_vector0.repeat(100, 1))
#p_vector1=(p_vector1.repeat(100, 1))


class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        #self.P=nn.Linear(in_features, out_features,bias=True)
        #FF
        self.P = my.Linear_quant_noise(in_features, out_features, weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std, noise_scale=noise_scale, bias=True)
        self.opt = Adam(self.P.parameters(),weight_decay=0, lr=0.0001)
        self.threshold = 4
        self.running_loss=0.0
        self.num_epochs_internal =25
        self.relu = torch.nn.ReLU()
        self.gelu=torch.nn.GELU()


    def forward(self, x,k):
      x=self.P(x)
      return self.gelu(x)



    def goodness(self,x_pos,x_neg,k):
      cos = nn.CosineSimilarity(dim=1, eps=1e-6)
      if k==0:
        g_p = (self.forward(x_pos,k))
        g_pos =cos(g_p, p_vector0)

        g_n = (self.forward(x_neg,k))
        g_neg =cos(g_n, p_vector0)
      if k==1:
        g_p = (self.forward(x_pos,k))
        g_pos =cos(g_p, p_vector1)

        g_n = (self.forward(x_neg,k))
        g_neg =cos(g_n, p_vector1)

      return g_pos,g_neg


    def train(self, x_pos, x_neg,k):
        self.running_loss=0.0
        for i in range(self.num_epochs_internal):

          g_pos,g_neg=self.goodness(x_pos,x_neg,k)
          delta=g_pos-g_neg
          loss = (torch.log(1 + torch.exp(
              -self.threshold*delta ))).mean()
          self.opt.zero_grad()
          loss.backward()
          self.opt.step()
          self.running_loss+=loss.item()
        return self.forward(x_pos,k).detach(), self.forward(x_neg,k).detach(), self.running_loss/self.num_epochs_internal


acc_test=[]
acc_train=[]

class FNet(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.J=torch.ones(40,40).to(device=device)
        self.e=torch.ones(1,40).to(device=device)
        #self.J=torch.ones(40,40)
        #self.e=torch.ones(1,40)
        self.layers = []
        self.para_weight=[]
        self.para_bias=[]
        self.loss_save=defaultdict(list)
        self.param_save=[]
        self.num_epochs =300
        self.num_label=10
        self.POOL = nn.AvgPool2d(3, stride=2,padding=1)
        self.Pad = nn.ZeroPad2d((0,0,0,4))
        self.sigmoid=nn.Sigmoid()
        self.N_layer=int(len(dims)/2)
        self.loss_list=[0] * int(len(dims)/2)
        for d in range(0, (len(dims)), 2):
            self.layers += [Layer( dims[d ], dims[d+1 ]).to(device=device)]
            #self.layers += [Layer( dims[d ], dims[d+1 ])]

    def predict(self, x):
        goodness_per_label = []
        for label in range(self.num_label):
            h = overlay_y_on_x(x, label)
            goodness = []
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            for k, layer in enumerate(self.layers):
              h = layer(h,k)
              g = (h)
              if k==0:
                goodness += [cos(g, p_vector0)]
              if k==1:
                goodness += [cos(g, p_vector1)]

            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, train_loader,val_loader):
        k=0
        for i in range(self.num_epochs):
            self.loss_list=[0] * self.N_layer
            for j, data in enumerate(train_loader, 0):
              x, y = data
              x, _ = my.data_quantization_sym(x, half_level=img_half_level)
              x, y = x.to(device=device),y.to(device=device)
              x_pos = overlay_y_on_x(x, y)
              y_n=y.clone()
              for s in range(x.size(0)):
                y_n[s]=int(choice(list(set(range(0, self.num_label)) - set([y[s].item()]))))
              x_neg = overlay_y_on_x(x, y_n)
              h_pos, h_neg = x_pos.to(device=device), x_neg.to(device=device)
              #h_pos, h_neg = x_pos, x_neg
              for k, layer in enumerate(self.layers):

                h_pos, h_neg,loss = layer.train(h_pos,  h_neg,k)

                self.loss_list[k]=self.loss_list[k]+loss


            self.loss_list=np.divide(self.loss_list, j+1)


            for l in range(self.N_layer):
                  self.loss_save[l].append(self.loss_list[l])
            print( f'[Epochs: {i + 1}], Loss: {self.loss_list}' )
            if i%10==9:
              with torch.no_grad():
                er=0.0
                for g, data in enumerate(train_loader, 0):
                  x_t, y_t = data
                  x_t, y_t = x_t.to(device=device), y_t.to(device=device)
                  er+=( self.predict(x_t).eq(y_t).float().mean().item())

                print('Train Accuracy:', (er)/(g+1)*100)
                acc_train.append((er)/(g+1)*100)
                er=0.0
                for t, data in enumerate(val_loader, 0):
                  x_te, y_te = data
                  x_te, y_te = x_te.to(device=device), y_te.to(device=device)
                  er+=( self.predict(x_te).eq(y_te).float().mean().item())
                print('Test Accuracy:', (er)/(t+1)*100)
                acc_test.append((er)/(t+1)*100)





def main():
    torch.set_default_dtype(torch.float32)
    #train_loader,val_loader=F_Mnist(10000,10000)
    train_loader,val_loader=F_Mnist(batchsize, batchsize)
    net = FNet([28*28,1000,1000,1000])
    net.train(train_loader,val_loader)
if __name__ == "__main__":
	main()
