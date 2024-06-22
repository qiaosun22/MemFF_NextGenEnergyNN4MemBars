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
from scipy import io
import argparse
import sys
import pickle
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision
import logging

def Mnist(batch_size_train,batch_size_test):


  compress_factor = 1
  reshape_f = lambda x: torch.reshape(x[0, ::compress_factor, ::compress_factor], (-1, ))

  transform_noise = transforms.RandomAffine(5, translate=(0.0, 0.022), scale=(0.97, 1.03))
  transforms_train = transforms.Compose([  transforms.ToTensor(),  Normalize((0.1307,), (0.3081,)),Lambda(lambda x: torch.flatten(x)) ])
  transforms_val = transforms.Compose([transforms.ToTensor(),  Normalize((0.1307,), (0.3081,)),Lambda(lambda x: torch.flatten(x))])

  train_dataset = MNIST('ml_dataset', train=True, download=True, transform=transforms_train)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)

  val_dataset = MNIST('ml_dataset', train=False, download=True, transform=transforms_val)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_test, shuffle=True)
  return train_loader,val_loader

path="/home/amomeni/scitas-examples/Basic/scitas-examples/Basic/result_Optics_Mnist.out"
f = open(path, "a")
f.write("Start loading data.mat \n")
f.close()
# Updoad the T matrix of MMOC
data_T = io.loadmat('/home/amomeni/scitas-examples/Basic/scitas-examples/Basic/data.mat')
T = np.array(data_T['T'])
T_inv = np.array(data_T['T_inv'])
mask_out = np.array(data_T['holo_params']['freq'][0][0][0][0][4])
mask_in = np.array(data_T['slm_params']['freq'][0][0][0][0][4])
height = 51
width = 51
batch_size = 2
logging.info('End')

f = open(path, "a")
f.write("End \n")
f.close()

def clip_mask(input):

    mask = input>0
    indices = np.nonzero (mask)
    x1 = np.amin(indices[0])
    x2 = np.amax(indices[0])
    y1 = np.amin(indices[1])
    y2 = np.amax(indices[1])

    mask_=input[x1:(x2+1),y1:(y2+1)]
    return mask_




def do_mask_1(stack, mask):
    data = stack[:, mask]
    return data
def fft2d_function_1(x, dtype=torch.complex128):
    # assumes x is of the size (batch, height, width)
    x = x.to(dtype)
    x_f = torch.fft.fftn(x, dim=(-2,-1))
    real_x_f, imag_x_f = x_f.real, x_f.imag
    real_x_f = torch.fft.fftshift(real_x_f, dim=(-2,-1))
    imag_x_f = torch.fft.fftshift(imag_x_f, dim=(-2,-1))

    return torch.complex(real_x_f, imag_x_f)

def ifft2d_function_1(x, dtype=torch.complex128):
    # assumes x is of the size (batch, height, width)
    x = x.to(dtype)
    real_x_f, imag_x_f = x.real, x.imag
    real_x_f = torch.fft.ifftshift(real_x_f, dim=(-2,-1))
    imag_x_f = torch.fft.ifftshift(imag_x_f, dim=(-2,-1))
    x_f = torch.complex(real_x_f, imag_x_f)
    x = torch.fft.ifftn(x_f, dim=(-2,-1))

    return x
def do_unmask_1(data, mask):
    batch_size = data.shape[0]
    h=mask.shape[0]
    w=mask.shape[1]
    mask = mask.reshape((1,-1))

    stack = torch.zeros((batch_size,h*w), dtype=torch.complex128).cuda()


    stack[:, mask[0,...]>0] = data[:,...]
    stack=torch.reshape(stack,(batch_size, h, w))

    return (stack)

mask_in_clip = clip_mask(mask_in)
mask_out_clip = clip_mask(mask_out)
def transmit_p (x, mask_in,mask_out, T):

    x_f = fft2d_function_1(x)
    x_f_m = do_mask_1(x_f,mask_in)
    x_f_m = x_f_m.transpose(-2, -1)
    y_f_mask = torch.matmul(torch.from_numpy(T).cuda(), x_f_m)
    y_f_mask = y_f_mask.transpose(-2, -1)
    y_f = do_unmask_1(y_f_mask,mask_out)
    y = ifft2d_function_1(y_f)
    return y




p_vector0=torch.normal(0, 1, size=(1, 26*26))
p_vector0=p_vector0/ (p_vector0.norm(2, 1, keepdim=True) + 1e-4)
p_vector1=torch.normal(0, 1, size=(1, 26*26))
p_vector1=p_vector1/ (p_vector1.norm(2, 1, keepdim=True) + 1e-4)
p_vector0=p_vector0.repeat(10000, 1)
p_vector0=p_vector0.cuda()
p_vector1=p_vector1.repeat(10000, 1)
p_vector1=p_vector1.cuda()

X=10
def overlay_y_on_x(x, y):
    x_ = torch.zeros(x.shape[0],X,x.shape[2]).cuda()
    x_[range(x.shape[0]), y,:] = (torch.abs(x).max())
    x=torch.cat((x_,x), 1)
    x=torch.cat((x,x_), 1)

    return x


class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=False, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.P=nn.Linear(in_features,out_features*N_F,bias=False).cuda()
        self.P1=nn.Linear(in_features,out_features*N_F,bias=False).cuda()
        self.P0=nn.Linear(in_features,out_features,bias=True).cuda()
        #self.opt = Adam(list(self.P.parameters())+list(self.P1.parameters()),weight_decay=0, lr=.03)
        self.opt = Adam(list(self.P0.parameters()),weight_decay=0, lr=0.00005)
        self.threshold = 5
        self.running_loss=0.0
        self.num_epochs_internal =20
        self.param=[]




    def forward(self, x,k):

        return (self.P0(x))

    def goodness(self,x_pos,x_neg,k):

      cos = nn.CosineSimilarity(dim=1, eps=1e-6)
      if k==0:
        g_p = torch.abs(self.forward(x_pos,k))
        g_pos =cos(g_p, p_vector0)

        g_n = torch.abs(self.forward(x_neg,k))
        g_neg =cos(g_n, p_vector0)
      else:
        g_p = torch.abs(self.forward(x_pos,k))
        g_pos =cos(g_p, p_vector1)

        g_n = torch.abs(self.forward(x_neg,k))
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


Scale=1
N_F=1
acc_test=[]
acc_train=[]
class FNet(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.J=torch.ones(40,40).cuda()
        self.e=torch.ones(1,40).cuda()
        self.layers = []
        self.para_weight=[]
        self.para_bias=[]
        self.loss_save=defaultdict(list)
        self.param_save=[]
        self.num_epochs =350
        self.num_label=10
        self.POOL = nn.AvgPool2d(3, stride=2,padding=1)
        self.Pad = nn.ZeroPad2d((0,0,0,4))
        self.sigmoid=nn.Sigmoid()
        self.N_layer=int(len(dims)/2)
        self.loss_list=[0] * int(len(dims)/2)
        for d in range(0, (len(dims)), 2):
            self.layers += [Layer( dims[d ], dims[d+1 ]).cuda()]

    def predict(self, x):
        goodness_per_label = []
        x=torch.reshape(x, (x.shape[0],int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1]))))
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        for label in range(self.num_label):
            h = overlay_y_on_x(x, label)
            #h=torch.cat((h,h), 1).cuda()
            goodness = []
            h1 = (h)
            for k, layer in enumerate(self.layers):
              if k==0:

                h_t=self.PNN_optics(h,h1,k)
              else:
                h=torch.reshape(h, (h.shape[0],int(np.sqrt(h.shape[1])), int(np.sqrt(h.shape[1]))))

                h_t=self.PNN_optics(h,h1,k)
              h = layer(h_t,k)
              g = torch.abs(h)
              if k==0:
                goodness += [cos(g, p_vector0)]
              else:
                goodness += [cos(g, p_vector1)]

            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def PNN_optics(self, x,x1, k):
      if k==0:
        x=torch.cat((x, x), 2)
        x=self.Pad(x)
      else:
        x=torch.repeat_interleave(x, 2,dim=1)
        x=torch.repeat_interleave(x, 2,dim=2)
        #x1=self.Pad(x1)
        #x=torch.cat((x, x1), 2)
      x=x[:,0:51,:51]
      Y = transmit_p(x, mask_in_clip,mask_out_clip, T)
      Y=self.POOL(torch.abs(torch.reshape(Y,(Y.shape[0],51, 51))))
      return torch.reshape(Y,(Y.shape[0],26*26)).detach()



    def train(self, train_loader,val_loader):
        k=0
        print('Start Training ...')
        for i in range(self.num_epochs):
            self.loss_list=[0] * self.N_layer
            for j, data in enumerate(train_loader, 0):
              x, y = data
              x, y = x.cuda(),y.cuda()
              x=torch.reshape(x, (x.shape[0],int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1]))))
              x_pos = overlay_y_on_x(x, y)
              y_n=y.clone()
              for s in range(x.size(0)):
                y_n[s]=int(choice(list(set(range(0, self.num_label)) - set([y[s].item()]))))
              x_neg = overlay_y_on_x(x, y_n)
              h_pos, h_neg = x_pos.cuda(), x_neg.cuda()

              h_pos1 = (x_pos)
              h_neg1 = (x_neg)
              for k, layer in enumerate(self.layers):

                if k==0:

                  p_t=self.PNN_optics(h_pos,h_pos1,k)
                  n_t=self.PNN_optics(h_neg,h_neg1,k)
                else:

                  h_pos=torch.reshape(h_pos, (h_pos.shape[0],int(np.sqrt(h_pos.shape[1])), int(np.sqrt(h_pos.shape[1]))))
                  h_neg=torch.reshape(h_neg, (h_neg.shape[0],int(np.sqrt(h_neg.shape[1])), int(np.sqrt(h_neg.shape[1]))))

                  p_t=self.PNN_optics(h_pos,h_pos1,k)
                  n_t=self.PNN_optics(h_neg,h_neg1,k)


                h_pos, h_neg,loss = layer.train(p_t,  n_t,k)

                self.loss_list[k]=self.loss_list[k]+loss


            self.loss_list=np.divide(self.loss_list, j+1)


            for l in range(self.N_layer):
                  self.loss_save[l].append(self.loss_list[l])
            #print( f'[Epochs: {i + 1}], Loss: {self.loss_list}' )

            f = open(path, "a")
            f.write(f' \n [Epochs: {i + 1}], Loss: {self.loss_list}' )
            f.close()

            if i%10==9:
              with torch.no_grad():
                er=0.0
                for g, data in enumerate(train_loader, 0):
                  x_t, y_t = data
                  x_t, y_t = x_t.cuda(), y_t.cuda()
                  er+=( self.predict(x_t).eq(y_t).float().mean().item())

                #print('Train Accuracy:', (er)/(g+1)*100)

                f = open(path, "a")
                f.write(f'\n Train Accuracy:, {(er)/(g+1)*100}')
                f.close()

                acc_train.append((er)/(g+1)*100)
                er=0.0
                for t, data in enumerate(val_loader, 0):
                  x_te, y_te = data
                  x_te, y_te = x_te.cuda(), y_te.cuda()
                  er+=( self.predict(x_te).eq(y_te).float().mean().item())

                f = open(path, "a")
                f.write(f'\n Test Accuracy:, {(er)/(t+1)*100}' )
                f.close()
                #print('Test Accuracy:', (er)/(t+1)*100)

                acc_test.append((er)/(t+1)*100)



acc_test=[]
acc_train=[]
torch.set_default_dtype(torch.float64)
f = open(path, "a")
f.write("START \n")
f.close()
train_loader,val_loader=Mnist(10000,10000)
net = FNet([26*26,26*26,26*26,26*26])
net.train(train_loader,val_loader)

