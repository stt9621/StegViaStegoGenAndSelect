#!/usr/bin/env python3

import os
import argparse
import numpy as np
import cv2
import logging
import random
import shutil
import time
import math
from PIL import *
import PIL.Image
import scipy.io as sio
from pathlib import Path
import matlab.engine

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

import train_net



BOSSBASE_COVER_DIR = '/data/stt/steganalysis/BossBase-1.01-cover-resample-256'
BOWS_COVER_DIR = '/data/stt/steganalysis/BOWS2-cover-resample-256'
BATCH_SIZE = 1







class ToTensor():
	def __call__(self, sample):
		data, label = sample['data'], sample['label']

		data = np.expand_dims(data, axis=0)
		data = np.expand_dims(data, axis=0)
		data = data.astype(np.float32)

		new_sample = {
			'data': torch.from_numpy(data),
			'label': torch.from_numpy(label).long(),
		}

		return new_sample




class MyDataset(Dataset):
	def __init__(self, index_path, transform=None):
		self.index_list = np.load(index_path)
		self.transform = transform

		self.bossbase_cover_path = BOSSBASE_COVER_DIR + '/{}.pgm'
		self.bows_cover_path = BOWS_COVER_DIR + '/{}.pgm'

	def __len__(self):
		return self.index_list.shape[0]

	def __getitem__(self, idx):
		file_index = self.index_list[idx]

		if file_index <= 10000:
			cover_path = self.bossbase_cover_path.format(file_index)
		else:
			cover_path = self.bows_cover_path.format(file_index - 10000)

		cover_data = cv2.imread(cover_path, -1)

		data = cover_data
		label = np.array([0], dtype='int32')

		sample = {'data': data, 'label': label}

		if self.transform:
			sample = self.transform(sample)

		return sample




'''



def calAbsBool(threshold, grad):


	temp_grad = grad.squeeze().cpu().detach().numpy()
	grad_1d = grad.squeeze().view(-1)
	grad_1d = grad_1d.cpu().detach().numpy()

	grad_1d.sort()
	grad_len = len(grad_1d)

	min_th = grad_1d[int(grad_len*(threshold/2) - 1)]
	max_th = grad_1d[int(grad_len - grad_len*(threshold/2))]

	abs_bool = (temp_grad < min_th) + (temp_grad > max_th)
	abs_bool = 1 * abs_bool


	return abs_bool


'''







def calSignGrad(pt_path, indexPath, grad_dir, gpu_num):

	print("\tread checkpoint path:", pt_path)
	print("\tsaved grad path:", grad_dir)

	Path(grad_dir).mkdir(parents=True, exist_ok=True)


	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
	device = torch.device("cuda")
	kwargs = {'num_workers': 1, 'pin_memory': True}

	data_transform = transforms.Compose([
		ToTensor()
	])
	data_dataset = MyDataset(index_path=indexPath, transform=data_transform)
	data_loader = DataLoader(data_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
	index_list = np.load(indexPath)

	model = train_net.Net().to(device)
	all_state = torch.load(pt_path)
	model.load_state_dict(all_state['original_state'])
	model.eval()

	#torch.set_printoptions(edgeitems=5)
	
	for i, sample in enumerate(data_loader):

		file_index = index_list[i]
		#print(str(i+1), "-", file_index, "---------------------")

		data, label = sample['data'], sample['label']
		shape = list(data.size())
		data = data.reshape(shape[0] * shape[1], *shape[2:])
		label = label.reshape(-1)

		data, label = data.to(device), label.to(device)
		data.requires_grad = True

		output = model(data)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(output, label)

		model.zero_grad()
		loss.backward()

		grad = data.grad.data
		sign_grad = torch.sign(grad)
		sign_grad = sign_grad.cpu().numpy().squeeze()
		temp_grad = grad.cpu().numpy().squeeze()


		#abs_bool = calAbsBool(threshold, grad)
		
		

		sio.savemat('{}/{}.mat'.format(grad_dir, str(file_index)), mdict={'sign_grad':sign_grad, 'grad':temp_grad})



	