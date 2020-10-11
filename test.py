import os
import argparse
import time
import cv2
import numpy as np
import scipy.io as sio
import matlab.engine

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

import train_net



BOSSBASE_COVER_DIR = '/data/stt/steganalysis/BossBase-1.01-cover-resample-256'
BOWS_COVER_DIR = '/data/stt/steganalysis/BOWS2-cover-resample-256'
BATCH_SIZE = 16 // 2




class ToTensor():
	def __call__(self, sample):
		data, label = sample['data'], sample['label']

		data = np.expand_dims(data, axis=1)
		#data = np.expand_dims(data, axis=0)
		data = data.astype(np.float32)

		new_sample = {
			'data': torch.from_numpy(data),
			'label': torch.from_numpy(label).long(),
		}

		return new_sample




class MyDataset(Dataset):
	def __init__(self, index_path, stego_dir, transform=None):
		self.index_list = np.load(index_path)
		self.transform = transform

		self.bossbase_cover_path = BOSSBASE_COVER_DIR + '/{}.pgm'
		self.bows_cover_path = BOWS_COVER_DIR + '/{}.pgm'
		self.all_stego_path = stego_dir + '/{}.pgm'

	def __len__(self):
		return self.index_list.shape[0]

	def __getitem__(self, idx):
		file_index = self.index_list[idx]

		if file_index <= 10000:
			cover_path = self.bossbase_cover_path.format(file_index)
		else:
			cover_path = self.bows_cover_path.format(file_index - 10000)
		stego_path = self.all_stego_path.format(file_index)

		cover_data = cv2.imread(cover_path, -1)
		stego_data = cv2.imread(stego_path, -1)

		data = np.stack([cover_data, stego_data])
		label = np.array([0, 1], dtype='int32')

		sample = {'data': data, 'label': label}

		if self.transform:
			sample = self.transform(sample)

		return sample








def test(index_path, temp_stego_dir, output_dir, it, gpu_num):

	pt_path = output_dir + '/' + str(it-1) + '-' + 'params.pt'
	print("\tread checkpoint path:", pt_path)


	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
	device = torch.device("cuda")
	kwargs = {'num_workers': 1, 'pin_memory': True}


	data_transform = transforms.Compose([
		ToTensor()
	])
	data_dataset = MyDataset(index_path=index_path, stego_dir=temp_stego_dir, transform=data_transform)
	data_loader = DataLoader(data_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
	index_list = np.load(index_path)


	model = train_net.Net().to(device)
	all_state = torch.load(pt_path)
	model.load_state_dict(all_state['original_state'])
	model.eval()

	#torch.set_printoptions(edgeitems=5)
	correct = 0
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
		pred = output.max(1, keepdim=True)[1]
		correct += pred.eq(label.view_as(pred)).sum().item()
		#print("correct", correct)


	accuracy = correct / (len(data_loader.dataset) * 2)
	#print("accuracy", accuracy)

	return accuracy
		

		

