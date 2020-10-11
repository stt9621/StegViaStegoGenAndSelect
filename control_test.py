import os
import argparse
import time
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

import test_net



# hyperparameters
T = 1

IMAGE_SIZE = 256
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16








def myParseArgs():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-g',
		'--gpuNum',
		help='Determine which gpu to use',
		type=str,
		choices=['0', '1', '2', '3'],
		required=True
	)

	parser.add_argument(
		'-p',
		'--payLoad',
		help='Determine the payload to embed',
		type=str,
		required=True
	)

	parser.add_argument(
		'-l',
		'--list_num',
		help='Determine the list num',
		type=str,
		required=True
	)

	args = parser.parse_args()
	
	return args





class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count





args = myParseArgs()

gpu_num = args.gpuNum
payload = float(args.payLoad)
list_num = int(args.list_num)


# list & path
all_list = './index_list/' + str(list_num) + '/all_list.npy'
step1_list = './index_list/' + str(list_num) + '/train_and_val_list.npy'
step2_list = './index_list/' + str(list_num) + '/test_list.npy'
step1_train_list = './index_list/' + str(list_num) + '/train_list.npy'
step1_val_list = './index_list/' + str(list_num) + '/val_list.npy'
step2_train_list = './index_list/' + str(list_num) + '/retrain_train_list.npy'
step2_test_list = './index_list/' + str(list_num) + '/retrain_test_list.npy'

des_dir = '/data/stt/adv_spa_srm_bl_p0.4'

model_dir = './adv-spa-srm-bl-p.4-test'
print("model_dir:", model_dir)
log_path = model_dir + '/test_log.txt'



if (os.path.exists(model_dir) == False):
	os.mkdir(model_dir)




xq_test_result = []



file_log = open(log_path, 'a')



###############################################################################################################################
###############################################1111111111111111111111111111####################################################


for it in range(0, T+1): 

	# time meausure
	train_net_time = AverageMeter()
	stego_dir = base_dir + '/stego' + str(it)


	# train net and save ckpt
	print("Train net in iteration", it)
	start_train_net = time.time()
	pt_name, accuracy  = test_net.trainNet(stego_dir, it, gpu_num, step2_train_list, step2_test_list, model_dir)
	xq_test_result.append(accuracy)
	print("\tAccuracy:", xq_test_result)
	train_net_time.update(time.time()-start_train_net)
	print("Train net in iteration: {:d}, \n\taccuracy: {:.3f}, \n\ttime: {:.3f}s".format(it, accuracy, train_net_time.val))


	file_log.write('It: ' + str(it) + '\t' + 'Acc: ' + str(accuracy) + '\n')
	file_log.flush()




file_log.close()

print("XQNet Test Accuracy:", xq_test_result)



