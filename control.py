import os
import argparse
import time
import numpy as np
import scipy.io as sio
import matlab.engine
import shutil

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

import gen_stego
import train_net
import get_gradient
import test
import train_net_from_scratch



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
		'-ln',
		'--listnum',
		help='Determine the list num to run',
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
list_num = int(args.listnum)


# list & path
ALL_LIST = './index_list/' + str(list_num) + '/all_list.npy'
STEP1_LIST = './index_list/' + str(list_num) + '/train_and_val_list.npy'
STEP2_LIST = './index_list/' + str(list_num) + '/test_list.npy'
STEP1_TRAIN_LIST = './index_list/' + str(list_num) + '/train_list.npy'
STEP1_VAL_LIST = './index_list/' + str(list_num) + '/val_list.npy'
STEP2_TRAIN_LIST = './index_list/' + str(list_num) + '/retrain_train_list.npy'
STEP2_TEST_LIST = './index_list/' + str(list_num) + '/retrain_test_list.npy'


des_dir = '/data/stt/adv_spa_p0.4'


output_dir = './adv-spa-p.4'
print("output_dir:", output_dir)
log_path = output_dir + '/val_log.txt'


if (os.path.exists(des_dir) == False):
	os.mkdir(des_dir)
if (os.path.exists(output_dir) == False):
	os.mkdir(output_dir)







file_log = open(log_path, 'a')




###############################################################################################################################
###############################################1111111111111111111111111111####################################################

iteration = 0
if iteration == 0:

	# time meausure
	gen_stego_time = AverageMeter()
	train_net_time = AverageMeter()
	gen_grad_time = AverageMeter()

	stego_dir = des_dir + '/stego' + str(iteration)
	cost_dir = des_dir + '/cost' + str(iteration)
	grad_dir = des_dir + '/grad'





	print("\n------------------------------------------------------------------------------")
	print("------------------------------- ITERATION:", iteration, "-------------------------------")

	
	
	# calculate stego and save cost for next iteration
	print("1 - Calculate stego & save cost and stego")
	start_gen_stego = time.time()
	gen_stego.genHillCostStego(ALL_LIST, stego_dir, cost_dir, payload)
	gen_stego_time.update(time.time()-start_gen_stego)
	print("1 - Calculate stego & save cost and stego: {:.3f}s".format(gen_stego_time.val))
	



	# train net and save ckpt
	print("2 - Train net in iteration", iteration)
	start_train_net = time.time()
	pt_name, it0_accuracy = train_net.trainNet(output_dir, stego_dir, iteration, gpu_num, STEP1_TRAIN_LIST, STEP1_VAL_LIST, STEP2_LIST, fiter=True)
	print("\tIteration 0 DengNet Accuracy:", it0_accuracy)
	train_net_time.update(time.time()-start_train_net)
	print("2 - Train net in iteration: {:d}, \n\taccuracy: {:.3f}, \n\ttime: {:.3f}s".format(iteration, it0_accuracy, train_net_time.val))

	
	#pt_name = './adv-spa-bl-p.4/0-params.pt'


	# calculate and save gradient
	print("3 - Calculate and save gradient")
	start_gen_grad = time.time()
	get_gradient.calSignGrad(pt_name, ALL_LIST, grad_dir, gpu_num)
	gen_grad_time.update(time.time()-start_gen_grad)
	print("3 - Calculate and save gradient: {:.3f}s".format(gen_grad_time.val))

	

	file_log.write('It: ' + str(iteration) + '\t' + 'Acc: ' + str(it0_accuracy) + '\n')
	file_log.flush()
	iteration = iteration + 1




###############################################################################################################################
###############################################2222222222222222222222222222####################################################




it = 1
if it == T: 

	# time meausure
	gen_stego_time = AverageMeter()
	train_net_time = AverageMeter()
	test_time = AverageMeter()
	select_stego_time = AverageMeter()

	stego_dir = des_dir + '/stego' + str(it)
	cost_dir = des_dir + '/cost' + str(it)
	params_dir = des_dir + '/params'
	


	pre_grad_dir = des_dir + '/grad'
	pre_stego_dir = des_dir + '/stego' + str(it-1)
	pre_cost_dir = des_dir + '/cost' + str(it-1)




	print("------------------------------------------------------------------------------")
	print("------------------------------ ITERATION:", it, "------------------------------")


	# calculate stego and save cost for next iteration
	print("1 - Calculate stego & save cost and stego")
	start_gen_stego = time.time()
	gen_stego.genCandiStegoAndSelect(STEP2_LIST, stego_dir, cost_dir, params_dir, pre_stego_dir, pre_cost_dir, pre_grad_dir, payload, list_num)
	gen_stego_time.update(time.time()-start_gen_stego)
	print("1 - Calculate stego & save cost and stego: {:.3f}s".format(gen_stego_time.val))



		

	# test model used adv samples
	print("2 - Test model with adv stego")
	start_test = time.time()
	adv_acc = test.test(STEP2_LIST, stego_dir, output_dir, it, gpu_num)
	print("\tIteration 1 Before Retrain DengNet Accuracy:", adv_acc)
	test_time.update(time.time()-start_test)
	print("2 - Test model with adv stego, accuracy: {:.3f}, \n\ttime: {:.3f}s".format(adv_acc, test_time.val))




	# train net and save ckpt
	print("3 - Retrain net in iteration", it)
	start_train_net = time.time()
	pt_name, retrain_accuracy = train_net_from_scratch.trainNet(output_dir, stego_dir, it, gpu_num, STEP2_TRAIN_LIST, STEP2_TEST_LIST, fiter=False)
	print("\tIteration 1 After Retrain DengNet Accuracy:", retrain_accuracy)
	train_net_time.update(time.time()-start_train_net)
	print("3 - Retrain net in iteration: {:d}, \n\taccuracy: {:.3f}, \n\ttime: {:.3f}s".format(it, retrain_accuracy, train_net_time.val))



	file_log.write('It: ' + str(it) + '\t' + 'Acc: ' + str(retrain_accuracy) + '\n')
	file_log.flush()







file_log.close()


print("Iteration 0 DengNet Accuracy:", it0_accuracy)
print("Iteration 1 Before Retrain DengNet Accuracy:", adv_acc)
print("Iteration 1 After Retrain DengNet Accuracy:", retrain_accuracy)


