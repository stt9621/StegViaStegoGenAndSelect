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


BOSSBASE_COVER_DIR = '/data/stt/steganalysis/BossBase-1.01-cover-resample-256'
BOWS_COVER_DIR = '/data/stt/steganalysis/BOWS2-cover-resample-256'





def genHillCostStego(index_path, temp_stego_dir, temp_cost_dir, payload):

	index_list = np.load(index_path)

	#torch.set_printoptions(edgeitems=3)
	eng = matlab.engine.start_matlab()

	flag = eng.hill_cost_embed(BOSSBASE_COVER_DIR, BOWS_COVER_DIR, temp_stego_dir, temp_cost_dir, payload)

	eng.quit()





def genCandiStegoAndSelect(index_path, stego_dir, cost_dir, params_dir, pre_stego_dir, pre_cost_dir, pre_grad_dir, payload, listnum):

	index_list = np.load(index_path)

	#torch.set_printoptions(edgeitems=3)
	eng = matlab.engine.start_matlab()

	flag = eng.gen_candi_and_select(BOSSBASE_COVER_DIR, BOWS_COVER_DIR, stego_dir, cost_dir, params_dir, pre_stego_dir, pre_cost_dir, pre_grad_dir, payload, listnum)

	eng.quit()



