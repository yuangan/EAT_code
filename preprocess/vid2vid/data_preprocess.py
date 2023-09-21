### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from tqdm import tqdm

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
if opt.dataset_mode == 'temporal':
    opt.dataset_mode = 'test'

opt.dataroot = '../video_fps25/'
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
input_nc = 1 if opt.label_nc != 0 else opt.input_nc

save_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
for i, data in tqdm(enumerate(dataset)):
    continue
