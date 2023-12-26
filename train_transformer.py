from tqdm import trange, tqdm
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model_transformer import *

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset_transformer25 import DatasetRepeater
import numpy as np

import math
def adjust_learning_rate(optimizer, itr, base_lr, max_itr, p, is_cosine_decay=False, min_lr=1e-6, warm_up_steps=1000, encoder_lr_ratio=1.0, freeze_params=[]):
    if itr < warm_up_steps:
        now_lr = min_lr + (base_lr - min_lr) * itr / warm_up_steps
    else:
        itr = itr - warm_up_steps
        max_itr = max_itr - warm_up_steps
        if is_cosine_decay:
            now_lr = min_lr + (base_lr - min_lr) * (math.cos(math.pi * itr /
                                                             (max_itr + 1)) +
                                                    1.) * 0.5
        else:
            now_lr = min_lr + (base_lr - min_lr) * (1 - itr / (max_itr + 1))**p

    for param_group in optimizer.param_groups:
        if encoder_lr_ratio != 1.0 and "encoder." in param_group["name"]:
            param_group['lr'] = (now_lr - min_lr) * encoder_lr_ratio + min_lr
        else:
            param_group['lr'] = now_lr

        for freeze_param in freeze_params:
            if freeze_param in param_group["name"]:
                param_group['lr'] = 0
                param_group['weight_decay'] = 0
                break
    return now_lr

def train_batch(config, generator, discriminator, kp_detector, audio2kptransformer, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    # optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    # optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    # optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_audio2kptransformer = torch.optim.Adam(audio2kptransformer.parameters(), lr=train_params['lr_audio2kptransformer'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk_a2kp(checkpoint, generator, discriminator, kp_detector, audio2kptransformer,
                                      None, None, None, optimizer_audio2kptransformer)
                                    #   optimizer_generator, optimizer_discriminator, None, None)
                                    #   optimizer_generator, optimizer_discriminator, optimizer_kp_detector, None)
    else:
        start_epoch = 0

    # scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
    #                                   last_epoch=start_epoch - 1)
    # scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
    #                                       last_epoch=start_epoch - 1)
    # scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
    #                                     last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_audio2kptransformer = MultiStepLR(optimizer_audio2kptransformer, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1)
                                        # last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=16, drop_last=True)

    generator_full = GeneratorFullModelBatch(kp_detector, audio2kptransformer, generator, discriminator, train_params, estimate_jacobian=config['model_params']['common_params']['estimate_jacobian'])
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
    count = 0
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in tqdm(dataloader):
                losses_generator, generated = generator_full(x, train_params['train_with_img'])

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator_full.module.audio2kptransformer.parameters(), 0.05)
                # optimizer_generator.step()
                # optimizer_generator.zero_grad()
                # optimizer_kp_detector.step()
                # optimizer_kp_detector.zero_grad()
                optimizer_audio2kptransformer.step()
                optimizer_audio2kptransformer.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    # optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    # optimizer_discriminator.step()
                    # optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                count += 1
                if count % 20 == 0 :
                    print(np.array(logger.loss_list).mean(axis=0))
                logger.log_iter(losses=losses)
                
                if count % 2000 == 0:
                    break
            # scheduler_generator.step()
            # scheduler_discriminator.step()
            # scheduler_kp_detector.step()
            scheduler_audio2kptransformer.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'audio2kptransformer': audio2kptransformer,
                                    #  'optimizer_generator': optimizer_generator,
                                    #  'optimizer_discriminator': optimizer_discriminator,
                                    #  'optimizer_kp_detector': optimizer_kp_detector,
                                     'optimizer_audio2kptransformer': optimizer_audio2kptransformer}, inp=x, out=generated, save_visualize=train_params['train_with_img'])


def train_batch_gen(config, generator, discriminator, kp_detector, audio2kptransformer, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    # optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    # optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    # optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_audio2kptransformer = torch.optim.Adam(audio2kptransformer.parameters(), lr=train_params['lr_audio2kptransformer'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk_a2kp_diffgen(checkpoint, generator, discriminator, kp_detector, audio2kptransformer,
                                      None, None, None, optimizer_audio2kptransformer)
                                    #   optimizer_generator, optimizer_discriminator, None, None)
                                    #   optimizer_generator, optimizer_discriminator, optimizer_kp_detector, None)
    else:
        start_epoch = 0

    # scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
    #                                   last_epoch=start_epoch - 1)
    # scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
    #                                       last_epoch=start_epoch - 1)
    # scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
    #                                     last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_audio2kptransformer = MultiStepLR(optimizer_audio2kptransformer, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1)
                                        # last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=16, drop_last=True)

    generator_full = GeneratorFullModelBatch(kp_detector, audio2kptransformer, generator, discriminator, train_params, estimate_jacobian=config['model_params']['common_params']['estimate_jacobian'])
    # discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        # discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
    count = 0
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in tqdm(dataloader):
                losses_generator, generated = generator_full(x, train_params['train_with_img'])

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator_full.module.audio2kptransformer.parameters(), 0.05)
                # optimizer_generator.step()
                # optimizer_generator.zero_grad()
                # optimizer_kp_detector.step()
                # optimizer_kp_detector.zero_grad()
                optimizer_audio2kptransformer.step()
                optimizer_audio2kptransformer.zero_grad()

                # if train_params['loss_weights']['generator_gan'] != 0:
                #     # optimizer_discriminator.zero_grad()
                #     losses_discriminator = discriminator_full(x, generated)
                #     loss_values = [val.mean() for val in losses_discriminator.values()]
                #     loss = sum(loss_values)

                #     loss.backward()
                #     # optimizer_discriminator.step()
                #     # optimizer_discriminator.zero_grad()
                # else:
                #     losses_discriminator = {}

                # losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                count += 1
                if count % 100 == 0 :
                    print(np.array(logger.loss_list).mean(axis=0))
                
                logger.log_iter(losses=losses)
                
                if count % 2000 == 0:
                    break

            # scheduler_generator.step()
            # scheduler_discriminator.step()
            # scheduler_kp_detector.step()
            scheduler_audio2kptransformer.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                    #  'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'audio2kptransformer': audio2kptransformer,
                                    #  'optimizer_generator': optimizer_generator,
                                    #  'optimizer_discriminator': optimizer_discriminator,
                                    #  'optimizer_kp_detector': optimizer_kp_detector,
                                     'optimizer_audio2kptransformer': optimizer_audio2kptransformer}, inp=x, out=generated, save_visualize=train_params['train_with_img'])


import time
def train_batch_deepprompt_eam3d_sidetuning(config, generator, discriminator, kp_detector, audio2kptransformer, emotionprompt, sidetuning, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    # optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    # optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_audio2kptransformer = torch.optim.Adam(audio2kptransformer.parameters(), lr=train_params['lr_audio2kptransformer'], betas=(0.5, 0.999))
    optimizer_emotionprompt = torch.optim.Adam(emotionprompt.parameters(), lr=train_params['lr_emotionprompt'], betas=(0.5, 0.999))
    optimizer_sidetuning = torch.optim.Adam(sidetuning.parameters(), lr=train_params['lr_sidetuning'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk_a2kp_prompt_sidetuning(checkpoint, generator, discriminator, kp_detector, audio2kptransformer, emotionprompt, sidetuning,
                                      None, None, None, optimizer_audio2kptransformer, optimizer_emotionprompt, optimizer_sidetuning)
                                    #   optimizer_generator, optimizer_discriminator, None, None)
                                    #   optimizer_generator, optimizer_discriminator, optimizer_kp_detector, None)
    else:
        start_epoch = 0
    # print( train_params['epoch_milestones']) # 180
    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch= - 1)
    # scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
    #                                       last_epoch=start_epoch - 1)
    # scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
    #                                     last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_audio2kptransformer = MultiStepLR(optimizer_audio2kptransformer, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1)
    scheduler_emotionprompt = MultiStepLR(optimizer_emotionprompt, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1)
    scheduler_sidetuning = MultiStepLR(optimizer_sidetuning, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1)
                                        # last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=16, drop_last=True)

    generator_full = GeneratorFullModelBatchDeepPromptSTEAM3D(kp_detector, audio2kptransformer, emotionprompt, sidetuning, generator, discriminator, train_params, estimate_jacobian=config['model_params']['common_params']['estimate_jacobian'])

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
    count = 0

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            time_start = time.time()
            for x in tqdm(dataloader):
                losses_generator, generated = generator_full(x, train_params['train_with_img'])

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator_full.module.audio2kptransformer.parameters(), 0.05)
                torch.nn.utils.clip_grad_norm_(generator_full.module.sidetuning.parameters(), 0.05)
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                # optimizer_kp_detector.step()
                # optimizer_kp_detector.zero_grad()
                optimizer_audio2kptransformer.step()
                optimizer_audio2kptransformer.zero_grad()

                optimizer_emotionprompt.step()
                optimizer_emotionprompt.zero_grad()

                optimizer_sidetuning.step()
                optimizer_sidetuning.zero_grad()

                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                count += 1
                if count % 100 == 0 :
                    print(np.array(logger.loss_list).mean(axis=0), count%len(dataloader), len(dataloader))
                logger.log_iter(losses=losses)
                
                time_now = time.time()
                if(time_now - time_start) > 30*60: # 30 min
                    break
                
                ### check learning rate 
                # print(optimizer_audio2kptransformer.param_groups[0]['lr'])
                # adjust_learning_rate(optimizer_emotionprompt, count, train_params['lr_emotionprompt'], 100000, 2)
                # print(optimizer_emotionprompt.param_groups[0]['lr'])
                # print(scheduler_audio2kptransformer.get_lr())
                # print(scheduler_emotionprompt.get_lr())
            scheduler_generator.step()
            # scheduler_discriminator.step()
            # scheduler_kp_detector.step()
            scheduler_audio2kptransformer.step()
            scheduler_emotionprompt.step()
            scheduler_sidetuning.step()
            
            

            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'audio2kptransformer': audio2kptransformer,
                                     'emotionprompt': emotionprompt,                                     
                                     'sidetuning': sidetuning,                                     
                                     'optimizer_generator': optimizer_generator,
                                    #  'optimizer_discriminator': optimizer_discriminator,
                                    #  'optimizer_kp_detector': optimizer_kp_detector,
                                     'optimizer_audio2kptransformer': optimizer_audio2kptransformer,
                                     'optimizer_emotionprompt': optimizer_emotionprompt,
                                     'optimizer_sidetuning': optimizer_sidetuning}, inp=x, out=generated, save_visualize=train_params['train_with_img'])

