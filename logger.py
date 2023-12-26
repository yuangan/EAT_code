import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)
    
    def visualize_rec0327(self, bg, out, num):
        image = self.visualizer.visualize0327(bg, out)
        # imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(num).zfill(self.zfill_num)), image)
        return image
    
    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        print('save to: ', cpk_path)
        print('save to: ', (not (os.path.exists(cpk_path) and emergent)))
        # if not (os.path.exists(cpk_path) and emergent):
        torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk_a2kp(checkpoint_path, generator=None, discriminator=None, kp_detector=None, a2kp=None,
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None, optimizer_a2kp=None):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if a2kp is not None and 'audio2kptransformer' in checkpoint.keys():
            a2kp.load_state_dict(checkpoint['audio2kptransformer'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_a2kp is not None and 'optimizer_audio2kptransformer' in checkpoint.keys():
            optimizer_a2kp.load_state_dict(checkpoint['optimizer_audio2kptransformer'])

        return checkpoint['epoch']

    @staticmethod
    def load_cpk_a2kp_diffgen(checkpoint_path, generator=None, discriminator=None, kp_detector=None, a2kp=None,
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None, optimizer_a2kp=None):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint2 = torch.load('./ckpt/000299_1024-checkpoint.pth.tar', map_location='cpu')
        if generator is not None:
            generator.load_state_dict(checkpoint2['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if a2kp is not None and 'audio2kptransformer' in checkpoint.keys():
            a2kp.load_state_dict(checkpoint['audio2kptransformer'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint2['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint2['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint2['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_a2kp is not None and 'optimizer_audio2kptransformer' in checkpoint.keys():
            optimizer_a2kp.load_state_dict(checkpoint['optimizer_audio2kptransformer'])

        return checkpoint['epoch']



    @staticmethod
    def load_cpk_a2kp_prompt_sidetuning(checkpoint_path, generator=None, discriminator=None, kp_detector=None, a2kp=None, emotionprompt=None, sidetuning=None,
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None, optimizer_a2kp=None, optimizer_emotionprompt=None, optimizer_sidetuning=None):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'], strict=False)
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if a2kp is not None and 'audio2kptransformer' in checkpoint.keys():
            a2kp.load_state_dict(checkpoint['audio2kptransformer'], strict=False)
        if emotionprompt is not None and 'emotionprompt' in checkpoint.keys():
                emotionprompt.load_state_dict(checkpoint['emotionprompt'])
        if sidetuning is not None and 'sidetuning' in checkpoint.keys():
            sidetuning.load_state_dict(checkpoint['sidetuning'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                if 'optimizer_discriminator' in checkpoint.keys():
                    optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
                else:
                    checkpoint2 = torch.load('pretrain_453.pth.tar', map_location='cpu')
                    optimizer_discriminator.load_state_dict(checkpoint2['optimizer_discriminator'])
                print('load optimizer_discriminator success')
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_a2kp is not None and 'optimizer_audio2kptransformer' in checkpoint.keys():
            try:
                optimizer_a2kp.load_state_dict(checkpoint['optimizer_audio2kptransformer'])
            except:
                print('load optimizer_a2kp failed, train it from scratch')
        if optimizer_sidetuning is not None and 'optimizer_sidetuning' in checkpoint.keys():
            optimizer_sidetuning.load_state_dict(checkpoint['optimizer_sidetuning'])

        if sidetuning is not None:
            if not 'sidetuning' in checkpoint.keys():
                sidetuning.init_sidetuning_weight(checkpoint['audio2kptransformer'])
                print('init sidetuning success')

        return checkpoint['epoch']


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
            print(self.names)
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out, save_visualize=False):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        # if save_visualize and self.epoch % 10 == 0:
        #     self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []
        bs =  driving.shape[0]
        sync_T = driving.shape[1]
        outmask = False
        
        # Source image with keypoints
        sources=  []
        source = source.data.cpu()
        for i in source:
            sources.append(i.unsqueeze(0).tile(sync_T, 1,1,1))
        source = torch.cat(sources, dim=0)
        # print(source.shape)
        kp_source = out['kp_source']['value'][:, :, :2].data.cpu().numpy()     # 3d -> 2d
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # # Equivariance visualization
        # if 'transformed_frame' in out:
        #     transformed = out['transformed_frame'].data.cpu().numpy()
        #     transformed = np.transpose(transformed, [0, 2, 3, 1])
        #     transformed_kp = out['transformed_kp']['value'][:, :, :2].data.cpu().numpy()   # 3d -> 2d
        #     images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'][:, :, :2].data.cpu().numpy()    # 3d -> 2d
        driving = driving.data.cpu().numpy().reshape(sync_T*bs, 3, 256,256)
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))
        images.append(driving)

        # Result
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)

        ## Occlusion map
        if 'occlusion_map' in out:
            # print(out['occlusion_map'].dtype)
            occlusion_map = out['occlusion_map'].float().data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)
        
        ## Mask
        if 'mask' in out and outmask:
            for i in range(out['mask'].shape[1]):
                mask = out['mask'][:, i:(i+1)].data.cpu().sum(2).repeat(1, 3, 1, 1)    # (n, 3, h, w)
                # mask = F.softmax(mask.view(mask.shape[0], mask.shape[1], -1), dim=2).view(mask.shape)
                mask = F.interpolate(mask, size=source.shape[1:3]).numpy()
                mask = np.transpose(mask, [0, 2, 3, 1])

                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['mask'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))
                
                if i != 0:
                    images.append(mask * color)
                else:
                    images.append(mask)

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image

    def visualize0327(self, driving, out):
        images = []
        outmask = False
        
        # Driving image with keypoints
        kp_driving = out[:, :, :2]   # 3d -> 2d
        driving = driving.reshape(-1, 3, 256,256)
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image


    def visualize_old(self, driving, source, out):
        images = []

        # Source image with keypoints
        source = source.data.cpu()
        kp_source = out['kp_source']['value'][:, :, :2].data.cpu().numpy()     # 3d -> 2d
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Equivariance visualization
        if 'transformed_frame' in out:
            transformed = out['transformed_frame'].data.cpu().numpy()
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = out['transformed_kp']['value'][:, :, :2].data.cpu().numpy()   # 3d -> 2d
            images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'][:, :, :2].data.cpu().numpy()    # 3d -> 2d
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        # Result
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)

        ## Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)
        
        ## Mask
        if 'mask' in out:
            for i in range(out['mask'].shape[1]):
                mask = out['mask'][:, i:(i+1)].data.cpu().sum(2).repeat(1, 3, 1, 1)    # (n, 3, h, w)
                # mask = F.softmax(mask.view(mask.shape[0], mask.shape[1], -1), dim=2).view(mask.shape)
                mask = F.interpolate(mask, size=source.shape[1:3]).numpy()
                mask = np.transpose(mask, [0, 2, 3, 1])

                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['mask'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))
                
                if i != 0:
                    images.append(mask * color)
                else:
                    images.append(mask)

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
    