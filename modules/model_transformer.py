#### Training code will be released after cleaning
#### If you have any question, please contact ganyuan@zju.edu.cn for more details. I will reply as soon as possible.

from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid_2d
from torchvision import models
import numpy as np
from torch.autograd import grad
import modules.hopenet as hopenet
from torchvision import transforms

from scipy.spatial import ConvexHull

from torchvision import transforms
data_transforms = transforms.Compose([
                                    transforms.Resize(size=(224, 224)),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
data_transforms_clip = transforms.Compose([
                                    transforms.Resize(size=(224, 224)),
                                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
                                ])

from modules.bilinear import crop_bbox_batch
from modules.syncnet import SyncNet_color as SyncNet

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict

class DownSample(torch.nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.down_pose = AntiAliasInterpolation2d(1,0.25)

    def forward(self, x):
        return self.down_pose(x)

class Transform:
    """
    Random tps transformation for equivariance constraints.
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid_2d((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree

'''
# beta version
def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    roll_mat = torch.cat([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll), 
                          torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
                          torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    pitch_mat = torch.cat([torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch), 
                           torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
                           -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),  
                         torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
                         torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', roll_mat, pitch_mat, yaw_mat)

    return rot_mat
'''

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)

    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, estimate_jacobian=True, dkc=None):
    kp = kp_canonical['value']    # (bs, k, 3)
    yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
    t, exp = he['t'], he['exp']
    # print(yaw.shape) # len 66

    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)
    
    # keypoint rotation
    if dkc == None:
        kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)
    else:
        bsize = rot_mat.shape[0]
        kp_new = dkc.reshape(bsize, 15, 3)+kp.tile(bsize,1,1)
        kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)
        # print('kp_new ', kp_new.shape) # b 15 3
        # print('rot ', rot_mat.shape) # b 3 3
        # print('kp_rotated ', kp_rotated.shape) # b 15 3
        
    # keypoint translation
    t = t.unsqueeze_(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    if estimate_jacobian:
        jacobian = kp_canonical['jacobian']   # (bs, k ,3, 3)
        jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)
    else:
        jacobian_transformed = None

    return {'value': kp_transformed, 'jacobian': jacobian_transformed}


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        print(m)
        m.eval()
    # if('weight_u' in m._buffers): # spectral norm eval
    #     print('set spectral norm, ', m)
    #     m.eval()
    # if m.find('decoder.G_middle') != -1 or m.find('decoder.up') != -1:
    #     m.eval()


class GeneratorFullModelBatchDeepPromptSTEAM3D(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, audio2kptransformer, emotionprompt, sidetuning, generator, discriminator, train_params, estimate_jacobian=True):
        super(GeneratorFullModelBatchDeepPromptSTEAM3D, self).__init__()
        self.kp_extractor = kp_extractor
        self.kp_extractor.eval()
        self.audio2kptransformer = audio2kptransformer
        self.emotionprompt = emotionprompt
        self.sidetuning = sidetuning
        self.generator = generator
        # self.generator.eval()
        # self.audio2kptransformer.eval()
        self.generator.apply(set_bn_eval)
        # self.generator.decoder.eval()
        # self.generator.dense_motion_network.eval()
        # self.generator.eval()
        self.audio2kptransformer.apply(set_bn_eval)
        # self.discriminator = discriminator
        # self.discriminator.eval()
        self.train_params = train_params
        self.scales = train_params['scales']
        # self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.estimate_jacobian = estimate_jacobian

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

        if self.loss_weights['headpose'] != 0:
            self.hopenet = hopenet.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)
            print('Loading hopenet')
            hopenet_state_dict = torch.load(train_params['hopenet_snapshot'])
            self.hopenet.load_state_dict(hopenet_state_dict)
            if torch.cuda.is_available():
                self.hopenet = self.hopenet.cuda()
                self.hopenet.eval()
        if self.loss_weights['dan'] != 0:
            self.dan = DAN(num_head=4, num_class=8, pretrained=False)
            dan_state_dict = torch.load(train_params['dan_snapshot'])
            self.dan.load_state_dict(dan_state_dict['model_state_dict'])
            if torch.cuda.is_available():
                self.dan = self.dan.cuda()
                self.dan.eval()
        if self.loss_weights['lmk'] != 0:
            self.fan = torch.jit.load('/root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip')
            if torch.cuda.is_available():
                self.fan = self.fan.cuda()
                self.fan.eval()

        if self.loss_weights['sync'] != 0:
            # 2D face bbox detection based on face_alignment from wav2lip
            # self.fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda:0'),\
            #            face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda:1'), \
            #            face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda:2'), \
            #            face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda:3')]
            self.syncnet = SyncNet()
            syncnet_state = torch.load('./ckpt/checkpoint_epoch02600_testloss0.8152778898516009_testacc0.5245564512668117.pth')
            s = syncnet_state["state_dict"]
            self.syncnet.load_state_dict(s)
            for p in self.syncnet.parameters():
                p.requires_grad = False
            if torch.cuda.is_available():
                self.syncnet = self.syncnet.cuda()
                self.syncnet.eval()
            self.logloss = nn.BCELoss()

        if self.loss_weights['clip'] != 0:
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14")
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
                # self.text_token = self.text_token.cuda()
                self.clip_model.eval()
            self.text_features = self.get_emotoken_12()

        self.freeze_param()

        self.expU = torch.from_numpy(np.load('./expPCAnorm_fin/U_mead.npy')[:,:32])
        self.expmean = torch.from_numpy(np.load('./expPCAnorm_fin/mean_mead.npy'))


    def freeze_param(self):
        """Set requires_grad=False for each of model.parameters()"""
        if sum(self.loss_weights['perceptual']) != 0:
            for param in self.vgg.parameters():
                param.requires_grad = False
        for param in self.kp_extractor.parameters():
            param.requires_grad = False
        if self.loss_weights['clip'] != 0:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def get_sync_loss(self, mel, g, device):
        def cosine_loss(a, v, y):
            d = F.cosine_similarity(a, v)
            with torch.cuda.amp.autocast(enabled=False):
                loss = self.logloss(d.unsqueeze(1), y)
            return loss

        g = g[:, :, :, g.size(3)//2:]
        g = torch.cat([g[:, :, i] for i in range(g.size(2))], dim=1)
        # B, 3 * T, H//2, W
        a, v = self.syncnet(mel, g)
        y = torch.ones(g.size(0), 1).float().to(device)
        return cosine_loss(a, v, y)


    def forward(self, x, train_with_img=False):
        bbs, bs, _ = x['he_driving']['yaw'].shape # bbs , bs, 66

        with torch.no_grad():
            kp_canonical = self.kp_extractor(x['source'], with_feature=True)     # {'value': value, 'jacobian': jacobian}   
        kp_cano = kp_canonical['value']
        he_source = x['he_source']

        ### emotion prompt
        emoprompt, deepprompt = self.emotionprompt(x)
        # print(emoprompt.shape)
        ### a2kp
        he_driving_emo, input_st = self.audio2kptransformer(x, kp_canonical, emoprompt=emoprompt, deepprompt=deepprompt, side=True)           # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
        emo_exp = self.sidetuning(input_st, emoprompt, deepprompt)
        he_driving = x['he_driving']
        exp = he_driving_emo['emo']
        device = exp.get_device()



        ### transfer pca exp to kp exp
        exp = torch.mm(exp, self.expU.t().to(device))
        exp = exp + self.expmean.expand_as(exp).to(device)
        exp = exp.reshape(bbs, bs, 45) # 45: 15 point * 3 xyz

        ### add emotional expression
        emo_exp = emo_exp.reshape(bbs, bs, 45)
        exp = exp+emo_exp

        source_areas = []
        kp_driving = []
        for i in range(bbs):
            source_area = ConvexHull(kp_cano[i].cpu().numpy()).volume
            source_areas.append(source_area)
            exp[i] = exp[i] * source_area
            kp_canonical_i = {'value': kp_cano[i].unsqueeze(0)}
            he_new_driving_i = {'yaw': he_driving['yaw'][i], 'pitch':he_driving['pitch'][i], 'roll':he_driving['roll'][i], 't':he_driving['t'][i], 'exp': exp[i]}
            kp_driving.append(keypoint_transformation(kp_canonical_i, he_new_driving_i, self.estimate_jacobian)['value'])

        he_new_driving = {'yaw': he_driving['yaw'], 'pitch':he_driving['pitch'], 'roll':he_driving['roll'], 't':he_driving['t'], 'exp': exp}

        
        loss_values = {}
        generated = {}

        ### generate image batchize
        if train_with_img:
            # he_driving = self.audio2kptransformer(x['driving'])      # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}

            # {'value': value, 'jacobian': jacobian}
            # print(kp_canonical['value'].shape) # 1 15 3
            
            ### Note: batch keypoint transformation operation is equal to 'for' iteration in every he source
            kp_source = keypoint_transformation(kp_canonical, he_source, self.estimate_jacobian)
            bbs = kp_canonical['value'].shape[0]
            bs = kp_driving[0].shape[0]
            rep = torch.LongTensor([bs]*bbs).to(device)
            kp_source['value'] = torch.repeat_interleave(kp_source['value'], rep, dim=0)
            # print(kp_source['value'], kp_source['value'].shape)
            source = torch.repeat_interleave(x['source'], rep, dim=0)
            kp_driving_bbs = {'value': torch.cat(kp_driving, dim = 0)}
            # with torch.no_grad():
            generated = self.generator(source, kp_source=kp_source, kp_driving=kp_driving_bbs, prompt=emoprompt)
            generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
            x['driving'] = x['driving'].reshape([bbs*bs, 3, 256, 256])

            if self.loss_weights['perceptual'] !=0 or self.loss_weights['generator_gan'] != 0:
                pyramide_real = self.pyramid(x['driving'])
                pyramide_generated = self.pyramid(generated['prediction'])

            if sum(self.loss_weights['perceptual']) != 0:
                value_total = 0
                for scale in self.scales:
                    x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                    y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                    for i, weight in enumerate(self.loss_weights['perceptual']):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

            if self.loss_weights['generator_gan'] != 0:
                discriminator_maps_generated = self.discriminator(pyramide_generated)
                discriminator_maps_real = self.discriminator(pyramide_real)
                value_total = 0
                for scale in self.disc_scales:
                    key = 'prediction_map_%s' % scale
                    if self.train_params['gan_mode'] == 'hinge':
                        value = -torch.mean(discriminator_maps_generated[key])
                    elif self.train_params['gan_mode'] == 'ls':
                        value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                    else:
                        raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

                    value_total += self.loss_weights['generator_gan'] * value
                loss_values['gen_gan'] = value_total

                if sum(self.loss_weights['feature_matching']) != 0:
                    value_total = 0
                    for scale in self.disc_scales:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                            if self.loss_weights['feature_matching'][i] == 0:
                                continue
                            value = torch.abs(a - b).mean()
                            value_total += self.loss_weights['feature_matching'][i] * value
                        loss_values['feature_matching'] = value_total

            if self.loss_weights['keypoint'] != 0:
                # print(kp_driving['value'].shape)     # (bs, k, 3)
                value_total = 0
                for i in range(kp_driving['value'].shape[1]):
                    for j in range(kp_driving['value'].shape[1]):
                        dist = F.pairwise_distance(kp_driving['value'][:, i, :], kp_driving['value'][:, j, :], p=2, keepdim=True) ** 2
                        dist = 0.1 - dist      # set Dt = 0.1
                        dd = torch.gt(dist, 0) 
                        value = (dist * dd).mean()
                        value_total += value

                kp_mean_depth = kp_driving['value'][:, :, -1].mean(-1)
                value_depth = torch.abs(kp_mean_depth - 0.33).mean()          # set Zt = 0.33

                value_total += value_depth
                loss_values['keypoint'] = self.loss_weights['keypoint'] * value_total

            if self.loss_weights['headpose'] != 0:
                transform_hopenet =  transforms.Compose([transforms.Resize(size=(224, 224)),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                driving_224 = transform_hopenet(x['driving'])

                yaw_gt, pitch_gt, roll_gt = self.hopenet(driving_224)
                yaw_gt = headpose_pred_to_degree(yaw_gt)
                pitch_gt = headpose_pred_to_degree(pitch_gt)
                roll_gt = headpose_pred_to_degree(roll_gt)

                yaw, pitch, roll = he_driving['yaw'], he_driving['pitch'], he_driving['roll']
                yaw = headpose_pred_to_degree(yaw)
                pitch = headpose_pred_to_degree(pitch)
                roll = headpose_pred_to_degree(roll)

                value = torch.abs(yaw - yaw_gt).mean() + torch.abs(pitch - pitch_gt).mean() + torch.abs(roll - roll_gt).mean()
                loss_values['headpose'] = self.loss_weights['headpose'] * value

            if self.loss_weights['expression'] != 0:
                value = torch.norm(he_driving['exp'], p=1, dim=-1).mean()
                loss_values['expression'] = self.loss_weights['expression'] * value

            if self.loss_weights['clip'] != 0:
                processed_img = data_transforms_clip(generated['prediction'])
                y_trg = x['y_trg']
                # print(self.clip_model(processed_img, self.text_token)[0].shape)
                image_features = self.clip_model.encode_image(processed_img)
                similarity = 1 - self.clip_model(image_features, self.text_features.to(device).detach())[0][:, y_trg] / 100
                # print(similarity[0])
                value = similarity.mean()
                loss_values['clip'] = self.loss_weights['clip'] * value

            if self.loss_weights['sync'] != 0 or self.loss_weights['img_l1'] != 0:
                ### +++ sync loss +++
                # 1. detect bbox of driving images.
                # preds = self.fa[device].get_detections_for_batch(x['driving'].permute(0,2,3,1).cpu().numpy()[...,::-1]*255).detach()
                preds = x['bboxs'].reshape(bbs*bs, 4)

                # if not None in preds:
                # preds = torch.from_numpy(np.array(preds)).to(device)/256.
                preds = preds.to(device)/256.
                box_to_feat = torch.from_numpy(np.array([i for i in range(bbs*bs)]))
                # 2. use F.grid_sample to crop and resize the generated image and driving image
                gt_bbox = crop_bbox_batch(x['driving'], preds, box_to_feat, 96)
                pre_bbox = crop_bbox_batch(generated['prediction'], preds, box_to_feat, 96)
                ### 2.5 save image  to check the crop result is right (important sometime, leave it here for debug)
                # from cv2 import imwrite
                # for i, gt in enumerate(gt_bbox):
                #     imwrite(f'./tmp/gtcrop_{i}.jpg', gt_bbox[i].permute(1,2,0).cpu().numpy()[..., ::-1]*255)
                #     imwrite(f'./tmp/predcrop_{i}.jpg', pre_bbox[i].permute(1,2,0).cpu().numpy()[..., ::-1]*255)

                # 3. loss of expert discriminator from wav2lip (hope it work)
                # l1 loss
                if self.loss_weights['img_l1'] != 0:
                    loss_values['img_l1'] = self.loss_weights['img_l1'] * torch.abs(gt_bbox-pre_bbox).mean()
                
                # 4. sync loss
                # pre_bbox: bbs*bs, 3, 96(H), 96(W) -> bbs, 3, bs(sync_T), H, W <Refer to wav2lip>
                if self.loss_weights['sync'] != 0:
                    pre_bbox = pre_bbox.reshape(bbs, bs, 3, 96, 96).permute(0, 2, 1, 3, 4)
                    value = self.get_sync_loss(x['sync_mel'], pre_bbox, device).mean()
                    loss_values['sync'] = self.loss_weights['sync'] * value
                # else:
                #     ### detect failed
                #     loss_values['img_l1'] = torch.abs(x['driving']-generated['prediction']).mean()
                #     loss_values['sync'] = torch.FloatTensor([0]).to(device).mean()

        if self.loss_weights['latent'] != 0:
            loss_values['latent_emo'] = self.loss_weights['latent'] * F.mse_loss(he_new_driving['exp'], he_driving['exp'].squeeze(0))
            # loss_values['latent_emo_5678'] = self.loss_weights['latent'] * torch.abs(he_new_driving['exp'][:, :, 15:27] - he_driving['exp'][:,:,15:27])
            # loss_values['latent_emo_5678y'] = self.loss_weights['latent']* 10 * torch.abs(he_new_driving['exp'].reshape(bbs, bs, 15, 3)[:,:, 5:8, 1] - \
            #                                                                              he_driving['exp'].reshape(bbs, bs, 15, 3)[:,:, 5:8, 1])

        if self.loss_weights['pca'] != 0:
            pca_exps = []
            for i in range(bbs):
                pca_exps.append(he_driving['exp'][i]/source_areas[i])
            pca_exps = torch.cat(pca_exps, 0)
            pca_exps = torch.mm(pca_exps - self.expmean.expand_as(pca_exps).to(device), self.expU.to(device))
            loss_values['pca_emo'] = self.loss_weights['pca'] * F.mse_loss(he_driving_emo['emo'], pca_exps)
        
        if self.loss_weights['dan'] != 0:
            img = data_transforms(generated['prediction'])
            gt_img = data_transforms(x['driving'])
            out_fake, fea_fake, heads_fake = self.dan(img)
            out_gt, fea_gt, heads_gt = self.dan(gt_img)
            loss_values['dan1'] = self.loss_weights['dan'] * torch.abs(fea_fake - fea_gt.detach()).mean()
            loss_values['dan2'] = self.loss_weights['dan'] * torch.abs(heads_fake - heads_gt.detach()).mean()
            loss_values['dan3'] = self.loss_weights['dan'] * F.mse_loss(out_fake, out_gt.detach())

        if self.loss_weights['lmk'] != 0:
            heatmap_lmk_fake = self.fan(generated['prediction']) # bs 68 64 64
            heatmap_lmk_gt = self.fan(x['driving'])
            value = torch.abs(heatmap_lmk_fake - heatmap_lmk_gt).mean()
            loss_values['lmk'] = self.loss_weights['lmk'] * value

        # if 'kp' in self.loss_weights.keys():
        #     if self.loss_weights['kp'] != 0:
        #         loss_values['kp'] = 0
        #         for i in range(bbs):
        #             he_driving_gt = {'yaw': he_driving['yaw'][i], 'pitch':he_driving['pitch'][i], 'roll':he_driving['roll'][i], 't':he_driving['t'][i], 'exp': he_driving['exp'][i]}
        #             kp_canonical_i = {'value': kp_cano[i].unsqueeze(0)}
        #             kp_driving_gt = keypoint_transformation(kp_canonical_i, he_driving_gt, self.estimate_jacobian)
        #             loss_values['kp'] = loss_values['kp'] + self.loss_weights['kp'] * torch.abs(kp_driving[i]['value'][:,5:9,:] - kp_driving_gt['value'][:,5:9,:].detach()).mean()
        #         loss_values['kp'] = loss_values['kp']/bbs

        return loss_values, generated

    def get_emotoken_12(self):
        text_anger = ['very angry.',
                        'A angry looking.',
                        'the person seems angry.',
                        'He or she seems to be angrily talking.',
                        'talking in a very angry expression.',
                        'A person who seems to be very angry.',
                        'The people seems to be angry and he or she is talking.',
                        'He or she is angry with brows furrowed.',
                        'The person looks angry.',
                        'an angry expression.',
                        'Brows furrowed, eyes wide, lips tightened and pressed together',
                        'The person is talking angrily.']
        text_contempt = ['very contemptuous.',
                        'A contemptuous looking.',
                        'the person seems contemptuous.',
                        'He or she seems to be contemptuously talking.',
                        'talking in a very contemptuous expression.',
                        'A person who seems to be very contemptuous.',
                        'The people seems to be contemptuous and he or she is talking.',
                        'He or she is contemptuous with simle.',
                        'The person looks contemptuous.',
                        'a contemptuous expression.',
                        'Smile, eyelids drooping',
                        'The person is talking contemptuously.']
        text_disgust = ['very disgusted.',
                        'A disgusted looking.',
                        'the person seems disgusted.',
                        'He or she seems to be disgustedly talking.',
                        'talking in a expression of disgust.',
                        'A person who seems to be very disgusted.',
                        'The people seems to be disgusted and he or she is talking.',
                        'He or she is disgusted.',
                        'The person looks disgusted.',
                        'an expression of disgust.',
                        'Eyes narrowed, nose wrinkled, lips parted, jaw dropped, tongue show',
                        'The person is talking disgustedly.']
        text_fear = ['very fearful.',
                    'A fear looking.',
                    'the person seems fearful.',
                    'He or she seems to be fearfully talking.',
                    'talking in a very fearful expression.',
                    'A person who seems to be very fearful.',
                    'The people seems to be fearful and he or she is talking.',
                    'He or she is fearful with brows furrowed.',
                    'The person looks fearful.',
                    'a fearful expression.',
                    'Eyebrows raised and pulled together, upper eyelid raised, lower eyelid tense, lips parted and stretched',
                    'The person is talking fearfully.']
        text_happy = ['very happy.',
                    'A happy looking.',
                    'the person seems happy as smiling with teeth showing.',
                    'He or she seems to be happily smiling.',
                    'talking in a very happy expression.',
                    'A person who seems to be very happy with a wide and toothy smile.',
                    'The people seems to be happy and he or she is talking.',
                    'He or she is smiling happily.',
                    'The person looks happy',
                    'a happy smiling expression.',
                    'Duchenne display.',
                    'The person is talking happily.']
        text_neutral = ['very neutral.',
                        'A neutral looking.',
                        'the person seems neutral.',
                        'He or she seems to be neutrally talking.',
                        'talking in a very neutral expression.',
                        'A person who seems to be very neutral.',
                        'The people seems to be neutral and he or she is talking.',
                        'He or she is neutral.',
                        'The person looks calm.',
                        'a neutral expression.',
                        'The person is talking with neutral expression.']
        text_sad = ['very sad.',
                    'A sad looking.',
                    'the person seems sad.',
                    'He or she seems to be sadly talking.',
                    'talking in a very sad expression.',
                    'A person who seems to be very sad.',
                    'The people seems to be sad and he or she is talking.',
                    'He or she is crying sadly.',
                    'The person looks sad.',
                    'a sad expression.',
                    'Brows knitted, eyes slightly tightened, lip corners depressed, lower lip raised.',
                    'The person is talking sadly.']
        text_surprise = ['very surprised.',
                        'A surprised looking.',
                        'the person seems surprised.',
                        'He or she seems to be surprisedly talking.',
                        'talking in a very surprised expression.',
                        'A person who seems to be very surprised.',
                        'The people seems to be surprised and he or she is talking.',
                        'He or she is surprised with eyebrows raised.',
                        'The person looks surprised.',
                        'an surprised expression.',
                        'Eyebrows raised, upper eyelid raised, lips parted, jaw dropped',
                        'The person is talking surprisedly.']
        ts4 = [text_anger, text_contempt, text_disgust, text_fear, text_happy, text_neutral, text_sad, text_surprise]
        text_features = []
        with torch.no_grad():
            for text in ts4:
                token = clip.tokenize(text).cuda()
                feature = self.clip_model.encode_text(token)
                text_features.append(torch.mean(feature,0,True))
        text_features = torch.cat(text_features, dim=0)
        return text_features
