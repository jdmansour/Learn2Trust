import torch
import matplotlib
import numpy as np
import torch.nn.functional as F


def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice


def cat_images(input,clip=False,size=256):
    if(clip):
        input1 = torch.pow(torch.clamp(input*1000+1500,0,2200)/2200,0.6)
    else:
        input1 = input*1+0
    imcat = torch.cat((input1[:4].view(-1,size),input1[4:8].view(-1,size)),1)
    imcat = torch.cat((imcat[:size*2,:],imcat[size*2:,:]),1)
    return imcat


def color_rgb(image,segment):
    cmap = matplotlib.cm.get_cmap('Set1')
    colors = torch.cat((torch.zeros(1,3),torch.from_numpy(cmap(np.linspace(0,1,9))[:,:3]).float()),0)
    colors = colors[torch.tensor([0,2,7,8,1,3]).long(),:]
    seg_rgb = colors[segment]

    img_rgb = image.unsqueeze(2).repeat(1,1,3)
    seg_rgb.view(-1,3)[segment.view(-1)==0,:] = img_rgb.view(-1,3)[segment.view(-1)==0,:]
    return seg_rgb


def get_img_batch(imgs_orig, idx):
    bspline = F.interpolate(F.avg_pool2d(F.avg_pool2d(torch.randn(16,2,32,32),7,stride=1,padding=3),7,stride=1,padding=3),scale_factor=8,mode='bilinear')
    grid = .15*bspline.permute(0,2,3,1) + F.affine_grid(torch.eye(2,3).unsqueeze(0)+.06*torch.randn(16,2,3),(16,1,256,256))
    augment = torch.rand(16,3,1,1,1)*1.4
    img_aug = torch.pow(torch.clamp(imgs_orig[idx],0,2500)/2500,augment[:,0]+.3)*(3.8+augment[:,1])-.8-augment[:,2]
    img_batch = F.grid_sample(img_aug,grid.float(),mode='bilinear')
    return img_batch


def get_seg_batch(segs, idx):
    bspline = F.interpolate(F.avg_pool2d(F.avg_pool2d(torch.randn(16,2,32,32),7,stride=1,padding=3),7,stride=1,padding=3),scale_factor=8,mode='bilinear')
    grid = .15*bspline.permute(0,2,3,1) + F.affine_grid(torch.eye(2,3).unsqueeze(0)+.06*torch.randn(16,2,3),(16,1,256,256))
    seg_batch = F.grid_sample(segs[idx].float().unsqueeze(1),grid.float(),mode='nearest').squeeze().long()
    return seg_batch


def aug_img_and_seg(imgs, segs, strength_aug_finetune=0.1):
    grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0) + strength_aug_finetune * torch.randn(imgs.shape[0], 2, 3),
                         (imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]))
    imgs_aug = F.grid_sample(imgs, grid)
    segs_aug = F.grid_sample(segs.unsqueeze(1).float(), grid, mode='nearest').long()
    return imgs_aug, segs_aug.squeeze()
