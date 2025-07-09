import os
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
import SimpleITK as sitk
import time
import torch
from torch.utils.data import DataLoader, Dataset
import random
import torch.nn as nn
import numpy as np
import copy
from scipy.ndimage import zoom
from network.UNet_multi_iter import UNet


import random
import torch.nn.functional as F
from dataloading_BUSI import *
from imageNames_BUSI import *

from gen_sdf import compute_sdf

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)  # 

def save_loss_value(loss_curve, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(loss_curve, json_file, indent=4)

def masked_cross_entropy_loss(seg_pred, targets, last_seg_pred, last_boundary_pred):
    # Apply mask to the loss
    loss = F.binary_cross_entropy(seg_pred, targets, reduction='none')
    mask = torch.ones_like(targets)
    mask = mask * (1 - last_seg_pred)
    mask = mask * (1 - last_boundary_pred)
    # mask[seg_pred >= 0.5] = 0.5
    # mask[boundary_pred >= 0] = 0.5
    masked_loss = loss * mask
    return masked_loss.mean()
 

if __name__ == '__main__':
    seed_torch(42)
    times = 5
    save_name = f'UNet_multi_iter{times}_maskloss_BUSI'
    save_root = 'weight'
    
    json_path = save_name + '.json'
    loss_curve = { 
        'train_loss':[],
        'val_loss':[]
    }
    os.makedirs(f'{save_root}/{save_name}', exist_ok=True)

    net = UNet(in_chns=3, in_chns_prompt=2, class_num=1)
    
    net = net.cuda()

    LR = 1e-3
    
    optimizer = torch.optim.Adam(net.parameters(), lr = LR)
    # optimizer = torch.optim.SGD(net.parameters(), lr = LR, momentum=0.9, weight_decay=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    # checkpoint_path =  f'{save_root}/{save_name}/checkpoint.pkl'
    weight_path =  f'{save_root}/{save_name}/weight.pkl'

    epoch = 0



    EPOCH = 1000

   

    criterion_seg = nn.BCELoss()
    # criterion_dice_loss = 
    criterion_regression = nn.MSELoss()
    # criterion_focal = FocalLoss()
    print('training loading...')
    dataloader_train = DataLoader2D(image_list=train_image_list, mask_list=train_mask_list, batch_size=2)

    print('val loading...')
    dataloader_val = DataLoader2D(image_list=val_image_list, mask_list=val_mask_list, batch_size=1)

    batchgenerator_train, batchgenerator_val = get_default_augmentation(dataloader_train, dataloader_val, None)
    
    
    # num_batch_per_epoch = 350
    # num_val_per_epoch = 166

    num_batch_per_epoch = 250
    num_val_per_epoch = 159
    
    min_loss = 10000
    while epoch < EPOCH:
        epoch += 1
    # for epoch in range(EPOCH):
        net.train()
        
        train_loss = 0
        count = 0

        for i in batchgenerator_train:
            image, mask = i['data'], i['target']

            sdf = compute_sdf(mask.numpy())
            sdf = torch.from_numpy(sdf).cuda().float()
            # print(sdf.shape, torch.max(sdf), torch.min(sdf), torch.max(mask), torch.min(mask))
            # exit()

            image = image.cuda()
            mask = mask.cuda()
            b, c, y, x = image.shape
            # print(torch.unique(mask))
        
            optimizer.zero_grad()
            for iter in range(times):
                

                if iter == 0:
                    prompt = torch.zeros(b, 2, y, x).cuda()
                    

                else:
                    prompt = torch.cat([seg_pred, boundary_pred], dim=1).cuda()

            # prompt = image
            
                seg_pred, boundary_pred = net(image, prompt)
                # print(torch.max(seg_pred).item(), torch.min(seg_pred).item(), torch.max(boundary_pred).item(), torch.min(boundary_pred).item())
                # print(cls_pred, label)

                loss1 = criterion_seg(seg_pred, mask)
                loss2 = criterion_regression(boundary_pred, sdf)
                if iter == 0:
                    last_seg_pred = seg_pred
                    last_boundary_pred = boundary_pred
                    loss = loss1 + loss2 / 2
                    print('epoch: {}, train loss:{}, loss1:{}, loss2:{}, lr:{}'.format(epoch, loss.item(), loss1.item(), loss2.item(), optimizer.param_groups[0]['lr']))
                else:
                    
                    loss3 = masked_cross_entropy_loss(seg_pred, mask, last_seg_pred, last_boundary_pred)
                    last_seg_pred = seg_pred
                    last_boundary_pred = boundary_pred
                    loss = loss1 * (iter + 1) + loss2 + loss3 / iter
                    print('epoch: {}, train loss:{}, loss1:{}, loss2:{}, loss3:{}, lr:{}'.format(epoch, loss.item(), loss1.item(), loss2.item(), loss3.item(), optimizer.param_groups[0]['lr']))
                
                # loss = loss1
                loss.backward(retain_graph=True)
                
                
            # with warm_scheduler.dampening():
            #     lr_scheduler.step()
           

                
                count += 1
            
        
                train_loss = train_loss + loss.item()

            # count += 1
            optimizer.step()
            if count // times >= num_batch_per_epoch:
                break
        
        loss_curve['train_loss'].append(train_loss / count)
        lr_scheduler.step()
        net.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for i in batchgenerator_val:
                image, mask = i['data'], i['target']

                image = image.cuda()
                mask = mask.cuda()

                b, c, y, x = image.shape

                for iter in range(times):
                

                    if iter == 0:
                        prompt = torch.zeros(b, 2, y, x).cuda()

                    else:
                        prompt = torch.cat([seg_pred, boundary_pred], dim=1)

                # prompt = image
                #     # print(image.shape, prompt.shape)
                    seg_pred, boundary_pred = net(image, prompt)
                loss = criterion_seg(seg_pred, mask)
                
                val_loss += loss.item()
                count += 1
                print(f'epoch: {epoch}, val loss:{loss.item()}')

                if count >= num_val_per_epoch:
                    break

            loss_curve['val_loss'].append(val_loss / count)
            val_loss = val_loss / count

            print(f'epoch: {epoch}, average val loss: {val_loss}, min loss: {min_loss}')
            
            
            save_loss_value(loss_curve, json_path)
    
            if val_loss < min_loss:
                print('save!!!')
                torch.save(net, weight_path)
                min_loss = val_loss
        