import os
import sys
import time
import math
import torch
import shutil
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split

from nets.unet import UNet

#For CMC
from nets.CMC_network import CMCNet, sigmoid_rampup, cosine_rampdown
from monai.losses import ContrastiveLoss, DiceCELoss
from utils.cmc_loss import CSC_loss_func, CAC_loss_func

from utils import ramps
from utils.asc_loss import ASC_loss


from utils.metrics import dice as dice_all
from utils.metrics import batch_dice,compute_dice
from utils.losses import BinaryDiceLoss
from utils.util import set_logging,Logger,read_list,plot_base,plot_dice2,AverageMeter
from dataloader.dataset_multi_semi import MultiSemiDataSets,TwoStreamBatchSampler,PatientBatchSampler

def train_net(start_time,base_dir,data_path,train_list,val_list,device,img_mode='',
            lr_scheduler='warmupMultistep',
            max_epoch=81,
            batch_size=56,
            labeled_bs=30,
            labeled_rate=0.1,
            unsup_epoch=0,  
            images_rate=1.0,
            base_lr=0.006,
            step_num_lr=4,
            weight_decay=0.0004,
            optim_name='adam',
            loss_name='bce',
            bce_w=10.0,
            dice_w=7.0,
            consistency=1.0,
            cons_ramp_type='sig_ram', #none sig_ram lin_ram cos_ram
            nce_weight=3.5, 
            sur_siml='dice',      #'cos' 'dice'
            pHead_sur='set_false',  #set_true set_false
            M2_epoch=51,
            T_epoch=10,
            T_num=4,
            bma_update_pool_num=6,
            M2_epoch_all=0, 
            m1_alpha=0.01,  
            m2_alpha=0.2,    
            teach_m2_WP_epoch=3,
            start_fusion_epoch=350,
            will_eval = False,
            ):
    val_3d_interval = 4 
    local_vars_dict = {}
    for var in train_net.__code__.co_varnames: 
        if var == 'local_vars_dict':
            break
        local_vars_dict[var] = locals()[var]

    if M2_epoch_all == 0:
        M2_epoch_all_num = 0
    elif M2_epoch_all == 1:
        M2_epoch_all_num = max_epoch - M2_epoch
    
    if teach_m2_WP_epoch==0:
        def get_WP2_weight(epoch):
            if epoch < int((max_epoch - M2_epoch)*0.2) + M2_epoch:
                return 0.0
            else:
                return ramps.sigmoid_rampup(epoch-M2_epoch, int((max_epoch - M2_epoch)*0.8))
    else:
        if teach_m2_WP_epoch==1:
            WP2_epoch = int((max_epoch - M2_epoch)*0.4) + M2_epoch
        elif teach_m2_WP_epoch==2:
            WP2_epoch = int((max_epoch - M2_epoch)*0.6) + M2_epoch
        elif teach_m2_WP_epoch==3:
            WP2_epoch = int((max_epoch - M2_epoch)*0.8) + M2_epoch
        def get_WP2_weight(epoch):
            if epoch < WP2_epoch:
                return 0.0
            else:
                return 1.0
    warm_up_epochs = int(max_epoch * 0.1)
    consistency_rampup = 200.0
    def get_current_consistency_weight(epoch):
        if cons_ramp_type=='sig_ram':
            # Consistency ramp-up from https://arxiv.org/abs/1610.02242
            return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)
        elif cons_ramp_type=='lin_ram':
            return consistency * ramps.linear_rampup(epoch, consistency_rampup)
        elif cons_ramp_type=='cos_ram':
            return consistency * ramps.cosine_rampdown(epoch, consistency_rampup)
    
    def update_bma_variables(model, bma_model, alpha_max, alpha_b):
        # Use the true average until the exponential average is more correct
        alpha = min(1-alpha_b, alpha_max)
        for bma_param, param in zip(bma_model.parameters(), model.parameters()):
            bma_param.data.mul_(alpha).add_(1 - alpha, param.data)

    image_channels = 1  # image_channels=3 for RGB images
    class_num = 1       # For 1 class and background, use class_num=1
    if class_num==1:
        mask_name='masks'
    elif class_num==3:
        mask_name='masks_all'
    
    """network"""
    def bma_model(bma=False):
        # Network definition
        net = UNet(image_channels,class_num,32)
        for param in net.parameters():
            param.detach_()
        return net
    #for CMC
    cmc_net = CMCNet(image_channels, class_num, 32).to(device=device)
    dice_CE_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6)
    
    # net_mode1 = UNet(image_channels,class_num,32)
    # net_mode2 = UNet(image_channels,class_num,32)
    # net_mode2_bma = bma_model(bma=True)

    # net_mode1.to(device=device)
    # net_mode2.to(device=device)
    # net_mode2_bma.to(device=device)
    # net_name = str(net_mode1)[0:str(net_mode1).find('(')]

    #For CMC
    net_name = str(cmc_net)[0:str(cmc_net).find('(')]
    
    train_dataset = MultiSemiDataSets(data_path,"train",img_mode,mask_name,train_list,images_rate)
    val_dataset = MultiSemiDataSets(data_path,"val",img_mode,mask_name,val_list)
    val_dataset_3d = MultiSemiDataSets(data_path,"val_3d",img_mode,mask_name,val_list_full)

    n_train = train_dataset.__len__()
    n_val = val_dataset.__len__()
    # define labeled and unlabeled data
    total_slices = len(train_dataset)
    labeled_slice = int(total_slices * labeled_rate)
    if labeled_slice < 60:
        labeled_slice = 60
    logging.info("Train Data: Labeled data is: <{}>, Total data is: {}".format(
          labeled_slice, total_slices))
    
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs,shuffle=True)
    def worker_init_fn(worker_id):
        random.seed(1111 + worker_id)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True,worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_dataset,batch_sampler=batch_sampler, num_workers=16, pin_memory=True,worker_init_fn=worker_init_fn)
    val_loader_2d = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False,worker_init_fn=worker_init_fn)
    slices_list = val_dataset_3d.__sampleList__()
    batch_samplerPatient = PatientBatchSampler(slices_list,patientID_list)
    val_loader_3d = DataLoader(val_dataset_3d,batch_sampler=batch_samplerPatient, num_workers=16, pin_memory=True,worker_init_fn=worker_init_fn)
    
    day_time = start_time.split(' ')
    time_str = str(day_time[0].split('-')[1] + day_time[0].split('-')[2] + day_time[1].split(':')[0] + day_time[1].split(':')[1])

    if optim_name=='adam':
        # optimizer = optim.Adam(net_mode1.parameters(), lr=base_lr, weight_decay=weight_decay)
        
        # optimizer2 = optim.Adam(net_mode2.parameters(), lr=base_lr, weight_decay=weight_decay)
        
        #For CMC
        optimizer_cmc = optim.Adam(cmc_net.parameters(), lr=base_lr, weight_decay=weight_decay)
        
    elif optim_name=='sgd':
        optimizer = optim.SGD(net_mode1.parameters(), lr=base_lr, momentum=0.9,weight_decay=weight_decay)
        optimizer2 = optim.SGD(net_mode2.parameters(), lr=base_lr, momentum=0.9,weight_decay=weight_decay)
    elif optim_name=='adamW':
        optimizer = optim.AdamW(net_mode1.parameters(), lr=base_lr, weight_decay=weight_decay)    
        optimizer2 = optim.AdamW(net_mode2.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    if lr_scheduler=='warmupMultistep':
        # warm_up_with_multistep_lr
        if step_num_lr == 2:
            lr1,lr2 = int(max_epoch*0.3) ,int(max_epoch*0.6)
            lr_milestones = [lr1,lr2]
        elif step_num_lr == 3:    
            lr1,lr2,lr3 = int(max_epoch*0.25) , int(max_epoch*0.4) , int(max_epoch*0.6)
            lr_milestones = [lr1,lr2,lr3]
        elif step_num_lr == 4:    
            lr1,lr2,lr3,lr4 = int(max_epoch*0.15) , int(max_epoch*0.35) , int(max_epoch*0.55) , int(max_epoch*0.7)
            lr_milestones = [lr1,lr2,lr3,lr4]
        warm_up_with_multistep_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
                                                else 0.1**len([m for m in lr_milestones if m <= epoch])
        # scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = warm_up_with_multistep_lr)
        # scheduler_lr2 = optim.lr_scheduler.LambdaLR(optimizer2,lr_lambda = warm_up_with_multistep_lr)
        
        #For CMC
        scheduler_lr_cmc = optim.lr_scheduler.LambdaLR(optimizer_cmc,lr_lambda = warm_up_with_multistep_lr)
    elif lr_scheduler=='warmupCosine':
        # warm_up_with_cosine_lr
        warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
                                else 0.5 * ( math.cos((epoch - warm_up_epochs) /(max_epoch - warm_up_epochs) * math.pi) + 1)
        scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = warm_up_with_cosine_lr)
    elif lr_scheduler=='autoReduce':
        scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=6, verbose=True, cooldown=2,min_lr=0)
    if loss_name=='bce':
        criterion = nn.BCELoss()

    dice_loss = BinaryDiceLoss()
                                                           #dice       #set_false
    contrast_loss = ASC_loss(batch_size-labeled_bs, device, sur_siml, pHead_sur)
    # optimizer_name = str(optimizer)[0:str(optimizer).find('(')]
    
    #For CMC
    optimizer_name = str(optimizer_cmc)[0:str(optimizer_cmc).find('(')]
    
    param_str = "Starting training:"
    for var in list(local_vars_dict.keys()):
        if var != 'device':
            var_value = local_vars_dict[var]
            param_str += "\n\t" + var + ":" + " "*(15-len(var)) + str(var_value)
    
    logging.info(param_str+f'''\n\tNet Name:\t\t{net_name}\n\tInput Channel:\t{image_channels}\n\tClasses Num:\t{class_num}\n\tImages Shape:\t{img_h}*{img_w}''')

    train_log = AverageMeter()
    val_log = AverageMeter()
    lr_curve = list()
    
    def apply_dropout(m):
        if type(m) == nn.Dropout:
            m.train()
    def de_apply_dropout(m):
        if type(m) == nn.Dropout:
            m.eval()

    best_performance_m2_dict = {}
    best_performance_m2_list_sort = []
    

    for epoch in range(max_epoch):
        # train the model
        # net_mode1.train()
        # net_mode1.apply(de_apply_dropout)
        # net_mode2.train()
        # net_mode2.apply(de_apply_dropout)
        # net_mode2_bma.apply(apply_dropout)
        
        #For CMC
        cmc_net.train()
        cmc_net.apply(de_apply_dropout)
        unsup_loss = 0.0
        csc_loss = torch.tensor(0.0)
        cac_loss = torch.tensor(0.0)
        
        with tqdm(total=labeled_slice, desc=f'Epoch {epoch + 1}/{max_epoch}', unit='img', leave=is_leave) as pbar:
            for i,batch in enumerate(train_loader):
                # forward
                imgs_mode1 = batch['mode1']
                imgs_mode2 = batch['mode2']
                true_masks = batch['mask']
                slice_name = batch['idx']
                # print("batch idxs:", slice_name)
                assert imgs_mode1.shape[1] == image_channels, \
                    f'Network has been defined with {image_channels} input channels, ' \
                    f'but loaded images have {imgs_mode1.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs_mode1 = imgs_mode1.to(device=device, dtype=torch.float32)
                imgs_mode2 = imgs_mode2.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                # only on mode 2
                unlabeled_imgs = imgs_mode2[labeled_bs:]
                
                
                # # ================================= start for superviser 
                
                # #for ce loss and dice loss
                # masks_pred_mode1 = net_mode1(imgs_mode1)
                # masks_pred_mode2 = net_mode2(imgs_mode2)

                # loss_base_mode1 = criterion(masks_pred_mode1[:labeled_bs], true_masks[:labeled_bs])
                # loss_base_mode2 = criterion(masks_pred_mode2[:labeled_bs], true_masks[:labeled_bs])

                # loss_dice_mode1 = dice_loss(masks_pred_mode1[:labeled_bs], true_masks[:labeled_bs])
                # loss_dice_mode2 = dice_loss(masks_pred_mode2[:labeled_bs], true_masks[:labeled_bs])

                # loss_sup_mode1 = loss_base_mode1 * 10.0 + loss_dice_mode1 * 7.0
                # loss_sup_mode2 = loss_base_mode2 * 10.0 + loss_dice_mode2 * 7.0

                
                #For CMC
                _,_, masks_pred_mode1_sup, masks_pred_mode2_sup = cmc_net(imgs_mode1[:labeled_bs], imgs_mode2[:labeled_bs])


                # # ================================= end for unsupervised 


                # # ================================= start for unsupervised 
                # # use both MSE and ASC for unlabeled data of modality 1 and modality 2
                # if cons_ramp_type != 'none':
                #     consistency_weight = get_current_consistency_weight(epoch)
                # else:
                #     consistency_weight = consistency
                
                # #MSE loss
                # consistency_loss = torch.mean(    
                #     (masks_pred_mode1[labeled_bs:]-masks_pred_mode2[labeled_bs:])**2) 

                # loss_mse = consistency_weight * consistency_loss
                # train_log.add_value({"loss_mse": consistency_loss.item()}, n=1)
                
                # ##ASC Loss
                # loss_contrast = contrast_loss(masks_pred_mode1[labeled_bs:],masks_pred_mode2[labeled_bs:])
                # train_log.add_value({"loss_contrast": loss_contrast.item()}, n=1)
                
                
                #Integrate CMC only for unlabeled data
                if epoch >= start_fusion_epoch: 
                    with torch.no_grad():
                        mode_1_img_F_ds, mode_2_img_F_ds, masks_pred_mode1_unsup, masks_pred_mode2_unsup = cmc_net(imgs_mode1[labeled_bs:], imgs_mode2[labeled_bs:])
                    
                    # print("mode1:", mode_1_img_F_ds.mean().item(), mode_1_img_F_ds.std().item())
                    # print("mode2:", mode_2_img_F_ds.mean().item(), mode_2_img_F_ds.std().item())
                    # print("csc_loss:", csc_loss.item())
                    
                     
                    # print(mode_1_img_F_ds)
                    # print(mode_2_img_F_ds)
                    
                    
                    csc_loss = CSC_loss_func(mode_1_img_F_ds, mode_2_img_F_ds)
                    
                    cac_loss = CAC_loss_func(masks_pred_mode1_unsup, masks_pred_mode2_unsup)
                    
                    ### compute CSC loss
                    consistency_weight_csc = sigmoid_rampup(epoch, max_epoch)
                    ### compute CAC loss
                    consistency_weight_cac = cosine_rampdown(epoch, max_epoch)

                    # print(consistency_weight_csc)
                    # unsup_loss = consistency_weight_csc * csc_loss + consistency_weight_cac * cac_loss
                    unsup_loss =  consistency_weight_csc * csc_loss + consistency_weight_cac * cac_loss
                    
            
                # # ================================= end for unsupervised 
                
                
                # # ===========================start for PReL
                # #M2_epoch = 51
                # #nce_weight = 3.5
                # if epoch < M2_epoch:
                #     loss_sup_all = loss_sup_mode1 + loss_sup_mode2
                #     loss = loss_sup_all + nce_weight * loss_contrast + loss_mse
                # #epoch 52
                # else:
                #     ## get uncertainty mask
                #     T = T_num
                #     _, _, w, h = unlabeled_imgs.shape
                #     unlabeled_bs = unlabeled_imgs.shape[0]
                #     preds = torch.zeros([T,unlabeled_bs, class_num, w, h]).cuda()
                #     for i in range(T):
                #         noise_inputs_un = unlabeled_imgs + \
                #             torch.clamp(torch.randn_like(
                #                 unlabeled_imgs) * 0.1, 0, 0.2)
                #         with torch.no_grad():
                #             preds[i] = net_mode2_bma(noise_inputs_un)
                #     preds = torch.mean(preds, dim=0)
                #     masks_pred_mode2_certainty = preds

                #     loss_base_mode1_2 = criterion(masks_pred_mode1[labeled_bs:], masks_pred_mode2_certainty)
                #     loss_dice_mode1_2 = dice_loss(masks_pred_mode1[labeled_bs:], masks_pred_mode2_certainty)
                #     loss_sup_mode1_2 = loss_base_mode1_2 * bce_w + loss_dice_mode1_2 * dice_w

                #     loss_contrast_mode1_2 = contrast_loss(masks_pred_mode2_certainty,masks_pred_mode1[labeled_bs:])

                #     loss_base_mode2_2 = criterion(masks_pred_mode2[labeled_bs:], masks_pred_mode2_certainty)
                #     loss_dice_mode2_2 = dice_loss(masks_pred_mode2[labeled_bs:], masks_pred_mode2_certainty)
                #     loss_sup_mode2_2 = loss_base_mode2_2 * bce_w + loss_dice_mode2_2 * dice_w

                #     loss_contrast_mode2_2 = contrast_loss(masks_pred_mode2_certainty,masks_pred_mode2[labeled_bs:])

                #     loss_teach_m1 = m1_alpha * loss_contrast_mode1_2 + (1-m1_alpha) * loss_sup_mode1_2

                #     loss_teach_m2 = m2_alpha * loss_contrast_mode2_2 + (1-m2_alpha) * loss_sup_mode2_2

                #     WP2 = get_WP2_weight(epoch)
                #     loss = loss_teach_m1 + loss_teach_m2 * WP2
                # # # ===========================end for PReL
                
                
                # ======================start for only sup
                # loss_sup_all = loss_sup_mode1 + loss_sup_mode2
                # loss = loss_sup_all
                #======================end for only sup
                
                #======================start for CMC
                mode_1_loss = dice_CE_loss(masks_pred_mode1_sup, true_masks[:labeled_bs])
                mode_2_loss = dice_CE_loss(masks_pred_mode2_sup, true_masks[:labeled_bs])
                
                sup_loss = (mode_1_loss + mode_2_loss)/2
            
                loss = sup_loss + unsup_loss
                #======================end for CMC
                
                
                #'''backward'''
                loss.backward()
                #'''update weights'''                  
                # optimizer.step()        # update parameters of net
                # optimizer.zero_grad()   # reset gradient
                # optimizer2.step()        # update parameters of net
                # optimizer2.zero_grad()   # reset gradient
                
                #For CMC
                optimizer_cmc.step()
                optimizer_cmc.zero_grad()

                # train_log.add_value({"loss": loss.item()}, n=1)  
                # pred_mode1 = (masks_pred_mode1 > 0.5).float()
                # pred_mode2 = (masks_pred_mode2 > 0.5).float()
                # dice_mode1_sum,num = batch_dice(pred_mode1.cpu().data[:labeled_bs], true_masks.cpu()[:labeled_bs])
                # dice_mode2_sum,num = batch_dice(pred_mode2.cpu().data[:labeled_bs], true_masks.cpu()[:labeled_bs])
                # dice_mode1 = dice_mode1_sum / num
                # dice_mode2 = dice_mode2_sum / num
                
                #For CMC
                train_log.add_value({"loss": loss.item(), "Sup_loss": sup_loss.item(), "CSC_loss": csc_loss.item(), "CAC_loss": cac_loss.item()}, n=1)  
                pred_mode1 = (masks_pred_mode1_sup > 0.5).float()
                pred_mode2 = (masks_pred_mode2_sup > 0.5).float()
                dice_mode1_sum,num = batch_dice(pred_mode1.cpu().data, true_masks.cpu()[:labeled_bs])
                dice_mode2_sum,num = batch_dice(pred_mode2.cpu().data, true_masks.cpu()[:labeled_bs])
                dice_mode1 = dice_mode1_sum / num
                dice_mode2 = dice_mode2_sum / num

                train_log.add_value({"dice_m1": dice_mode1,}, n=1)
                train_log.add_value({"dice_m2": dice_mode2,}, n=1)
                if set_args == True:
                    pbar.update(labeled_bs)

            train_log.updata_avg()
            mean_loss = train_log.res_dict["loss"][epoch]
            #For CMC
            mean_csc_loss = train_log.res_dict["CSC_loss"][epoch]
            mean_cac_loss = train_log.res_dict["CAC_loss"][epoch]
            mean_sup_loss = train_log.res_dict["Sup_loss"][epoch]
            
            mean_dice_mode1 = train_log.res_dict["dice_m1"][epoch]
            mean_dice_mode2 = train_log.res_dict["dice_m2"][epoch]

        
        if will_eval == True:
            #validate the model
            # net_mode1.eval()
            # net_mode2.eval()
            cmc_net.eval()

            n_val_2d = len(val_loader_2d)  
            n_val_3d = len(val_loader_3d)  
            
            # #===========start for PReL
            # if epoch == M2_epoch-2:
            #     val_3d_interval = 1
            # #===========end for PReL  
            
            
            if epoch % val_3d_interval==0:
                compute_3d = True
                n_val = n_val_3d
                val_loader = val_loader_3d
            else:
                compute_3d = False
                n_val = n_val_2d
                val_loader = val_loader_2d
                
            with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
                for j,batch_val in enumerate(val_loader):
                    imgs_mode1 = batch_val['mode1']
                    imgs_mode2 = batch_val['mode2']
                    true_masks = batch_val['mask']

                    imgs_mode1 = imgs_mode1.to(device=device, dtype=torch.float32)
                    imgs_mode2 = imgs_mode2.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.float32)
                    if compute_3d==True:
                        batch_num = imgs_mode1.shape[0]
                        batch_fore = int(batch_num / 3)

                        imgs_fore_m1 = imgs_mode1[:batch_fore]
                        imgs_mid_m1 = imgs_mode1[batch_fore:batch_fore*2]
                        imgs_after_m1 = imgs_mode1[batch_fore*2:]

                        imgs_fore_m2 = imgs_mode2[:batch_fore]
                        imgs_mid_m2 = imgs_mode2[batch_fore:batch_fore*2]
                        imgs_after_m2 = imgs_mode2[batch_fore*2:]

                        with torch.no_grad():
                            # mask_pred_fore_m1 = net_mode1(imgs_fore_m1)
                            # mask_pred_mid_m1 = net_mode1(imgs_mid_m1)
                            # mask_pred_after_m1 = net_mode1(imgs_after_m1)

                            # mask_pred_fore_m2 = net_mode2(imgs_fore_m2)
                            # mask_pred_mid_m2 = net_mode2(imgs_mid_m2)
                            # mask_pred_after_m2 = net_mode2(imgs_after_m2)

                            #For CMC
                            _,_,mask_pred_fore_m1, mask_pred_fore_m2 = cmc_net(imgs_fore_m1, imgs_fore_m2)
                            _,_,mask_pred_mid_m1, mask_pred_mid_m2 = cmc_net(imgs_mid_m1, imgs_mid_m2)
                            _,_,mask_pred_after_m1, mask_pred_after_m2 = cmc_net(imgs_after_m1,imgs_after_m2)


                        mask_pred_mode1 = torch.cat([mask_pred_fore_m1, mask_pred_mid_m1, mask_pred_after_m1], dim=0)
                        mask_pred_mode2 = torch.cat([mask_pred_fore_m2, mask_pred_mid_m2, mask_pred_after_m2], dim=0)
                        
                        # dice
                        pred_mode1 = mask_pred_mode1.ge(0.5).float()
                        pred_mode2 = mask_pred_mode2.ge(0.5).float()
                        pred_np_mode1 = pred_mode1.cpu().data.numpy().astype("uint8")
                        pred_np_mode2 = pred_mode2.cpu().data.numpy().astype("uint8")
                        true_np = true_masks.cpu().numpy().astype("uint8")
                        if class_num == 1:
                            # 3D dice for class number == 1
                            dice_val_3d_mode1 = dice_all(pred_np_mode1, true_np)
                            dice_val_3d_mode2 = dice_all(pred_np_mode2, true_np)

                        elif class_num == 3:
                            # 3D dice for class number == 3
                            dice_val_3d_m1_1 = dice_all(pred_np_mode1[:,0,:,:], true_np[:,0,:,:])
                            dice_val_3d_m1_2 = dice_all(pred_np_mode1[:,1,:,:], true_np[:,1,:,:])
                            dice_val_3d_m1_3 = dice_all(pred_np_mode1[:,2,:,:], true_np[:,2,:,:])
                            dice_val_3d_mode1 = (dice_val_3d_m1_1 + dice_val_3d_m1_2 + dice_val_3d_m1_3) / 3

                            dice_val_3d_m2_1 = dice_all(pred_np_mode2[:,0,:,:], true_np[:,0,:,:])
                            dice_val_3d_m2_2 = dice_all(pred_np_mode2[:,1,:,:], true_np[:,1,:,:])
                            dice_val_3d_m2_3 = dice_all(pred_np_mode2[:,2,:,:], true_np[:,2,:,:])
                            dice_val_3d_mode2 = (dice_val_3d_m2_1 + dice_val_3d_m2_2 + dice_val_3d_m2_3) / 3

                        val_log.add_value({"dice_3d_m1": dice_val_3d_mode1}, n=1)
                        val_log.add_value({"dice_3d_m2": dice_val_3d_mode2}, n=1)

                        val_log.add_value({"3d_interval_list": epoch}, n=1)
                        dice_val_sum_mode1,nidus_num,nidus_start = compute_dice(pred_mode1.cpu().data, true_masks.cpu(),deNoNidus=True)
                        dice_val_sum_mode2,nidus_num,nidus_start = compute_dice(pred_mode2.cpu().data, true_masks.cpu(),deNoNidus=True)
                        nidus_end = nidus_start+nidus_num-1
                        mask_pred_mode1 = mask_pred_mode1[nidus_start:nidus_end+1]
                        mask_pred_mode2 = mask_pred_mode2[nidus_start:nidus_end+1]
                        true_masks = true_masks[nidus_start:nidus_end+1]
                    else:
                        with torch.no_grad():
                            # mask_pred_mode1 = net_mode1(imgs_mode1)
                            # mask_pred_mode2 = net_mode2(imgs_mode2)
                            
                            #For CMC
                            _, _,mask_pred_mode1,mask_pred_mode2 = cmc_net(imgs_mode1,imgs_mode2)
                            
                        # 2D dice
                        pred_mode1 = mask_pred_mode1.ge(0.5).float()
                        dice_val_sum_mode1,nidus_num = batch_dice(pred_mode1.cpu().data, true_masks.cpu())
                        
                        pred_mode2 = mask_pred_mode2.ge(0.5).float()
                        dice_val_sum_mode2,nidus_num = batch_dice(pred_mode2.cpu().data, true_masks.cpu())
                        
                    val_log.add_value({"dice_m1": dice_val_sum_mode1}, n=nidus_num)
                    val_log.add_value({"dice_m2": dice_val_sum_mode2}, n=nidus_num)
                    
                    # vlaid loss
                    # loss_val_base_mode1 = criterion(mask_pred_mode1, true_masks)
                    # loss_val_base_mode2 = criterion(mask_pred_mode2, true_masks)
                    # loss_val_dice_mode1 = dice_loss(mask_pred_mode1, true_masks)
                    # loss_val_dice_mode2 = dice_loss(mask_pred_mode2, true_masks)
                    # loss_val_mode1 = loss_val_base_mode1 * 10.0+ loss_val_dice_mode1 * 7.0
                    # loss_val_mode2 = loss_val_base_mode2 * 10.0+ loss_val_dice_mode2 * 7.0
                    # loss_val = loss_val_mode1 + loss_val_mode2
                    
                    loss_mode1 = dice_CE_loss(mask_pred_mode1, true_masks)
                    loss_mode2 = dice_CE_loss(mask_pred_mode2, true_masks)
                    
                    loss_val = (loss_mode1 + loss_mode2)/2                
                    
                
                    val_log.add_value({"loss": loss_val.item()}, n=1)
                    if set_args == True:
                        pbar.update()
                    
                val_log.updata_avg()
                valid_loss_mean = val_log.res_dict["loss"][epoch]
                valid_dice_mean_mode1 = val_log.res_dict["dice_m1"][epoch]
                valid_dice_mean_mode2 = val_log.res_dict["dice_m2"][epoch]

                if epoch % val_3d_interval==0:
                    valid_dice_m1_3d = val_log.res_dict["dice_3d_m1"][-1]
                    valid_dice_m2_3d = val_log.res_dict["dice_3d_m2"][-1]
                    logging.info("valid_dice_m1_3d:{:.4f},valid_dice_m2_3d:{:.4f}".format(valid_dice_m1_3d,valid_dice_m2_3d))

            # # ===========start for PReL
            ## update the bma_model2 wetights 
            # if T_epoch < epoch < M2_epoch + M2_epoch_all_num:
            #     valid_dice_m2 = round(valid_dice_mean_mode2,4)
                
            #     if len(best_performance_m2_dict) < bma_update_pool_num:
            #         tag = str(epoch)
            #         best_performance_m2_dict[tag] = valid_dice_m2
            #         logging.info(f"Epoch: {epoch}, m2_dice: {valid_dice_m2}, start push in bma pool")
            #         best_performance_m2_list_sort = sorted(best_performance_m2_dict.items(), key=lambda x: x[1])
                
            #     elif epoch == (T_epoch+1) + bma_update_pool_num:
            #         update_bma_variables(net_mode2,net_mode2_bma,0.99,1.0)
                
            #     elif valid_dice_m2 > best_performance_m2_list_sort[0][1]: ## compare with the minimum value in the dict
            #         pop_id = best_performance_m2_list_sort[0][0]
            #         best_performance_m2_dict.pop(pop_id)
            #         tag = str(epoch)
            #         best_performance_m2_dict[tag] = valid_dice_m2
            #         logging.info(f"Epoch: {epoch}, m2_dice: {valid_dice_m2},m2_dice last: {best_performance_m2_list_sort[-1][1]}, start push in bma pool")
            #         alpha_b = (valid_dice_m2 - best_performance_m2_list_sort[0][1]) / valid_dice_m2
            #         update_bma_variables(net_mode2,net_mode2_bma,0.99,alpha_b)
            #         best_performance_m2_list_sort = sorted(best_performance_m2_dict.items(), key=lambda x: x[1])
            # # ===========start for PReL
            
        ## update lr
        if lr_scheduler=='autoReduce':
            scheduler_lr.step(valid_loss_mean)
        else:
            # scheduler_lr.step()
            # scheduler_lr2.step()
            #For CMC 
            scheduler_lr_cmc.step()
            
        # lr_epoch = optimizer.param_groups[0]['lr']
        #For CMC
        lr_epoch = optimizer_cmc.param_groups[0]['lr']
        lr_curve.append(lr_epoch)

        # logging.info(
        #     'Epoch:[{:0>3}/{:0>3}], Train Loss: {:.4f} , Val Loss: {:.4f} ,Train Dice: mode1 {:.4f} mode2 {:.4f},  Val Dice: mode1 {:.4f} mode2 {:.4f},LR: {:.6f}'.format(
        #             epoch,max_epoch,         mean_loss, valid_loss_mean,        mean_dice_mode1, mean_dice_mode2  ,valid_dice_mean_mode1 , valid_dice_mean_mode2,lr_epoch))

        if will_eval == True:
            logging.info(
                'Epoch:[{:0>3}/{:0>3}], Train Loss: {:.4f} , Sup Loss {:.4f}, CSC_Loss {:.4f}, CAC_loss {:.4f} , Val Loss: {:.4f}, Train Dice: mode1 {:.4f} mode2 {:.4f} ,LR: {:.6f}'.format(
                        epoch,max_epoch,         mean_loss,     mean_sup_loss,      mean_csc_loss,    mean_cac_loss, valid_loss_mean,       mean_dice_mode1, mean_dice_mode2, lr_epoch))
        else: 
            logging.info(
                'Epoch:[{:0>3}/{:0>3}], Train Loss: {:.4f} , Sup Loss {:.4f}, CSC_Loss {:.4f}, CAC_loss {:.4f} , Train Dice: mode1 {:.4f} mode2 {:.4f} ,LR: {:.6f}'.format(
                        epoch,max_epoch,         mean_loss,     mean_sup_loss,      mean_csc_loss,    mean_cac_loss,       mean_dice_mode1, mean_dice_mode2, lr_epoch))

    return train_log.res_dict,val_log.res_dict,lr_curve,cmc_net #,net_mode1,net_mode2

def main():
    base_dir = 'res-BraTs-Semi-CML'
    img_mode = 't2_t1ce'  # t2_t1ce  ct_pet
    
    if set_args==True:
        args = set_argparse()
        print('WARNING!!! Using argparse for parameters to obtain ')
        base_dir = args.base_dir
        img_mode = args.img_mode
    assert 'res-' in base_dir, \
            f'base_dir should include string:\'res-\',but base_dir is \'{base_dir}\'.'
    base_dir = base_dir.replace('res-',f'res-{img_mode}-',1)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    backup_code(base_dir)
    log_path = os.path.join(base_dir, 'training.log') 
    sys.stdout = Logger(log_path=log_path)
    set_logging(log_path=log_path)
    set_random_seed(seed_num=1111)
    
    """GPU ID"""
    gpu_list = [args.gpu] #[0,1]
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device : {device}\n'
                 f'\tGPU ID is [{os.environ["CUDA_VISIBLE_DEVICES"]}],using {torch.cuda.device_count()} device\n'
                 f'\tdevice name:{torch.cuda.get_device_name(0)}')

    start_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    
    global img_h,img_w
    if img_mode=='ct_pet':
        data_path = "/mnt/HDD2/dat/med/semi-CML-public/dataset/Hecktor_slice" 
        img_h,img_w = 144,144
    else:
        data_path = "/mnt/HDD2/dat/med/semi-CML-public/dataset/BraTS_slice" 
        img_h,img_w = 160,160
    train_list = 'randP1_slice_nidus_train.list'
    val_list = 'randP1_slice_nidus_val.list'
    global val_list_full
    val_list_full = 'randP1_slice_all_val.list'
    global patientID_list
    patientID_list = read_list(data_path+'/randP1_volume_val.list')
    time_tic = time.time()
    img_mode1 = img_mode.split('_')[0]
    img_mode2 = img_mode.split('_')[1]
    global is_leave

    logging.info('============ Start  train ==============')
    if set_args==True:
        ########## Using Argument Parser ##########
        is_leave = False
        train_log,val_log,lr_curve,cmc_net = train_net(start_time, # return net_mode1,net_mode2 
                                                                    base_dir,
                                                                    data_path,
                                                                    args.train_list,
                                                                    args.val_list,device,
                                                                    args.img_mode,
                                                                    args.lr_scheduler,
                                                                    args.max_epoch,
                                                                    args.batch_size,
                                                                    args.labeled_bs,
                                                                    args.labeled_rate,
                                                                    args.unsup_epoch,
                                                                    args.images_rate,
                                                                    args.base_lr,
                                                                    args.step_num_lr,
                                                                    args.weight_decay,
                                                                    args.optim_name,
                                                                    args.loss_name,
                                                                    args.bce_w,
                                                                    args.dice_w,
                                                                    args.consistency,
                                                                    args.cons_ramp_type,
                                                                    args.nce_weight,
                                                                    args.sur_siml,
                                                                    args.pHead_sur,
                                                                    args.M2_epoch,
                                                                    args.T_epoch,
                                                                    args.T_num,
                                                                    args.bma_update_pool_num,
                                                                    args.M2_epoch_all,
                                                                    args.m1_alpha,
                                                                    args.m2_alpha,
                                                                    args.teach_m2_WP_epoch,
                                                                    args.start_fusion_epoch,
                                                                    args.will_eval)
    else:   
        is_leave = True
        train_log,val_log,lr_curve,net_mode1,net_mode2 = train_net(start_time,
                                                                   base_dir,
                                                                   data_path,
                                                                   train_list,
                                                                   val_list,
                                                                   device,
                                                                   img_mode)
    
    # net_name = str(net_mode1)[0:str(net_mode1).find('(')]
    # model_path_name1 = base_dir + '/' + f'model_{net_name}_last_mode1.pth'
    # torch.save(net_mode1, model_path_name1)
    # model_path_name2 = base_dir + '/' + f'model_{net_name}_last_mode2.pth'
    # torch.save(net_mode2, model_path_name2)
    
    cmc_net_name = str(cmc_net)[0:str(cmc_net).find('(')]
    model_path_name = base_dir + '/' + f'model_{cmc_net_name}_last.pth'
    torch.save(cmc_net, model_path_name)
    logging.info('Model saved !')    

    """Plot"""
    plot_dice_loss(train_log,val_log,lr_curve,base_dir,img_mode1,img_mode2)

    time_toc = time.time()
    time_s = time_toc - time_tic
    time_end = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time_toc))
    logging.info("Train finished time: {}".format(time_end))
    logging.info("Time consuming : {:.2f} min in train and test".format(time_s / 60))

def set_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=7)
    parser.add_argument('--base_dir', type=str,default='res-BraTs-seed-2-CMC-10%_seed_1111_3rd', 
                        help='base dir name')
    parser.add_argument('--train_list', type=str,default='randP1_slice_nidus_train.list', 
                        help='a list of train data')
    parser.add_argument('--val_list', type=str,default='randP1_slice_nidus_val.list', 
                        help='a list of val data')
    parser.add_argument('--img_mode', type=str,default='t2_t1ce', #ct_pet  t2_t1ce
                        help='medical images mode')
    
    
    
    #For CMC
    parser.add_argument('--max_epoch', type=int,default=81, 
                        help='maximum epoch')
    parser.add_argument('--start_fusion_epoch', default=30, type=int)
    parser.add_argument('--will_eval', default=1, type=bool)
    
    parser.add_argument('--batch_size', type=int,default=24,
                        help='batch size per gpu')
    parser.add_argument('--labeled_bs', type=int,default=20,
                        help='labeled batch_size per gpu')
    parser.add_argument('--labeled_rate', type=float,default=0.1,
                        help='labeled rate for supervised train')
    parser.add_argument('--images_rate', type=float,default=1,
                        help='images rate')
    parser.add_argument('--base_lr', type=float,default=0.006,
                        help='segmentation network learning rate')
    parser.add_argument('--step_num_lr', type=int,default=4,
                        help='step_num for warmupMultistep lr')
    parser.add_argument('--weight_decay', type=float,default=0.0004,
                        help='weight decay(L2 Regularization)')   
    parser.add_argument('--optim_name', type=str,default='adam', 
                        help='optimizer name')
    parser.add_argument('--loss_name', type=str,default='bce', 
                        help='loss name')   
    parser.add_argument('--bce_w', type=float,default=10.0,
                        help='bce sup Weight')
    parser.add_argument('--dice_w', type=float,default=7.0,
                        help='dice sup Weight')     
    parser.add_argument('--lr_scheduler', type=str,default='warmupMultistep', 
                        help='lr scheduler')
    parser.add_argument('--unsup_epoch', type=int,default=0,
                        help='start epoch for unsupervised loss')
    parser.add_argument('--consistency', type=float,default=1.0,
                        help='Consistency loss Weight')
    parser.add_argument('--cons_ramp_type', type=str,default='sig_ram',
                        help='Consistency rampup type')
    parser.add_argument('--nce_weight', type=float,default=3.5,
                        help='contrast loss weight')
    parser.add_argument('--sur_siml', type=str,default='dice', 
                        help='sur_siml')
    parser.add_argument('--pHead_sur', type=str,default='set_false', 
                        help='pHead_sur')

    parser.add_argument('--M2_epoch', type=int,default=51,
                        help='start epoch for freeze Modality 2 net')
    parser.add_argument('--T_epoch', type=int,default=10,
                        help='start epoch for generating teacher ')
    parser.add_argument('--T_num', type=int,default=4,
                        help='T_num')
    parser.add_argument('--bma_update_pool_num', type=int,default=6,
                        help='bma_update_pool_num')

    parser.add_argument('--M2_epoch_all', type=int,default=0,
                        help='M2_epoch_all')  
    parser.add_argument('--m1_alpha', type=int,default=0.1,
                        help='m1_alpha')  
    parser.add_argument('--m2_alpha', type=int,default=0.1,
                        help='m2_alpha')  
    parser.add_argument('--teach_m2_WP_epoch', type=int,default=3,
                        help='teach_m2_WP_epoch')      
    
    
    args = parser.parse_args()
    return args

def set_random_seed(seed_num):
    if seed_num != '':
        logging.info(f'set random seed: {seed_num}')
        cudnn.benchmark = False      
        cudnn.deterministic = True   
        random.seed(seed_num)    
        np.random.seed(seed_num)  
        torch.manual_seed(seed_num) 
        torch.cuda.manual_seed(seed_num) 

def plot_dice_loss(train_dict,val_dict,lr_curve,base_dir,img_mode1,img_mode2):
    # plot dice curve
    plot_dice2(train_dict['dice_m1'],val_dict['dice_m1'],base_dir,f'Dice_{img_mode1}',val_dict['dice_3d_m1'],val_dict['3d_interval_list'])
    plot_dice2(train_dict['dice_m2'],val_dict['dice_m2'],base_dir,f'Dice_{img_mode2}',val_dict['dice_3d_m2'],val_dict['3d_interval_list'])
    # plot loss curve
    for key in train_dict:
        if 'loss' in key:
            if key in val_dict:
                plot_base(train_dict[key],val_dict[key],base_dir,mode=key)
            else:
                plot_base(train_dict[key],[],base_dir,mode=key)
    # plot lr curve
    lr_x = range(len(lr_curve))
    lr_y = lr_curve
    plt.plot(lr_x, lr_y)
    plt.legend(['learning_rate'],loc='upper right')
    plt.ylabel('lr value')
    plt.xlabel('epoch')
    plt.title("Learning Rate" )
    plt.savefig('{}/lr.jpg'.format(base_dir))
    plt.close()   
    
def backup_code(base_dir):
    code_path = os.path.join(base_dir, 'code') 
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    train_name = os.path.basename(__file__)
    dataset_name = 'dataset_multi_semi.py'
    nmodel_name = 'unet.py'
    shutil.copy('dataloader/' + dataset_name, code_path + '/' + dataset_name)
    shutil.copy('nets/' + nmodel_name, code_path + '/' + nmodel_name)
    shutil.copy(train_name, code_path + '/' + train_name)

if __name__ == '__main__':
    set_args = True
    main()
