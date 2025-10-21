import os
import time
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from medpy import metric
import skimage.io as io
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from utils.metrics import dice,dice_3
from utils.util import read_list,set_logging
from nets.unet import UNet
import SimpleITK as sitk
plt.switch_backend('agg')

def save_imgs(img_mode,out_path,img_np,true_mask,pred_mask,patientSliceID,dice_test,exp):
    img_uint8 = np.clip(img_np, 0, 1)   # nếu dữ liệu đã normalize về [0,1]
    img_uint8 = (img_uint8 * 255).astype(np.uint8)
    io.imsave('{}/{}_img{}.jpg'.format(out_path,patientSliceID,exp),img_uint8)

    '''save true mask'''
    true_mask = true_mask * 255
    mask_true_img = Image.fromarray(true_mask)
    mask_true_img.save('{}/{}_true{}.png'.format(out_path,patientSliceID,exp), cmap="gray")

    '''save pred mask'''
    pred_mask_gray = pred_mask * 255
    pred_mask_img = Image.fromarray(pred_mask_gray)
    pred_mask_img.save('{}/{}_pred_{:.4f}{}.png'.format(out_path,patientSliceID,dice_test,exp), cmap="gray")
    # np.save('{}/{}_pred_{:.4f}.npy'.format(out_path,patientSliceID,dice_test),pred_mask)# (160, 160) dtype('uint8')0 1 2 4
    def save_plt(num,np,title):
        plt.subplot(2,2,num)
        plt.imshow(np, cmap="gray")
        plt.axis('off')
        plt.title(title)

    save_plt(1,img_np,img_mode)
    save_plt(2,img_np,img_mode)
    save_plt(3,true_mask,"true mask")
    save_plt(4,pred_mask,title="pred mask-dice:{:.4f}".format(dice_test))
    # patientSliceID = patientSliceID.replace("_slice_","_")
    plt.savefig('{}/{}{}.png'.format(out_path,patientSliceID,exp))
    plt.show()

def predict_img(net,device,true_mask,init_img,scale_factor=1,out_threshold=0.5):
    """
    test 2 images from each modality
    """
    # step 1/4 : path --> img_chw
    if len(init_img[0].shape) == 2:
        img_chw_mode_1 = np.expand_dims(init_img[0], axis=0)
    if len(init_img[1].shape) == 2:
        img_chw_mode_2 = np.expand_dims(init_img[1], axis=0)
    # img_chw = img_chw / 255
    # step 2/4 : img --> tensor
    img_tensor = [torch.tensor(img_chw_mode_1).to(torch.float32).unsqueeze_(0).to(device),torch.tensor(img_chw_mode_2).to(torch.float32).unsqueeze_(0).to(device)]
    # img_tensor.unsqueeze_(0)  #or img = img.unsqueeze(0)
    # img_tensor = img_tensor.to(device)
    # step 3/4 : tensor --> features
    time_tic = time.time()
    with torch.no_grad():
        _,_,mask_pred_mode_1,mask_pred_mode_2 = net.forward_onlyMIA(img_tensor[0], img_tensor[1])
    if len(mask_pred_mode_1) != 2:
        outputs_mode_1 = mask_pred_mode_1
        outputs_mode_2 = mask_pred_mode_2
    else:
        outputs_mode_1 = mask_pred_mode_1[0]
        outputs_mode_2 = mask_pred_mode_2[0]

        if torch.min(outputs_mode_1) < 0 or torch.max(outputs_mode_1) > 1:
            outputs_mode_1 = torch.sigmoid(outputs_mode_1)
        if torch.min(outputs_mode_2) < 0 or torch.max(outputs_mode_2) > 1:
            outputs_mode_2 = torch.sigmoid(outputs_mode_2)
            
    time_toc = time.time()
    time_s = time_toc - time_tic
    # logging.info("time is ",time_s)
    pred_mask_mode_1 = outputs_mode_1.ge(out_threshold).cpu().data.numpy().astype("uint8")
    pred_mask_mode_2 = outputs_mode_2.ge(out_threshold).cpu().data.numpy().astype("uint8")

    
    pred_mask_mode_1 = pred_mask_mode_1.squeeze()
    pred_mask_mode_2 = pred_mask_mode_2.squeeze()

    dice_test_mode_1 = dice(pred_mask_mode_1, true_mask)
    dice_test_mode_2 = dice(pred_mask_mode_2, true_mask)
    
    return pred_mask_mode_1,pred_mask_mode_2,dice_test_mode_1,dice_test_mode_2,time_s

def test_images(net,device,img_path,true_path,in_files,out_path=None,is_save_img=True,exp=''):
    img_mode_1 = img_path[0].split('/')[-1].split('_')[-1]
    img_mode_2 = img_path[1].split('/')[-1].split('_')[-1]
    
    dice_total_mode_1 = 0.
    dice_total_mode_2 = 0.
    mean_dice_mode_1= 0.
    mean_dice_mode_2= 0.
    for i, file_name in enumerate(in_files):
        # logging.info("{}.Predicting image {} ...".format(i,file_name))
        path_file = [os.path.join(img_path[0], file_name+'.npy'),os.path.join(img_path[1], file_name+'.npy')]
        true_file = os.path.join(true_path, file_name+'.npy')
        namesplit = os.path.splitext(file_name)
        patientSliceID = namesplit[0]
        # img = Image.open(fn)  ##when data is image,jpg,png...
        img = [np.load(path_file[0]),np.load(path_file[1])]     ##when data is npy
        true_mask = np.load(true_file)
        pred_mask_mode_1,pred_mask_mode_2,dice_test_mode_1,dice_test_mode_2,time_s = predict_img(net,device,true_mask,init_img=img,scale_factor=1,out_threshold=0.5)
        dice_total_mode_1 += dice_test_mode_1 
        dice_total_mode_2 += dice_test_mode_2 
        
        # logging.info("Patient [{}] dice is [{:.4f}]. Time consuming is {:.4f}".format(patientSliceID,dice_test,time_s))
        if is_save_img==True:
            save_imgs(img_mode,out_path,img,true_mask,pred_mask,patientSliceID,dice_test,exp)
            
    mean_dice_mode_1 = dice_total_mode_1 / (i+1)
    mean_dice_mode_2 = dice_total_mode_2 / (i+1)
    return mean_dice_mode_1,mean_dice_mode_2

def test_volumes(net,device,img_path,true_path,in_files,out_path=None):
    img_mode_1 = img_path[0].split('/')[-1].split('_')[-1]
    img_mode_2 = img_path[1].split('/')[-1].split('_')[-1]
    
    slice_len = len(in_files)
    path_file1 = os.path.join(true_path, in_files[0]+'.npy')
    w,h = np.load(path_file1).shape
    test_img_tensor_mode_1 = np.zeros((slice_len,1,w,h),np.float32)
    test_img_tensor_mode_2 = np.zeros((slice_len,1,w,h),np.float32)
    
    image_3d_true_mode_1 = np.zeros((slice_len,w,h),np.uint8)
    image_3d_true_mode_2 = np.zeros((slice_len,w,h),np.uint8)
    
    for i, file_name in enumerate(in_files):
        # logging.info("{}.Predicting image {} ...".format(i,file_name))
        path_file = [os.path.join(img_path[0], file_name+'.npy'),os.path.join(img_path[1], file_name+'.npy')]
        true_file = os.path.join(true_path, file_name+'.npy')
        patient_name = file_name.split('.')[0]
        patient_slice = patient_name.split('_')[-1]
        patient_slice = int(patient_slice) - 1
        # img = Image.open(fn)  ##when data is image,jpg,png...
        img = [np.load(path_file[0]),np.load(path_file[1])]     ##when data is npy
        true_mask = np.load(true_file)
        # step 1/4 : path --> img_chw
        if len(img[0].shape) == 2:
            img_chw_mode_1 = np.expand_dims(img[0], axis=0)
        if len(img[1].shape) == 2:
            img_chw_mode_2 = np.expand_dims(img[1], axis=0)
            
            
        test_img_tensor_mode_1[patient_slice] = img_chw_mode_1
        test_img_tensor_mode_2[patient_slice] = img_chw_mode_2
        
        image_3d_true_mode_1[patient_slice] = true_mask
        image_3d_true_mode_2[patient_slice] = true_mask
        
    img_tensor_mode_1 = torch.tensor(test_img_tensor_mode_1).to(device,torch.float32)
    img_tensor_mode_2 = torch.tensor(test_img_tensor_mode_2).to(device,torch.float32)
    
    with torch.no_grad():
        _,_,mask_pred_mode_1,mask_pred_mode_2 = net.forward_onlyMIA(img_tensor_mode_1, img_tensor_mode_2)
        
            
    if len(mask_pred_mode_1) != 2:
        outputs_mode_1 = mask_pred_mode_1
        outputs_mode_2 = mask_pred_mode_2
    else:
        outputs_mode_1 = mask_pred_mode_1[0]
        outputs_mode_2 = mask_pred_mode_2[0]

        if torch.min(outputs_mode_1) < 0 or torch.max(outputs_mode_1) > 1:
            outputs_mode_1 = torch.sigmoid(outputs_mode_1)
        if torch.min(outputs_mode_2) < 0 or torch.max(outputs_mode_2) > 1:
            outputs_mode_2 = torch.sigmoid(outputs_mode_2)
    
    pred_mask_tensor_mode_1 = outputs_mode_1.ge(0.5).cpu().data.numpy().astype("uint8")
    pred_mask_tensor_mode_2 = outputs_mode_2.ge(0.5).cpu().data.numpy().astype("uint8")
    
    image_3d_pred_mode_1 = np.squeeze(pred_mask_tensor_mode_1,axis=1)
    image_3d_pred_mode_2 = np.squeeze(pred_mask_tensor_mode_2,axis=1)

    
    return image_3d_pred_mode_1,image_3d_pred_mode_2,image_3d_true_mode_1,image_3d_true_mode_2

def compute_3d_dice(net,device,img_path_full,true_path_full,in_files_full_list,patientID_list,out_dir):
    file_names = in_files_full_list      
    patient_dice_3d_t_mode_1 = 0.
    patient_ppv_3d_t_mode_1 = 0.
    patient_sen_3d_t_mode_1 = 0.
    patient_asd_3d_t_mode_1 = 0.
    patient_hd95_3d_t_mode_1 = 0.
    
    patient_dice_3d_t_mode_2 = 0.
    patient_ppv_3d_t_mode_2 = 0.
    patient_sen_3d_t_mode_2 = 0.
    patient_asd_3d_t_mode_2 = 0.
    patient_hd95_3d_t_mode_2 = 0.
    for i , id in enumerate(patientID_list):
        img_names = list(filter(lambda x: x.startswith(id), file_names))
        imgnp_3d_pred_mode_1,imgnp_3d_pred_mode_2,imgnp_3d_true_mode_1,imgnp_3d_true_mode_2 = test_volumes(net,device,img_path_full,true_path_full,in_files=img_names)
        patient_dice_3d_mode_1, patient_ppv_3d_mode_1, patient_sen_3d_mode_1 = dice_3(imgnp_3d_pred_mode_1, imgnp_3d_true_mode_1)
        patient_dice_3d_mode_2, patient_ppv_3d_mode_2, patient_sen_3d_mode_2 = dice_3(imgnp_3d_pred_mode_2, imgnp_3d_true_mode_2)
        
        
        patient_dice_3d_t_mode_1 += patient_dice_3d_mode_1
        patient_ppv_3d_t_mode_1 += patient_ppv_3d_mode_1
        patient_sen_3d_t_mode_1 += patient_sen_3d_mode_1
        
        patient_dice_3d_t_mode_2 += patient_dice_3d_mode_2
        patient_ppv_3d_t_mode_2 += patient_ppv_3d_mode_2
        patient_sen_3d_t_mode_2 += patient_sen_3d_mode_2
        
        if np.sum(imgnp_3d_pred_mode_1)==0:
            asd_3d_mode_1 = 0
            hd95_3d_mode_1 = 0
        else:
            asd_3d_mode_1 = metric.binary.asd(imgnp_3d_pred_mode_1, imgnp_3d_true_mode_1)
            hd95_3d_mode_1 = metric.binary.hd95(imgnp_3d_pred_mode_1, imgnp_3d_true_mode_1)
        
        if np.sum(imgnp_3d_pred_mode_2)==0:
            asd_3d_mode_2 = 0
            hd95_3d_mode_2 = 0
        else:
            asd_3d_mode_2 = metric.binary.asd(imgnp_3d_pred_mode_2, imgnp_3d_true_mode_2)
            hd95_3d_mode_2 = metric.binary.hd95(imgnp_3d_pred_mode_2, imgnp_3d_true_mode_2)
            
            
        patient_asd_3d_t_mode_1 += asd_3d_mode_1
        patient_hd95_3d_t_mode_1 += hd95_3d_mode_1
        
        patient_asd_3d_t_mode_2 += asd_3d_mode_2
        patient_hd95_3d_t_mode_2 += hd95_3d_mode_2
        
        logging.info("{} Patient {}'s 3d dice mode 1 is {:.4f}".format(i+1,id,patient_dice_3d_mode_1))
        logging.info("{} Patient {}'s 3d dice mode 2 is {:.4f}".format(i+1,id,patient_dice_3d_mode_2))
        
    mean_dice_all_3d_mode_1 = patient_dice_3d_t_mode_1 / (i+1)
    mean_ppv_all_3d_mode_1 = patient_ppv_3d_t_mode_1 / (i+1)
    mean_sen_all_3d_mode_1 = patient_sen_3d_t_mode_1 / (i+1)
    mean_asd_all_3d_mode_1 = patient_asd_3d_t_mode_1 / (i+1)
    mean_hd95_all_3d_mode_1 = patient_hd95_3d_t_mode_1 / (i+1)
    
    mean_dice_all_3d_mode_2 = patient_dice_3d_t_mode_2 / (i+1)
    mean_ppv_all_3d_mode_2 = patient_ppv_3d_t_mode_2 / (i+1)
    mean_sen_all_3d_mode_2 = patient_sen_3d_t_mode_2 / (i+1)
    mean_asd_all_3d_mode_2 = patient_asd_3d_t_mode_2 / (i+1)
    mean_hd95_all_3d_mode_2 = patient_hd95_3d_t_mode_2 / (i+1)
    
    
    logging.info("=================Mode 1===============")
    logging.info("Mean 3d asd on all patients mode 1: {:.4f} ".format(mean_asd_all_3d_mode_1))
    logging.info("Mean 3d hd95 on all patients mode 1: {:.4f} ".format(mean_hd95_all_3d_mode_1))
    logging.info("Mean 3d ppv on all patients mode 1: {:.4f} ".format(mean_ppv_all_3d_mode_1))
    logging.info("Mean 3d sen on all patients mode 1: {:.4f} ".format(mean_sen_all_3d_mode_1))
    logging.info("Mean 3d dice on all patients mode 1: {:.4f} ".format(mean_dice_all_3d_mode_1))
    
    logging.info("=================Mode 2===============")
    logging.info("Mean 3d asd on all patients mode 2: {:.4f} ".format(mean_asd_all_3d_mode_2))
    logging.info("Mean 3d hd95 on all patients mode 2: {:.4f} ".format(mean_hd95_all_3d_mode_2))
    logging.info("Mean 3d ppv on all patients mode 2: {:.4f} ".format(mean_ppv_all_3d_mode_2))
    logging.info("Mean 3d sen on all patients mode 2: {:.4f} ".format(mean_sen_all_3d_mode_2))
    logging.info("Mean 3d dice on all patients mode 2: {:.4f} ".format(mean_dice_all_3d_mode_2))
    
    
    return mean_dice_all_3d_mode_1, mean_dice_all_3d_mode_2

if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    gpu_list = [0] #[0,1]
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    """Test master directory """
    predict_file = 'res-t2_t1ce-OnlyMIA_BraTs-seed-2-CMC-10%_seed_1111/'
    base_dir = './' + predict_file
    log_path = os.path.join(base_dir, 'predict_t1ce.log') 
    set_logging(log_path=log_path)

    data_path = "A:\Dat\semi-CML-public\dataset\BraTS_slice"
    img_mode_1 = 't2'
    img_mode_2 = 't1ce'
    out_path = os.path.join(base_dir, 'outputs_new') 
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    model_path = os.path.join(base_dir, 'model_CMCNet_last.pth')  
    logging.info("Loading model {}".format(model_path))
    net = torch.load(model_path, map_location=device,weights_only=False)    
    net.to(device=device)
    net.eval()
    logging.info("Model loaded !")
    img_path_mode_1 = '{}/imgs_{}'.format(data_path,img_mode_1)
    img_path_mode_2 = '{}/imgs_{}'.format(data_path,img_mode_2)
    img_path = [img_path_mode_1, img_path_mode_2]
    true_path = '{}/masks'.format(data_path)

    """1. Test some images and save them"""
    test_list_path = data_path+'/randP1_slice_nidus_val.list'
    in_files_list = read_list(test_list_path)
    # logging.info(in_files)
    logging.info("data number:{}".format(len(in_files_list)))
    mean_dice_mode_1, mean_dice_mode_2 = test_images(net,device,
                    img_path,true_path,in_files_list,out_path,is_save_img=False,exp='')
    
    logging.info("mean dice mode 1 is {:.4f}".format(mean_dice_mode_1))
    logging.info("mean dice mode 2 is {:.4f}".format(mean_dice_mode_2))

    """2. Calculate mean 3D Dice"""
    time_tic_test = time.time()
    full_test_list_path = '{}/randP1_slice_all_val.list'.format(data_path)
    in_files_full_list = read_list(full_test_list_path)
    patientID_test_list_path = '{}/randP1_volume_val.list'.format(data_path)
    patientID_list = read_list(patientID_test_list_path)
    # logging.info(in_files_full_list)
    mean_dice_all_3d_mode_1, mean_dice_all_3d_mode_2 = compute_3d_dice(net,device,
                            img_path,true_path,in_files_full_list,patientID_list,out_dir=base_dir)
    time_end_test = time.time()
    logging.info("3D Dice test time :{:.2f} min".format((time_end_test-time_tic_test)/60))
    # logging.info("Mean 3d dice on all patients mode 1 :{:.4f} ".format(mean_dice_all_3d_mode_1))
    # logging.info("Mean 3d dice on all patients mode 2 :{:.4f} ".format(mean_dice_all_3d_mode_2))
        
