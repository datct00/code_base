import os
import torch
import random
import logging
import itertools
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class MultiSemiDataSets(Dataset):
    def __init__(self, data_dir=None, mode='train',img_mode='flair-t1',mask_name='mask',list_name='slice_nidus_all.list',images_rate=1, transform=None):
        self._data_dir = data_dir
        self.sample_list = []
        self.mode = mode
        self.img_mode = img_mode
        self.mask_name = mask_name
        self.list_name = list_name
        self.transform = transform
        self.img_mode1 = self.img_mode.split('_')[0]
        self.img_mode2 = self.img_mode.split('_')[1]

        list_path = os.path.join(self._data_dir,self.list_name)

        with open(list_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.sample_list.append(line)      
        # print(self.mode,self.sample_list)              
        logging.info(f'Creating total {self.mode} dataset with {len(self.sample_list)} examples')                

        if images_rate !=1 and self.mode == "train":
            images_num = int(len(self.sample_list) * images_rate)
            self.sample_list = self.sample_list[:images_num]
        logging.info(f"Creating factual {self.mode} dataset with {len(self.sample_list)} examples")
            

    def __len__(self):
        return len(self.sample_list)
    def __sampleList__(self):
        return self.sample_list

    def __getitem__(self, idx):
        if self.mode=='val_3d':
            case = idx
        else:
            case = self.sample_list[idx]

        mode1_np_path = os.path.join(self._data_dir,'imgs_{}/{}.npy'.format(self.img_mode1,case))
        mode2_np_path = os.path.join(self._data_dir,'imgs_{}/{}.npy'.format(self.img_mode2,case))
    
        mask_np_path = os.path.join(self._data_dir,'{}/{}.npy'.format(self.mask_name,case))

        mode1_np = np.load(mode1_np_path)
        mode2_np = np.load(mode2_np_path)
        mask_np = np.load(mask_np_path)
        
        if len(mode1_np.shape) == 2:
            mode1_np = np.expand_dims(mode1_np, axis=0)
        if len(mode2_np.shape) == 2:
            mode2_np = np.expand_dims(mode2_np, axis=0)
        if len(mask_np.shape) == 2:
            mask_np = np.expand_dims(mask_np, axis=0)
        sample = {'mode1': mode1_np.copy(),'mode2': mode2_np.copy(), 'mask': mask_np.copy(),'idx':case}
        return sample
        
class PatientBatchSampler(Sampler):
    def __init__(self, slices_list,patientID_list):
        self.slices_list = slices_list
        self.patientID_list = patientID_list
        assert len(self.slices_list) >= len(self.patientID_list) > 0

    def __iter__(self):
        return (
            list(filter(lambda x: x.startswith(id), self.slices_list))
            for i,id
            in enumerate(self.patientID_list)
        )

    def __len__(self):
        return len(self.patientID_list)

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the labeled indices.
    During the epoch, the unlabeled indices are iterated through
    as many times as needed.
    """

    def __init__(self,labeled_idxs, unlabeled_idxs, batch_size, unlabeled_batch_size,shuffle=True):
        self.shuffle=shuffle
        self.labeled_idxs = labeled_idxs
        self.unlabeled_idxs = unlabeled_idxs
        self.labeled_batch_size = batch_size - unlabeled_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size

        assert len(self.labeled_idxs) >= self.labeled_batch_size > 0
        assert len(self.unlabeled_idxs) >= self.unlabeled_batch_size > 0

    def __iter__(self):
        if self.shuffle == True:
            labeled_iter = iterate_once(self.labeled_idxs) 
            unlabeled_iter = iterate_eternally(self.unlabeled_idxs)
        else:
            labeled_iter = self.labeled_idxs
            unlabeled_iter = self.unlabeled_idxs
        
        return (
            labeled_batch + unlabeled_batch
            for (labeled_batch, unlabeled_batch)
            in zip(grouper(labeled_iter, self.labeled_batch_size),
                   grouper(unlabeled_iter, self.unlabeled_batch_size))
        )

    def __len__(self):
        return len(self.labeled_idxs) // self.labeled_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
