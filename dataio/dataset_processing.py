import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import sys
import os
import imageio 
import imgaug.augmenters as iaa
import glob 
import natsort 

from .utils import *

class DatasetInput(data.Dataset):
    def __init__(self, data_sources, fold_id, fold_mean, fold_std, in_h, in_w,
                 mode='train', tcga_path=None):

        # Check valid data directories
        if not os.path.exists(data_sources.img_data_dir):
            sys.exit("Invalid images directory %s" %data_sources.img_data_dir)
        if not os.path.exists(data_sources.fold_splits):
            sys.exit("Invalid fold splits directory %s" %data_sources.fold_splits)
        if not os.path.exists(data_sources.labels_file):
            sys.exit("Invalid feature labels path %s" %data_sources.labels_file)

        self.img_data_dir = data_sources.img_data_dir
        self.fold_splits = data_sources.fold_splits
        self.labels_file = data_sources.labels_file
        self.fold_id = fold_id
        self.fold_mean = fold_mean
        self.fold_std = fold_std
        self.in_height = in_h 
        self.in_width = in_w
        self.mode = mode

        # Data files
        if self.mode == 'train':
            patient_subset_txt = self.fold_splits + '/' + str(self.fold_id) + '_train.txt'
        elif self.mode == 'test':
            patient_subset_txt = self.fold_splits + '/' + str(self.fold_id) + '_test.txt'
        elif self.mode == 'val':
            patient_subset_txt = self.fold_splits + '/' + str(self.fold_id) + '_val.txt'
        elif self.mode == 'tcga':
            img_paths = glob.glob(tcga_path + '/**/*.jpg', recursive=True)
            img_paths = natsort.natsorted(img_paths)
            print(len(img_paths))
        else:
            sys.exit('Invalid mode - choose from train, val, or test')

        if self.mode != 'tcga':
            img_paths = get_data_dirs_split(patient_subset_txt, self.img_data_dir)

            img_paths_reformat = [x.split('/')[-2] + '_' + os.path.basename(x).split('.')[0] for x in img_paths]

            # All the labels for this data subset
            data_df = pd.read_csv(self.labels_file, index_col=0)
            
            # Some img spots are not in the tsv files
            self.img_paths_reformat = list(set(img_paths_reformat) & set(data_df.index.to_list()))
            self.labels_df = data_df.loc[self.img_paths_reformat]
            
            self.img_paths = [(self.img_data_dir + '/' + x.split('_')[0] + '/' + x.split('_')[1] + '.jpg') for x in self.img_paths_reformat]
        else:
            self.img_paths = img_paths
            self.img_paths_reformat = [x.split('/')[-2] + '_' + os.path.basename(x).split('.')[0] for x in img_paths]


    def augment_data(self, batch_raw):
        batch_raw = np.expand_dims(batch_raw, 0)

        # Original, horizontal
        random_flip = np.random.randint(2, size=1)[0]
        # 0, 90, 180, 270
        random_rotate = np.random.randint(4, size=1)[0]

        # Flips
        if random_flip == 0:
            batch_flip = batch_raw*1
        else:
            batch_flip = iaa.Flipud(1.0)(images=batch_raw)
                
        # Rotations
        if random_rotate == 0:
            batch_rotate = batch_flip*1
        elif random_rotate == 1:
            batch_rotate = iaa.Rot90(1, keep_size=True)(images=batch_flip)
        elif random_rotate == 2:
            batch_rotate = iaa.Rot90(2, keep_size=True)(images=batch_flip)
        else:
            batch_rotate = iaa.Rot90(3, keep_size=True)(images=batch_flip)
        
        images_aug_array = np.array(batch_rotate)

        return images_aug_array


    def normalise_images(self, imgs):        
        return (imgs - self.fold_mean)/self.fold_std


    def __len__(self):
            'Denotes the total number of samples'
            return len(self.img_paths)


    def __getitem__(self, index):
            'Generates one sample of data'
            img_path = self.img_paths[index]
            ID = self.img_paths_reformat[index]

            img = imageio.imread(img_path)

            assert(img.shape[0] == self.in_height)
            assert(img.shape[1] == self.in_width)

            img = self.normalise_images(img)

            if self.mode == 'train':
                img = np.squeeze(self.augment_data(img))
            
            img = np.moveaxis(img, -1, 0)

            if self.mode != 'tcga':
                # Get labels
                labels = self.labels_df.loc[ID].values.tolist()
                labels = np.array(labels)
                labels_torch = torch.from_numpy(labels).float()

                # Convert to tensor
                img_torch = torch.from_numpy(img).float()

                if self.mode == 'train':
                    return img_torch, labels_torch
                else:
                    return img_torch, labels_torch, ID
                    
            else:
                # Convert to tensor
                img_torch = torch.from_numpy(img).float()

                return img_torch, ID