import os
import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

# class MyDataset(Dataset):
#     def __init__(self, data_path, subset='train'):
#         super(MyDataset, self).__init__()
#         labels = os.listdir(data_path)
#         self.subset = subset
#         self.data_paths = []
#         self.data_labels = []
#         for label in labels:
#             image_folder = os.path.join(data_path, label)
#             for image_name in os.listdir(image_folder):
#                 self.data_labels.append(int(label))
#                 self.data_paths.append(os.path.join(image_folder, image_name))

#     def __getitem__(self, index):
#         path = self.data_paths[index]
#         img = np.load(path)
#         #img = np.expand_dims(img, 2)
#         if self.subset == 'test':
#             img = img.transpose(0, 3, 1, 2)
#             img = torch.from_numpy(img).float()
#             label = self.data_labels[index] 
#             label = torch.from_numpy(np.array([label for i in range(img.shape[0])])).long()
#         else:
#             img = img.transpose(2,0,1)
#             img = torch.from_numpy(img).float()
#             label = self.data_labels[index] 
#         # import pdb
#         # pdb.set_trace()
#         return {'img': img, 'label': label}

#     def __len__(self):
#         return len(self.data_paths)



class MyDataset(Dataset):
    def __init__(self, data_path, subset='train'):
        super(MyDataset, self).__init__()
        self.subset = subset
        if subset == 'train':
            labels = os.listdir(data_path)
            self.data_paths = []
            self.data_labels = []
            for label in labels:
                image_folder = os.path.join(data_path, label)
                for image_name in os.listdir(image_folder):
                    self.data_labels.append(int(label))
                    self.data_paths.append(os.path.join(image_folder, image_name))
        else:
            images = os.listdir(data_path)
            # import pdb
            # pdb.set_trace()
            self.data_paths = [os.path.join(data_path, x)  for x in images if 'img' in x]
            self.data_labels = [os.path.join(data_path, x)  for x in images if 'label' in x]
            self.data_paths.sort()
            self.data_labels.sort()


    def __getitem__(self, index):
        path = self.data_paths[index]
        img = np.load(path)
        #img = np.expand_dims(img, 2)
        if self.subset == 'test':
            img = img.transpose(0, 3, 1, 2)
            img = torch.from_numpy(img).float()
            label = np.load(self.data_labels[index]) 
            label = torch.from_numpy(label).long()
        else:
            img = img.transpose(2,0,1)
            img = torch.from_numpy(img).float()
            label = self.data_labels[index] 
        # import pdb
        # pdb.set_trace()
        return {'img': img, 'label': label}

    def __len__(self):
        return len(self.data_paths)