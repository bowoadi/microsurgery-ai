# %%
import random
import math
import os
import time
import torch
import torch.utils.data as data
import torchvision.transforms.functional as fn

from numpy.lib.stride_tricks import as_strided
from torchvision import transforms
from PIL import Image, ImageOps
from tqdm import tqdm

import numpy as np
from torch import Tensor
from  torch.nn.modules.pooling import _MaxPoolNd
from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t, _ratio_3_t, _ratio_2_t

import json

# %%
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

# %%
classes = {
    "Efficiency - Does not pull needle out of field_1":0,
    "Efficiency - Many wasted moves_1":0,
    "Efficiency - Regrasps multiple times_1":0,
    
    "Efficiency - Sometimes pulls needle out of field_3":0,
    "Efficiency - Regrasps occasionally_3":0,
    "Efficiency - Some wasted moves_3":0,
    
    "Efficiency - Always pulls out of field_5" : 0,
    "Efficiency - Grasps once only_5":0,
    "Efficiency - No wasted moves_5":0,

    "Handling - Does not bolster_1": 1,
    "Handling - Grasps tip of needle_1":1,
    "Handling - Multiple passes_1":1,
    "Handling - Pulls needle out not on the curve_1":1,
    
    "Handling - A few passes_3":1,
    "Handling - Sometimes bolsters_3":1,
    "Handling - Sometimes grasps the tip_3":1,
    "Handling - Sometimes pulls needle out on the curve_3": 1,
    
    "Handling - Always bolsters_5": 1,
    "Handling - Always pull needle out on the curve_5": 1,
    "Handling - Never grasps the tip_5": 1,
    "Handling - Single passes_5": 1,

    "Preparation - Ends set up poorly in approximating clamp_1":2,
    "Preparation - Forgets Background_1": 2,
    "Preparation - Forgets dilatation_1": 2,
    "Preparation - No adventitial stripping_1": 2,
    
    "Preparation - Excessive inadequate adventitial stripping_3": 2,
    "Preparation - Rough dilatation_3": 2,
    
    "Preparation - Approximating clamp applied correctly_5": 2,
    "Preparation - Background in place_5": 2,
    "Preparation - Clean adventitial stripping_5": 2,
    "Preparation - Gentle dilatation_5": 2,

    "Quality of Knot - Cut ends too long short_1": 3,
    "Quality of Knot - Loose_1": 3,
    "Quality of Knot - Not square_1":3,
    
    "Quality of Knot - Cut ends OK length_3": 3,
    "Quality of Knot - Partially square_3": 3,
    "Quality of Knot - Somewhat loose_3": 3,
    
    "Quality of Knot - Cut ends proper length_5": 3,
    "Quality of Knot - Snug_5": 3,
    "Quality of Knot - Square_5": 3,
}

classes_dict = classes
main_class = f"expertise"

print(main_class)

# %% [markdown]
# # HYPERPARAMETER

# %%
# len(os.listdir("./data_full_backup/"))

# %%
video_sec = 12
FPS = 30
FRAME_COUNT_TOTAL = video_sec * FPS
RANDOM_SEED = 0
CLIP_LIMIT = 7
DROPOUT = 0.1

augmented = False
fuzzy = True
oversample = True
tsn = True
weighted = True
temperature = 1.3

loss_weight = [2.0, 1.0, 2.0]
batch_size = 1
val_ratio = 0.2
num_workers = 0
n_classes = 3
lr_rate = 1e-3
weight_decay = 1e-9
frame_skip = 5
frame_count = FRAME_COUNT_TOTAL // frame_skip
frame_size = 256
log_interval = 10
n_epochs = 30

root_path = f"./data_frameskip_{frame_skip}_{video_sec}s_{frame_size}size"

metadata_dir = "./model_training_data"
model_dir = "./models"
frame = f"frame_{frame_count}_skip_{frame_skip}_size_{frame_size}"
sub_class = f"exp-{'tsn-' if tsn else ''}resnet10_softmax"
if fuzzy:
    sub_class += "-fuzzy"
if augmented:
    sub_class += "-augmented"
if oversample:
    sub_class += "-oversampled"
if weighted:
    sub_class += "-weighted"
if DROPOUT != 0:
    sub_class += f"-dropout_{DROPOUT}"
if temperature != 1:
    sub_class += f"-temperature_{temperature}"

# Default no resume or pretrained
resume_path = None
continue_from = None
pretrained_path = None

# pretrain_path = os.path.join("jigsaws_models", "best_train_model_frame_180_skip_2_size_256_knot_tying_exp-tsn-resnet10-fuzzy-dropout_0.3.pth")
# resume_path = pretrain_path
# continue_from = 0

if pretrained_path:
    sub_class += "-pretrained"

# resume_path = os.path.join(model_dir, "best_train_model_"+frame+"_"+main_class+"_"+sub_class+".pth")
# continue_from = 31

# device preparation
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
print(device)

# %%
random.seed(RANDOM_SEED)

# %% [markdown]
# # DATA LOADER

# %%
def read_dataset(root_path, has_subclass=False, val=False, val_ratio=0.2):
    videos_per_class = {}
    frames_total = []
    if has_subclass:
        # TODO: Remove or fix this
        for video_class_folder in sorted(os.listdir(root_path)):
            if "." not in video_class_folder:
                for video_sub_class_folder in sorted(os.listdir(root_path+"/"+video_class_folder)):
                    video_name_list = sorted(os.listdir(root_path+"/"+video_class_folder+"/"+video_sub_class_folder))
                    if val:
                        video_name_list = video_name_list[:int(len(video_name_list)*val_ratio)]
                    else:
                        video_name_list = video_name_list[int(len(video_name_list)*val_ratio):]

                    for video_name in video_name_list:
                        # print(len(os.listdir(root_path+"/"+video_class_folder+"/"+video_sub_class_folder+"/"+video_name)))
                        if len(os.listdir(root_path+"/"+video_class_folder+"/"+video_sub_class_folder+"/"+video_name))>= 360:
                            frames_total.append(len(os.listdir(root_path+"/"+video_class_folder+"/"+video_sub_class_folder+"/"+video_name)))
                            data = {
                                "path" : root_path+"/"+video_class_folder+"/"+video_sub_class_folder+"/"+video_name,
                                "label": video_class_folder
                            }
                            videos_per_class.append(data)
    else:
        for video_class_folder in sorted(os.listdir(root_path)):
            videos = []
            if "." not in video_class_folder:
                video_name_list = sorted(os.listdir(root_path+"/"+video_class_folder))
                
                # Skip unused class
                if video_class_folder not in classes_dict:
                    continue
                
                for video_name in video_name_list:
                    n_frame = len(os.listdir(root_path+"/"+video_class_folder+"/"+video_name))
                    if n_frame >= frame_count:
                        frames_total.append(n_frame)
                        
                        # Add start and end frame according to the frame_count
                        stop = False
                        for start_idx in range(0, n_frame, frame_count):
                            # If there are overhead frames, sample the frame
                            if start_idx + frame_count*2 > n_frame:
                                start_idx = random.randint(0, max(start_idx, n_frame-frame_count-1))
                                stop = True

                            data = {
                                    "path" : root_path+"/"+video_class_folder+"/"+video_name,
                                    "label": video_class_folder,
                                    "n_frame": n_frame,
                                    "index": (start_idx, start_idx+frame_count)
                                }
                            videos.append(data)

                            if stop:
                                break

            # Shuffle so we get different validation each time
            random.shuffle(videos)

            if val:
                videos = videos[:int(len(videos)*val_ratio)]
            else:
                videos = videos[int(len(videos)*val_ratio):]
            videos_per_class.setdefault(classes_dict[video_class_folder], []).extend(videos)

    max_frame = max(frames_total)
    min_frame = min(frames_total)
    # print(frames_total)
    # print("total_frame: ", sum(frames_total))
    # print("total_video: ", len(frames_total))
    # print("max_frame: ", max_frame)
    # print("min_frame: ", min_frame)
    # return videos, max_frame, min_frame
    return list(videos_per_class.values()), max_frame, min_frame
# read_dataset test
# read_dataset(root_path, val_ratio=val_ratio)

def read_dataset_tsn(root_path, has_subclass=False, val=False, val_ratio=0.2):
    videos_per_class = {}
    frames_total = []
    for video_class_folder in sorted(os.listdir(root_path)):
        if "." not in video_class_folder:
            video_name_list = sorted(os.listdir(root_path+"/"+video_class_folder))
            # Skip unused class
            if video_class_folder not in classes_dict:
                continue
                
            video_not_parted = {}
            for video_name in video_name_list:
                video_name_without_part = video_name.split("_")[0]
                n_frame = len(os.listdir(root_path+"/"+video_class_folder+"/"+video_name))
                if n_frame >= frame_count:
                    frames_total.append(n_frame)
                    clips = []
                    
                    # Add start and end frame according to the frame_count
                    for start_idx in range(0, n_frame, frame_count):
                        # If there are overhead frames, sample the frame
                        # if start_idx + frame_count*2 > n_frame:
                        #     start_idx = random.randint(0, max(start_idx, n_frame-frame_count-1))
                        #     stop = True

                        data = {
                                "path" : root_path+"/"+video_class_folder+"/"+video_name,
                                "label": video_class_folder,
                                "n_frame": n_frame,
                                "index": (start_idx, start_idx+frame_count)
                            }
                        clips.append(data)

                    video_not_parted.setdefault(video_name_without_part, []).extend(clips)
            
            # Separate into multiple clips if video is too long
            excluded_videos = []
            for video_name, clips in video_not_parted.items():
                if len(clips) > CLIP_LIMIT:
                    for i in range(0, len(clips), CLIP_LIMIT):
                        videos_per_class.setdefault(classes_dict[video_class_folder], []).append(clips[i:i+CLIP_LIMIT])
                    # Add video to be excluded on final video addition
                    excluded_videos.append(video_name)

            # Exclude too long videos
            for video_name in excluded_videos:
                del video_not_parted[video_name]

            # Add normal videos
            videos_per_class.setdefault(classes_dict[video_class_folder], []).extend(video_not_parted.values())

    # Shuffle so we get different validation for each seed
    random.shuffle(videos_per_class)

    # Split train and val
    if val:
        videos_per_class = {k: v[:int(len(v)*val_ratio)] for k, v in videos_per_class.items()}
    else:
        videos_per_class = {k: v[int(len(v)*val_ratio):] for k, v in videos_per_class.items()}

    max_frame = max(frames_total)
    min_frame = min(frames_total)
    
    return list(videos_per_class.values()), max_frame, min_frame
# read_dataset test
# print(read_dataset_tsn(root_path, val_ratio=val_ratio))

# %%
class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        # for index in out:
        #     if len(out) >= self.size:
        #         break
        #     out.append(index)

        return out

# %%
from concurrent.futures import ThreadPoolExecutor

class MicrosurgeryData(data.Dataset):
    def __init__(self,
                root_path,
                total_frame = 16,
                has_subclass=True,
                val=False, 
                val_ratio=0.2,
                augmented=False,
                oversample=False):
        self.data, max_frame, min_frame = read_dataset(root_path,has_subclass, val, val_ratio)
        self.augmentations = None
        if augmented:
            self.augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.RandomResizedCrop(224),
            ])

        if oversample:
            self._balance_all_class()
        
        # Flatten the video per type into one list
        self.data = [item for sublist in self.data for item in sublist]

        # Preshuffle the data
        random.shuffle(self.data)

    def _frame_loader(self, path):
        frame = Image.open(path).convert('RGB')
        if self.augmentations != None:
            frame = self.augmentations(frame)
        return frame
    
    def _balance_all_class(self): 
        # Find the class with the most videos
        max_class = max(self.data, key=lambda x: len(x))
        max_class_count = len(max_class)
        print("Max class count: ", max_class_count)

        # Oversample the other classes
        for i, video_class in enumerate(self.data):
            print(f"Before: class {i} count: {len(video_class)}")
            if len(video_class) < max_class_count:
                diff = max_class_count - len(video_class)
                for j in range(diff):
                    video_class.append(video_class[j % len(video_class)])
            print(f"After: class {i} count: {len(video_class)}")
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]["path"]
        
        frame_indices = list(range(*self.data[index]["index"]))
        
        # frame_indices = self.temporal_transform(frame_indices)
        clip = []
        target =[]

        # Apply temporal transform to get selected frame indices
        # frame_indices = self.temporal_transform(len(frame_files))
        
        with ThreadPoolExecutor() as executor:
            # clip = executor.map(lambda idx: Image.open(os.path.join(path, 'image_{}.jpg'.format(idx))).convert('RGB'), frame_indices)
            clip = executor.map(lambda idx: self._frame_loader(os.path.join(path, 'image_{}.jpg'.format(idx))), frame_indices)

        clip = list(clip)

        # Vectorized processing of clip
        clip = torch.from_numpy(np.array(clip).astype(np.float32)).permute(3, 0, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

        # Apply frame-level transformations
        clip /= 255.
                
        # clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        target = classes_dict.get(self.data[index]["label"])
        
        return clip, target

    def __len__(self):
        return len(self.data)
    
class TSNMicrosurgeryData(data.Dataset):
    def __init__(self,
                root_path,
                total_frame = 16,
                has_subclass=True,
                val=False, 
                val_ratio=0.2,
                augmented=False,
                oversample=False):
        self.data, max_frame, min_frame = read_dataset_tsn(root_path,has_subclass, val, val_ratio)
        self.augmentations = None
        if augmented:
            self.augmentations = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.RandomResizedCrop(224),
            ])

        if oversample:
            self._balance_all_class()
        
        # Flatten the video per type into one list
        self.data = [item for sublist in self.data for item in sublist]

        # Preshuffle the data
        random.shuffle(self.data)

    def _frame_loader(self, path):
        frame = Image.open(path).convert('RGB')
        if self.augmentations != None:
            frame = self.augmentations(frame)
        return frame
    
    def _balance_all_class(self): 
        # Find the class with the most videos
        max_class = max(self.data, key=lambda x: len(x))
        max_class_count = len(max_class)
        print("Max class count: ", max_class_count)

        # Oversample the other classes
        for i, video_class in enumerate(self.data):
            print(f"Before: class {i} count: {len(video_class)}")
            if len(video_class) < max_class_count:
                diff = max_class_count - len(video_class)
                for j in range(diff):
                    video_class.append(video_class[j % len(video_class)])
            print(f"After: class {i} count: {len(video_class)}")
    
    def _load_clip(self, clip_data):
        path = clip_data["path"]
            
        # n_frame = len(sorted(os.listdir(path)))
        frame_indices = list(range(*clip_data["index"]))
        
        # frame_indices = self.temporal_transform(frame_indices)
        clip = []

        # Apply temporal transform to get selected frame indices
        # frame_indices = self.temporal_transform(len(frame_files))
        
        with ThreadPoolExecutor() as executor:
            clip = executor.map(lambda idx: self._frame_loader(os.path.join(path, 'image_{}.jpg'.format(idx))), frame_indices)

        clip = list(clip)

        # Vectorized processing of clip
        clip = torch.from_numpy(np.array(clip).astype(np.float32)).permute(3, 0, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

        # Apply frame-level transformations
        # clip = transforms.Normalize(self.mean, self.std)(clip)
        clip /= 255.

        return clip


    def __getitem__(self, index):
        video_clips_all = []
        
        with ThreadPoolExecutor() as executor:
            video_clips_all = executor.map(self._load_clip, self.data[index])
            
        video_clips_all = list(video_clips_all)
        video_clips_all = torch.stack(video_clips_all, 0)
        target = classes_dict.get(self.data[index][0]["label"])
        
        return video_clips_all, target

    def __len__(self):
        return len(self.data)

# %%
torch.multiprocessing.set_start_method('spawn', force=True)

# %%
if tsn:
    train_data = TSNMicrosurgeryData(root_path, total_frame=frame_count, has_subclass=False, 
                                    val_ratio=val_ratio, augmented=augmented, oversample=oversample)
    val_data = TSNMicrosurgeryData(root_path, total_frame=frame_count, has_subclass=False, 
                                    val=True, val_ratio=val_ratio)
else:
    train_data = MicrosurgeryData(root_path, total_frame=frame_count, has_subclass=False, 
                                val_ratio=val_ratio, augmented=augmented, oversample=oversample)
    val_data = MicrosurgeryData(root_path, total_frame=frame_count, has_subclass=False, 
                                val=True, val_ratio=val_ratio)
    
train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

print("Train data: ", len(train_data))
print("Val data: ", len(val_data))

# %%
import time
if num_workers == 0 and False:
    dataloader_iterator = iter(train_loader)
    # longest = 0
    for i in range(5):
        try:
            start = time.time()
            train_features, train_labels = next(dataloader_iterator)
            end = time.time()
            print(train_features.shape)
            # if train_features.shape[1] > longest:
            #     longest = train_features.shape[1]
            print(f"Time taken: {end-start}")
            # print(f"Feature batch shape: {train_features.size()}")
            # print(f"Labels batch shape: {train_labels.size()}")
        except StopIteration:
            dataloader_iterator = iter(train_loader)
            data, target = next(dataloader_iterator)

    # dataloader_iterator = iter(val_loader)
    # for i in range(1):
    #     try:
    #         val_features, val_labels = next(dataloader_iterator)
    #         print(f"Feature batch shape: {val_features.size()}")
    #         print(f"Labels batch shape: {val_labels.size()}")
    #     except StopIteration:
    #         dataloader_iterator = iter(val_loader)
    #         data, target = next(dataloader_iterator)

# %% [markdown]
# # MODELLING

# %% [markdown]
# ## FUZZY LOGIC

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

# %%
## Fuzzy membership function u1
def member_fun_u1(pixel):
    r_max = 6
    d = 3
    c = 1
    res_pixel = 0   
    
    if pixel>d:
        res_pixel = 0
    elif c<=pixel and pixel<=d:
        p_1 = d - pixel
        p_2 = d-c
        res_pixel = p_1/p_2
    elif pixel < c:
        res_pixel = 1

    return res_pixel

## Fuzzy membership function u2
def member_fun_u2(pixel):
    r_max = 6
    a = 1.5
    m = 3
    b =  m + a
    res_pixel = 0

    if pixel<=a:
        res_pixel = 0
    elif a<=pixel and pixel<=m:
        p_1 = pixel - a
        p_2 = m-a
        res_pixel = p_1/p_2
#     elif m== b:
#         res_pixel = 0

    return res_pixel

## Fuzzy membership function u3
def member_fun_u3(pixel):
    r_max = 6
    r = 3
    q = 4.5
    res_pixel = 0
    
    if pixel < r:
        res_pixel = 0
    elif r <= pixel and pixel <= q:
        p_1 = pixel - r
        p_2 = q - r
        res_pixel = p_1/p_2
    elif pixel > q:
        res_pixel = 1

    return res_pixel

# %%

def get_max_membership_sum_function(a, b, c):
    if a >= b and a>=c:
        return 1
    elif b>=a and b>=c:
        return 2
    else :
        return 3  

# %%
def fuzzy_pooling_2d(AA, kernel_size, stride):
    # convert tensor to numpy 
    A = torch.Tensor.cpu(AA).detach().numpy()[:,:,:,:]
    # define the image size , channel size
    batch,x1,x2,x3,x4 = A.shape

    new_mat_size1 = (x3 - kernel_size)//stride + 1
    new_mat_size2 = (x4 - kernel_size)//stride + 1

    #print(new_mat_size1)

    pool_ar = np.zeros((x1,x2,new_mat_size1,new_mat_size2))

    batch_result = []
    for b_idx in range(0, batch):
        for image_i in range(0, x1):
            for channel_j in range(0,x2):
                mat = A[b_idx][image_i][channel_j]
                
                # now do the pooling work

                # divide the matrix into k*k kernel
                output_shape = ((mat.shape[0] - kernel_size)//stride + 1,(mat.shape[1] - kernel_size)//stride + 1)            
                kernel_size1 = (kernel_size, kernel_size)    
                        
                mat_k = as_strided(mat, shape = output_shape + kernel_size1, strides = (stride*mat.strides[0],stride*mat.strides[1]) + mat.strides)
                mat_k = mat_k.reshape(-1, * kernel_size1)
                
                number_of_kernel = mat_k.shape[0]
                
                fuzzy_mat = np.zeros(number_of_kernel)
                
                for k in range(0, number_of_kernel):
                    ur1 = np.zeros((kernel_size,kernel_size))
                    ur2 = np.zeros((kernel_size,kernel_size))
                    ur3 = np.zeros((kernel_size,kernel_size))

                    u1_sum , u2_sum , u3_sum = 0, 0, 0

                    for i in range(0, kernel_size) :
                        for j in range(0, kernel_size) :
                            ur1[i][j] =  member_fun_u1(mat_k[k][i][j])
                            ur2[i][j] =  member_fun_u2(mat_k[k][i][j])
                            ur3[i][j] =  member_fun_u3(mat_k[k][i][j])

                            # equation 9
                            u1_sum = u1_sum + ur1[i][j]
                            u2_sum = u2_sum + ur2[i][j]
                            u3_sum = u3_sum + ur3[i][j]

                            #Fuzzy Algebric Sum
                            #Equation 9 Modification 29-06-2021
                            #u1_sum = (u1_sum + ur1[i][j])-(u1_sum*ur1[i][j])
                            #u2_sum = (u2_sum + ur2[i][j])-(u2_sum*ur2[i][j])
                            #u3_sum = (u3_sum + ur3[i][j])-(u3_sum*ur3[i][j])
                
                    #check max membership sum value equation 10
                    id = get_max_membership_sum_function(u1_sum, u2_sum, u3_sum)
                
                    # equation 11
                    kernel_value = 0

                    if id == 1:
                        if u1_sum != 0:
                            for i in range(0, kernel_size) :
                                for j in range(0, kernel_size) :
                                    tmp = ur1[i][j] * mat_k[k][i][j]
                                    kernel_value = kernel_value + tmp
                            kernel_value = kernel_value / u1_sum
                    elif id == 2:
                        if u2_sum != 0:
                            for i in range(0, kernel_size) :
                                for j in range(0, kernel_size) :
                                    tmp = ur2[i][j] * mat_k[k][i][j]
                                    kernel_value = kernel_value + tmp
                            kernel_value = kernel_value / u2_sum
                    else :
                        if u3_sum != 0:
                            for i in range(0, kernel_size) :
                                for j in range(0, kernel_size) :
                                    tmp = ur3[i][j] * mat_k[k][i][j]
                                    kernel_value = kernel_value + tmp
                            kernel_value = kernel_value / u3_sum
                
                    fuzzy_mat[k] = kernel_value

                fuzzy_mat = fuzzy_mat.reshape(output_shape)
                pool_ar[image_i][channel_j] = fuzzy_mat

        batch_result.append(pool_ar)

    batch_result = torch.from_numpy(np.array(batch_result)).to(device, dtype=torch.float32)
    return batch_result

# %%
class FuzzyPool2d_from_scratch(_MaxPoolNd):
    kernel_size: _size_2_t
    stride: _size_2_t

    def forward(self, input: Tensor) -> Tensor:
        return fuzzy_pooling_2d(input, self.kernel_size, self.stride)


# %%
import matplotlib.pyplot as plt

def visualize_video_frames(x, rows, cols):
    # Define the frame step
    frame_step = frame_count // (rows * cols)  # Determine the frame step to evenly distribute frames
    print("Skipped every: ", frame_step, "frames")
    
    # Generate the indices of frames to be sampled with the frame step
    sample_indices = np.arange(0, frame_count, frame_step)[:rows * cols]

    # Plotting setup
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))

    # Flatten axes to easily iterate through them
    axes = axes.flatten()

    # Loop through the 60 frames (5x12) and display them
    for i, idx in enumerate(sample_indices):
        # Extracting the frame (for batch 0, channel 0, at the idx frame)
        img = x[0, :, idx, :, :]  # Shape: [3, 256, 256]
        
        # Convert from tensor to numpy and permute channels (3, 256, 256) -> (256, 256, 3)
        img = img.permute(1, 2, 0).numpy()

        # Normalize to 0-1 range for display
        img = (img - img.min()) / (img.max() - img.min())

        # Plot the image
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Frame {idx}")

    # Adjust layout
    plt.tight_layout()
    plt.show()


# %%
from micro_model import fuzzy_pooling_2d_vectorized
test_tensor = next(iter(train_loader))[0]

visualize_video_frames(test_tensor[0], 3, 3)
visualize_video_frames(fuzzy_pooling_2d_vectorized(test_tensor[0], 3, 2), 1, 2)

# %%
from micro_model import resnet10, TSN

resnet_shortcut = 'B'
last_fc = True
torch.cuda.empty_cache()
model = resnet10(num_classes=n_classes, fuzzy=fuzzy, shortcut_type=resnet_shortcut,
                sample_size=frame_size, sample_duration=frame_count,
                last_fc=last_fc, softmax=tsn, temperature=temperature)
if tsn:
    model = TSN(model, n_classes)
model = model.to(device)
# model

# %%
# from torchinfo import summary

# summary(model, input_shape=(1, 3, frame_count, frame_size, frame_size))

# %%
torch.cuda.empty_cache()

# %%
if num_workers == 0 and False:
    # model trial testing
    # train_features, train_labels = next(iter(train_loader))
    dataloader_iterator = iter(train_loader)
    for i in range(len(train_loader)):
        start_total = time.perf_counter()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        train_features, train_labels = next(dataloader_iterator)
        train_features, train_labels = train_features.to(device), train_labels.to(device)
        print(train_features.shape)
        end = time.perf_counter()
        print(f"Load data time taken: {end-start}")

        start = time.perf_counter()
        output = model(train_features)
        end = time.perf_counter()

        print(f"Forward pass time taken: {end-start}")

        torch.cuda.synchronize()
        end_total = time.perf_counter()
        print(f"Time taken: {end_total-start_total}")

# %% [markdown]
# # TRAINING PREPARATION

# %%
import csv


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    # batch_size = targets.size(0)

    # _, pred = outputs.topk(1, 1, True)
    # pred = pred.t()
    # correct = pred.eq(targets.view(1, -1))
    # n_correct_elems = correct.float().sum().item()

    # return n_correct_elems / batch_size
    _, pred = outputs.topk(1, 1, True)
    correct = pred.view(-1).eq(targets)
    return correct.sum().item() / targets.size(0)


# %%
crnn_params = list(model.parameters())
optimizer = torch.optim.Adam(crnn_params, lr=lr_rate, weight_decay=weight_decay)
class_weights = None
if weighted:
    class_weights = torch.tensor(loss_weight).to(device)  # Higher weights for classes 0 and 2
criterion = nn.CrossEntropyLoss(weight=class_weights)


# %%
def resume_model(resume_path, model, optimizer):
    """ Resume model """
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model Restored from Epoch {}".format(checkpoint['epoch']))
    start_epoch = checkpoint['epoch'] + 1
    return model, start_epoch

# %%
if resume_path:
    model, start_epoch = resume_model(resume_path, model, optimizer)
    if continue_from is not None:
        start_epoch = continue_from
else:
    start_epoch = 1
print("Start Epoch ", start_epoch)

# %%
import time
def train_epoch(model, data_loader, criterion, optimizer, epoch, log_interval, device):
    model.train()

    train_loss = 0.0
    losses = AverageMeter()
    accuracies = AverageMeter()
    model = model.to(device)
    for batch_idx, (data, targets) in enumerate(tqdm(data_loader)):
        data, targets = data.to(device), targets.to(device)
        
        outputs = model(data)

        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        train_loss += loss.item()
        losses.update(loss.item(), data.size(0))
        accuracies.update(acc, data.size(0))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

#         if (batch_idx + 1) % log_interval == 0:
#             avg_loss = train_loss / log_interval
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, (batch_idx + 1) * len(data), len(data_loader.dataset), 100. * (batch_idx + 1) / len(data_loader), avg_loss))
#             train_loss = 0.0

    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        len(data_loader.dataset), losses.avg, accuracies.avg * 100))

    return losses.avg, accuracies.avg  

# %%
def val_epoch(model, data_loader, criterion, device):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    model = model.to(device)
    with torch.no_grad():
        for index, (data, targets) in enumerate(tqdm(data_loader)):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)  

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

    # show info
    print('Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(len(data_loader.dataset), losses.avg, accuracies.avg * 100))
    return losses.avg, accuracies.avg


# %%
save_interval =1

# %%
if resume_path:
    if not os.path.exists(os.path.join(metadata_dir, "training_data_"+frame+"_"+main_class+"_"+sub_class+".json")):
        # raise Exception("Training data json file not found for", main_class)
        print("Training data json file not found for", main_class)
        train_losses = []
        train_accs = []
        best_train_acc = 0
        best_train_loss = float('inf')

        val_losses = []
        val_accs = []
        best_val_acc = 0
        best_val_loss = float('inf')
    else:
        with open(os.path.join(metadata_dir, "training_data_"+frame+"_"+main_class+"_"+sub_class+".json")) as outfile:
            print(os.path.join(metadata_dir, "training_data_"+frame+"_"+main_class+"_"+sub_class+".json"))
            metadata = json.load(outfile)
            
            train_losses = metadata["train_losses"][:start_epoch]
            train_accs = metadata["train_accs"][:start_epoch]
            best_train_acc = metadata["best_train_acc"]
            best_train_loss = metadata["best_train_loss"]

            val_losses = metadata["val_losses"][:start_epoch]
            val_accs = metadata["val_accs"][:start_epoch]
            best_val_acc = metadata["best_val_acc"]
            best_val_loss = metadata["best_val_loss"]
else:
    train_losses = []
    train_accs = []
    best_train_acc = 0
    best_train_loss = float('inf')

    val_losses = []
    val_accs = []
    best_val_acc = 0
    best_val_loss = float('inf')

# %%
import json
def write_metadata():
    # Save training data to json
    training_data = {
        "name": frame+"_"+main_class+"_"+sub_class,
        "classes": list(classes_dict.keys()),
        "best_train_acc": best_train_acc,
        "best_val_acc": best_val_acc,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs
    }

    with open(os.path.join(metadata_dir, "training_data_"+frame+"_"+main_class+"_"+sub_class+".json"), 'w') as outfile:
        json.dump(training_data, outfile)

# early_stop_patience_train = 10
# early_stop_patience_val = 5

is_training = True
if is_training:
    for epoch in range(start_epoch, n_epochs + 1):
        print(epoch)
        train_loss, train_acc = train_epoch( model, train_loader, criterion, optimizer, epoch, log_interval, "cuda")
        val_loss, val_acc = val_epoch(model, val_loader, criterion, "cuda")
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        # if train_loss < best_train_loss and train_acc > best_train_acc:
        best_train_loss = train_loss
        best_train_acc = train_acc
        torch.save(state,  os.path.join(model_dir, "best_train_model_"+frame+"_"+main_class+"_"+sub_class+".pth"))
        print("New train model saved")
            # Reset early stop counter
            # early_stop_patience_train = 0

        if val_loss < best_val_loss and val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(state,  os.path.join(model_dir, "best_val_model_"+frame+"_"+main_class+"_"+sub_class+".pth"))
            print("Best val model saved")

            # Reset early stop counter
            # early_stop_patience_val = 0

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if epoch % 5 == 0:
            write_metadata()

        # saving weights to checkpoint
    #     if (epoch) % save_interval == 0:
    # #         summary_writer.add_scalar( 'losses/train_loss', train_loss, global_step=epoch)
    # #         summary_writer.add_scalar('losses/val_loss', val_loss, global_step=epoch)
    # #         summary_writer.add_scalar('acc/train_acc', train_acc * 100, global_step=epoch)
    # #         summary_writer.add_scalar('acc/val_acc', val_acc * 100, global_step=epoch)

    #         state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    # #         torch.save(state, os.path.join('snapshots', f'{model}-Epoch-{epoch}.pth'))
    #         direc = "model-"+str(epoch)+"-"+str(train_acc)+".pth"
    #         torch.save(state,  direc)
    #         print("Epoch {} model saved!\n".format(epoch))

# %%
model.load_state_dict(torch.load(os.path.join(model_dir, "best_val_model_"+frame+"_"+main_class+"_"+sub_class+".pth"))["state_dict"])
model.eval()
val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
print(val_acc)
