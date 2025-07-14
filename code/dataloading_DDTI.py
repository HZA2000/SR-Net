import os
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
import random
import torch.nn.functional as F
# from dataloading import *
from scipy.ndimage import label
from skimage.measure import regionprops
import pandas as pd
import time
from skimage.transform import resize

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

from custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    MaskTransform, ConvertSegmentationToRegionsTransform
from pyramid_augmentations import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from copy import deepcopy
import cv2

random.seed(42)


class DataLoader2D(SlimDataLoaderBase):
    def __init__(self, image_list, mask_list, batch_size=2):
        
        data_list = []
        seg_list = []
        
        

        self.batch_size = batch_size

        for img_path, mask_path in zip(image_list, mask_list):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.COLOR_RGB2GRAY)
           

            img = cv2.resize(img, dsize=(256, 256), interpolation=3)
            seg = cv2.resize(mask, dsize=(256, 256), interpolation=0)
            img = np.transpose(img, [2, 0, 1])

            if len(seg.shape) > 2:
                seg = seg[..., -1]
            print(seg.shape, mask_path)
            img = img / 255.
            seg = seg / 255.

            # print(img.shape, seg.shape, np.unique(seg), np.max(img), np.min(img))
            # exit()
            
            data_list.append(img)
            seg_list.append(seg)
            

                    
            # if len(data_list) > 20:
            #     break
        
        # print(infor_list)
        # data_arr = np.stack(data_list, axis=0)
        data_arr = np.stack(data_list, axis=0)
        seg_arr = np.stack(seg_list, axis=0)

        # data_arr = np.expand_dims(data_arr, axis=1)
        seg_arr = np.expand_dims(seg_arr, axis=1)
        
        # infor_arr = np.array(infor_list)

        self.data_arr = data_arr
        self.seg_arr = seg_arr
        
    def generate_train_batch(self):
        selected_nums = np.random.randint(0, self.data_arr.shape[0], self.batch_size)
        data_arr = self.data_arr[selected_nums]
        seg_arr = self.seg_arr[selected_nums]
        
        

        # infor_list = self.infor_list[selected_nums]

        # print(data_arr.shape, seg_arr.shape, label_arr.shape)
        return {'data': data_arr, 'seg': seg_arr}

default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,

    "do_elastic": True,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2,

    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis": 1,
    "p_scale": 0.2,

    "do_rotation": True,
    "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,

    "random_crop": False,
    "random_crop_dist_to_border": None,

    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    "do_mirror": True,
    "mirror_axes": (0, 1, 2),

    "dummy_2D": False,
    "mask_was_used_for_normalization": None,
    "border_mode_data": "constant",

    "all_segmentation_labels": None,  # used for cascade
    "move_last_seg_chanel_to_data": False,  # used for cascade
    "cascade_do_cascade_augmentations": False,  # used for cascade
    "cascade_random_binary_transform_p": 0.4,
    "cascade_random_binary_transform_p_per_label": 1,
    "cascade_random_binary_transform_size": (1, 8),
    "cascade_remove_conn_comp_p": 0.2,
    "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
    "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,

    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,

    "num_threads": 12 if 'nnUNet_n_proc_DA' not in os.environ else int(os.environ['nnUNet_n_proc_DA']),
    "num_cached_per_thread": 1,
}

default_2D_augmentation_params = deepcopy(default_3D_augmentation_params)

default_2D_augmentation_params["elastic_deform_alpha"] = (0., 200.)
default_2D_augmentation_params["elastic_deform_sigma"] = (9., 13.)
default_2D_augmentation_params["rotation_x"] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
default_2D_augmentation_params["rotation_y"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
default_2D_augmentation_params["rotation_z"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)

# sometimes you have 3d data and a 3d net but cannot augment them properly in 3d due to anisotropy (which is currently
# not supported in batchgenerators). In that case you can 'cheat' and transfer your 3d data into 2d data and
# transform them back after augmentation
default_2D_augmentation_params["dummy_2D"] = False
default_2D_augmentation_params["mirror_axes"] = (0, 1)  # this can be (0, 1, 2) if dummy_2D=True



def get_default_augmentation(dataloader_train, dataloader_val, patch_size, params=default_2D_augmentation_params,
                             border_val_seg=-1, pin_memory=True,
                             seeds_train=None, seeds_val=None, regions=None):
    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"
    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size

    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
        alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant",
        border_cval_seg=border_val_seg,
        order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))

    if params.get("do_mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
        if params.get("cascade_do_cascade_augmentations") and not None and params.get(
                "cascade_do_cascade_augmentations"):
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                p_per_sample=params.get("cascade_random_binary_transform_p"),
                key="data",
                strel_size=params.get("cascade_random_binary_transform_size")))
            tr_transforms.append(RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                key="data",
                p_per_sample=params.get("cascade_remove_conn_comp_p"),
                fill_with_other_class_p=params.get("cascade_remove_conn_comp_max_size_percent_threshold"),
                dont_do_if_covers_more_than_X_percent=params.get("cascade_remove_conn_comp_fill_with_other_class_p")))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))

    tr_transforms = Compose(tr_transforms)
    # from batchgenerators.dataloading import SingleThreadedAugmenter
    # batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)
    # import IPython;IPython.embed()

    batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                  params.get("num_cached_per_thread"), seeds=seeds_train,
                                                  pin_memory=pin_memory)

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    # batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)
    batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads') // 2, 1),
                                                params.get("num_cached_per_thread"), seeds=seeds_val,
                                                pin_memory=pin_memory)
    return batchgenerator_train, batchgenerator_val


if __name__ == '__main__':
    dataloader_train = DataLoader3D()
    # from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
    
    batchgenerator_train, _ = get_default_augmentation(dataloader_train, dataloader_train, None)
    # tr_transforms = []
    # params=default_3D_augmentation_params
    # # params=default_2D_augmentation_params
    # tr_transforms.append(SpatialTransform(
    #     None, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
    #     alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
    #     do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
    #     angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
    #     border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant",
    #     border_cval_seg=-1,
    #     order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
    #     p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
    #     independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    # ))
    # tr_transforms = Compose(tr_transforms)
    # # batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)
    # batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
    #                                               params.get("num_cached_per_thread"), seeds=None,
    #                                               pin_memory=True)
    # for i in range(10):
    #     _ = batchgenerator_train.next()
    # for i in tr:
    t0 = time.perf_counter()
    count = 0
    for i in batchgenerator_train:
        image, mask, label, infor = i['data'], i['target'], i['label'], i['infor']
        t1 = time.perf_counter()
        count += 1
        print(torch.unique(mask), torch.max(image), torch.min(image))
        print(image.shape, mask.shape, label.shape, (t1 - t0) / count)

        image = image.numpy()[0][1]
        mask = mask.numpy()[0][1]

        print(image.shape, mask.shape, np.squeeze(image).shape)
        # print(infor, np.unique(mask))
        direction, spacing, ori = infor[0]

        arr2itk_write(arr=np.squeeze(image), ori=ori, spacing=spacing, direction=direction, filepath='aug_image.nii.gz')
        arr2itk_write(arr=np.squeeze(mask), ori=ori, spacing=spacing, direction=direction, filepath='aug_mask.nii.gz')
        # print(image.shape, mask.shape, label.shape, infor)
        exit()