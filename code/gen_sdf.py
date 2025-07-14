import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

def compute_signed_distance_field(mask):
    # 计算前景到背景的欧几里得距离场
    # print(np.unique(~mask))
    foreground_dist = distance_transform_edt(mask)  # 前景到背景的距离（正值）
    
    # 计算背景到前景的欧几里得距离场
    background_dist = distance_transform_edt(~mask)  # 背景到前景的距离（正值）

    # 结合两个距离场，得到符号距离场
    sdf = foreground_dist - background_dist  # 前景为负，背景为正

    # return foreground_dist
    return background_dist
    # return sdf

def compute_sdf(img_gt):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros_like(img_gt).astype(np.float32)
    # print(normalized_sdf.dtype)
    # print(img_gt.shape)
    # exit()

    for b in range(img_gt.shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool_)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            # print(np.max(boundary), np.min(boundary))
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            # print(np.max(sdf), np.min(sdf))
            sdf[boundary==1] = 0
            # print(np.max(sdf), np.min(sdf), np.max(normalized_sdf), np.min(normalized_sdf))
            # print(b)
            # print(sdf)
            normalized_sdf[b] = sdf
            # print('*****', np.max(normalized_sdf), np.min(normalized_sdf))
            # exit()
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

def compute_sdf(img_gt):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros_like(img_gt)
    # print(img_gt.shape)
    # exit()

    
    posmask = img_gt.astype(np.bool_)
    if posmask.any():
        negmask = ~posmask
        posdis = distance(posmask)
        negdis = distance(negmask)
        boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
        # print(np.max(boundary), np.min(boundary))
        sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
        # print(np.max(sdf), np.min(sdf))
        sdf[boundary==1] = 0
        normalized_sdf= sdf
        
        # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
        # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

if __name__ == '__main__':
    # path = 'dataset/BUSI/malignant/malignant (63)_mask.png'
    # path = 'dataset/BUSI/benign/benign (56)_mask.png'
    path = 'pred/AttenUNet_BUSI/benign (56)_mask_pred.png'
    mask = cv2.imread(path, cv2.COLOR_RGB2GRAY)
    mask = cv2.resize(mask, (256, 256), interpolation=0)
    mask = mask / 255.
    sdf = compute_sdf(mask)
    print(np.max(sdf), np.min(sdf))
    
    sdf = (sdf - sdf.min()) / (sdf.max() - sdf.min() + 1e-5)
    print(np.max(sdf), np.min(sdf))

    print(mask.shape, sdf.shape)

    cv2.imwrite('mask.png', mask * 255)
    cv2.imwrite('sdf.png', sdf * 255)