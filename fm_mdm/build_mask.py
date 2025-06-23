import numpy as np
import torch
def bbox2mask(img_shape, bbox, dtype='uint8'):
    height, width = img_shape[:2]
    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 1
    return mask

def get_center_mask(image_size):
    h, w = image_size
    mask = bbox2mask(image_size, (h//4, w//4, h//2, w//2))
    return torch.from_numpy(mask).permute(2,0,1)

def build_inpaint_center(image_size, rank):
    center_mask = get_center_mask([image_size, image_size])[None,...] # [1,1,256,256]
    center_mask = center_mask.to(rank)

    def inpaint_center(img):
        # img: [-1,1]
        mask = center_mask
        # img[mask==0] = img[mask==0], img[mask==1] = 1 (white)
        return img * (1. - mask) + mask, mask

    return inpaint_center
