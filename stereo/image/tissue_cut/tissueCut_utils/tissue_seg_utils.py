import torch
import numpy as np
from PIL import Image
import cv2



class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        return dict(im=im, lb=lb)


def dice(y_true, y_pred):
    # 这里的acc就是dice系数，因为保存最好模型的地方只能识别val_acc
    y_pred = (y_pred > 0).astype(np.uint8)
    y_true = (y_true > 0).astype(np.uint8)
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def hole_fill(binary_image):
    ''' 孔洞填充 '''
    hole = binary_image.copy()  ## 空洞填充
    hole = cv2.copyMakeBorder(hole, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0])  # 首先将图像边缘进行扩充，防止空洞填充不完全
    hole2 = hole.copy()
    cv2.floodFill(hole, None, (0, 0), 255)  # 找到洞孔
    hole = cv2.bitwise_not(hole)
    binary_hole = cv2.bitwise_or(hole2, hole)[1:-1, 1:-1]
    return binary_hole


def down_sample(img, shape):

    ori_image = Image.fromarray(img)
    image_thumb = ori_image.resize((shape[1], shape[0]), Image.NEAREST)
    image_thumb = np.array(image_thumb)
    return image_thumb


def up_sample(image, ori_shape):

    mask_thumb = Image.fromarray(image)
    marker = mask_thumb.resize((ori_shape[1], ori_shape[0]), Image.NEAREST)
    marker = np.array(marker).astype(np.uint8)
    return marker


def transfer_16bit_to_8bit(image_16bit):

    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)

    return image_8bit


def contrast_adjust(img):
    standard = 0 - (int(np.mean(img)) + int(np.std(img) * 2))
    image_adjust = (1 - standard / 255) * img + standard
    image_adjust[image_adjust < 0] = 0
    return image_adjust.astype(np.uint8)


def contrast_adjust_by_max(img):
    min_img = np.min(img)
    max_img = np.max(img) - 150
    mean = np.mean(img)
    std = np.std(img)
    if mean < 10 and std < 10:
        
        image_c = np.rint(255 * ((img - min_img) / (max_img - min_img)))
    image_c[image_c > 255] = 255
    return image_c.astype(np.uint8)


def contrast_adjust_by_max(img):
    min_img = np.min(img)
    max_img = np.max(img) - 150
    mean = np.mean(img)
    std = np.std(img)
    if mean < 10 and std < 10:
        
        image_c = np.rint(255 * ((img - min_img) / (max_img - min_img)))
    image_c[image_c > 255] = 255
    return image_c.astype(np.uint8)
