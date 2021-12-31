#######################
# intensity seg
# network infer
#######################
import numpy as np
import os
import tifffile
import glog
import torch
import tissueCut_utils.tissue_seg_utils as util
import cv2
import tissueCut_utils.tissue_seg_net as tissue_net
from skimage import measure, exposure
from tissueCut_utils.tissue_seg_utils import ToTensor
import time

torch.set_grad_enabled(False)
np.random.seed(123)


class tissueCut(object):
    def __init__(self, path, out_path, type, deep, model_path, backbone_path):

        self.is_gpu = False
        self.path = path
        self.type = type  # image type
        self.deep = deep  # segmentation method
        self.out_path = out_path
        self.model_path = model_path
        self.backbone_path = backbone_path
        glog.info('image type: %s' % ('ssdna' if type else 'RNA'))
        glog.info('using method: %s' % ('deep learning' if deep else 'intensity segmentation'))
        # init property
        self.img = []
        self.shape = []
        self.img_thumb = []
        self.mask_thumb = []
        self.mask = []
        self.file = []
        self.file_name = []
        self.file_ext = []

        self.is_gpu = torch.cuda.is_available()
        self._preprocess_file(path)

    # parse file name
    def _preprocess_file(self, path):

        if os.path.isdir(path):
            self.path = path
            file_list = os.listdir(path)
            self.file = file_list
            self.file_name = [os.path.splitext(f)[0] for f in file_list]
            self.file_ext = [os.path.splitext(f)[1] for f in file_list]
        else:
            self.path = os.path.split(path)[0]
            self.file = [os.path.split(path)[-1]]
            self.file_name = [os.path.splitext(self.file[0])[0]]
            self.file_ext = [os.path.splitext(self.file[0])[-1]]

    # RNA image bin
    def _bin(self, img):

        if img.dtype == 'uint8':
            print(img.dtype)
            bin_size = 20
        else:
            print('16', img.dtype)
            bin_size = 200

        kernel = np.zeros((bin_size, bin_size), dtype=np.uint8)
        kernel += 1
        img_bin = cv2.filter2D(img, -1, kernel)

        return img_bin

    # def save_tissue_mask(self):
    #
    #     # for idx, tissue_thumb in enumerate(self.mask_thumb):
    #     #     tifffile.imsave(os.path.join(self.out_path, self.file_name[idx] + r'_tissue_cut_thumb.tif'), tissue_thumb)
    #
    #     for idx, tissue in enumerate(self.mask):
    #         tifffile.imsave(os.path.join(self.out_path, self.file_name[idx] + r'_tissue_cut.tif'),
    #                         (tissue > 0).astype(np.uint8))
    #     glog.info('seg results saved in %s' % self.out_path)

    # 新函数
    def save_tissue_mask(self):

        # for idx, tissue_thumb in enumerate(self.mask_thumb):
        #     tifffile.imsave(os.path.join(self.out_path, self.file_name[idx] + r'_tissue_cut_thumb.tif'), tissue_thumb)
        for idx, tissue in enumerate(self.mask):
            if np.sum(tissue) == 0:
                h, w = tissue.shape[:2]
                tissue = np.ones((h, w), dtype=np.uint8)
                tifffile.imsave(os.path.join(self.out_path, self.file_name[idx] + r'_tissue_cut.tif'), tissue)
            else:
                tifffile.imsave(os.path.join(self.out_path, self.file_name[idx] + r'_tissue_cut.tif'),
                                (tissue > 0).astype(np.uint8))
        glog.info('seg results saved in %s' % self.out_path)

    # preprocess image for deep learning

    def get_thumb_img(self):

        glog.info('image loading and preprocessing...')

        for ext, file in zip(self.file_ext, self.file):
            assert ext in ['.tif', '.tiff', '.png', '.jpg']
            if ext == '.tif' or ext == '.tiff':

                img = tifffile.imread(os.path.join(self.path, file))
                if len(img.shape) == 3:
                    img = img[:, :, 0]
            else:

                img = cv2.imread(os.path.join(self.path, file), 0)

            self.img.append(img)

            self.shape.append(img.shape)

            if self.deep:

                if self.type:
                    """ssdna: equalizeHist"""

                    if img.dtype != 'uint8':
                        img = util.transfer_16bit_to_8bit(img)

                    if np.mean(img) > 50 and np.mean(img) > np.std(img) * 0.8:
                        img = util.contrast_adjust(img)
                        print(self.file_name)

                    img_pre = img
                    # img_pre = cv2.equalizeHist(img)

                else:
                    """rna: bin """

                    img = self._bin(img)
                    if img.dtype != 'uint8':
                        img = util.transfer_16bit_to_8bit(img)

                    img_pre = exposure.adjust_log(img)
                # img_pre = img
                # tifffile.imsave(os.path.join(self.out_path, file + '_contract.tif'), img_pre.astype(np.uint8))
                img_thumb = util.down_sample(img_pre, shape=(1024, 2048))
                # tifffile.imsave(os.path.join(self.out_path, file + '_contract_deep.tif'), img_thumb.astype(np.uint8))
                self.img_thumb.append(img_thumb)

    # infer tissue mask by network
    def tissue_infer_deep(self):
        # network infer

        self.get_thumb_img()

        # define tissueCut_model
        net = tissue_net.TissueSeg(2, self.backbone_path)

        # if self.type:
        #     model_path = os.path.join(os.path.split(__file__)[0], '../tissueCut_model/ssdna_seg.pth')
        # else:
        #     model_path = os.path.join(os.path.split(__file__)[0], '../tissueCut_model/rna_seg.pth')

        net.load_state_dict(torch.load(self.model_path, map_location='cpu'), strict=False)
        # net.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage), strict=False)
        net.eval()
        if self.is_gpu:
            net.cuda()

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(device)
        # net.to(device)

        # prepare data
        to_tensor = ToTensor(
            mean=(0.3257, 0.3690, 0.3223),
            std=(0.2112, 0.2148, 0.2115),
        )
        glog.info('tissueCut_model infer...')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        for shape, im, file, img_thumb in zip(self.shape, self.img_thumb, self.file, self.img_thumb):
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            # im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)#.cuda()
            # # inference
            # out = np.array(net(im, )[0].argmax(dim=1).squeeze().detach().cpu().numpy(), dtype=np.uint8)

            if self.is_gpu:
                im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
            else:
                im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)
            out = np.array(net(im, ).squeeze().detach().cpu().numpy(), dtype=np.uint8)
            out = util.hole_fill(out).astype(np.uint8)
            img_open = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

            self.mask_thumb.append(np.uint8(img_open > 0))

        self.__get_roi()

    # tissue segmentation by intensity filter
    def tissue_seg_intensity(self):

        def getArea(elem):
            return elem.area

        self.get_thumb_img()

        glog.info('segment by intensity...')
        for idx, ori_image in enumerate(self.img):
            shapes = ori_image.shape

            # downsample ori_image
            if not self.type:
                ori_image = self._bin(ori_image)

            # else:
            #     if np.mean(ori_image) < 50 and np.mean(ori_image) > np.std(ori_image) * 0.8:
            #         ori_image = util.contrast_adjust(ori_image)

            # tifffile.imsave(os.path.join(self.out_path, self.file[idx] + '_contract.tif'), ori_image)

            image_thumb = util.down_sample(ori_image, shape=(shapes[0] // 5, shapes[1] // 5))

            if image_thumb.dtype != 'uint8':
                image_thumb = util.transfer_16bit_to_8bit(image_thumb)

            self.img_thumb.append(image_thumb)

            # binary
            ret1, mask_thumb = cv2.threshold(image_thumb, 125, 255, cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))  # 椭圆结构
            mask_thumb = cv2.morphologyEx(mask_thumb, cv2.MORPH_CLOSE, kernel, iterations=8)

            # choose tissue prop
            label_image = measure.label(mask_thumb, connectivity=2)
            props = measure.regionprops(label_image, intensity_image=mask_thumb)
            props.sort(key=getArea, reverse=True)
            areas = [p['area'] for p in props]
            if np.std(areas) * 10 < np.mean(areas):
                label_num = len(areas)
            else:
                label_num = int(np.sum(areas >= np.mean(areas)))
            result = np.zeros((image_thumb.shape)).astype(np.uint8)
            for i in range(label_num):
                prop = props[i]
                result += np.where(label_image != prop.label, 0, 1).astype(np.uint8)

            result_thumb = util.hole_fill(result)
            # result_thumb = cv2.dilate(result_thumb, kernel, iterations=10)

            self.mask_thumb.append(np.uint8(result_thumb > 0))

        self.__get_roi()

    # filter noise tissue
    def __filter_roi(self, props):

        if len(props) == 1:
            return props
        else:
            filtered_props = []
            for id, p in enumerate(props):

                black = np.sum(p['intensity_image'] == 0)
                sum = p['bbox_area']
                ratio_black = black / sum
                pixel_light_sum = np.sum(np.unique(p['intensity_image']) > 128)
                if ratio_black < 0.75 and pixel_light_sum > 20:
                    filtered_props.append(p)
            return filtered_props

    def __get_roi(self):

        """get tissue area from ssdna"""
        for idx, tissue_mask in enumerate(self.mask_thumb):

            label_image = measure.label(tissue_mask, connectivity=2)
            props = measure.regionprops(label_image, intensity_image=self.img_thumb[idx])

            # remove noise tissue mask
            filtered_props = self.__filter_roi(props)
            if len(props) != len(filtered_props):
                tissue_mask_filter = np.zeros((tissue_mask.shape), dtype=np.uint8)
                for tissue_tile in filtered_props:
                    bbox = tissue_tile['bbox']
                    tissue_mask_filter[bbox[0]: bbox[2], bbox[1]: bbox[3]] += tissue_tile['image']
                self.mask_thumb[idx] = np.uint8(tissue_mask_filter > 0)

            # if self.deep:
            #
            #     ratio = int(self.img[idx].shape[0] // self.mask_thumb[idx].shape[0] // 5)
            #
            #     if ratio == 0: ratio = 1
            #
            #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20 // ratio, 20 // ratio))
            #
            #     self.mask_thumb[idx] = cv2.dilate(self.mask_thumb[idx], kernel, iterations=10)

            self.mask.append(util.up_sample(self.mask_thumb[idx], self.img[idx].shape))

    def tissue_seg(self):

        # try:
        if self.deep:
            self.tissue_infer_deep()

        else:
            self.tissue_seg_intensity()

        self.save_tissue_mask()

        return 1

        # except:
        #     glog.info('Tissue seg throw exception!')
        #     return 0
