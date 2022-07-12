import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import copy
import math
import traceback

import cv2
import numpy as np
from skimage import filters

from . import tissue_seg_bcdu_model as M
from . import tissue_seg_bcdu_uity as uity

class cl_bcdu(object):
    def __init__(self, model_path):
        self.is_init = False
        self.model_path = model_path
        self.model = None
        pass

    def init_model(self):
        try:
            self.model = M.BCDU_net_D3(input_size=(512, 512, 1))
            self.model.load_weights(self.model_path)
            self.is_init = True
        except:
            traceback.print_exc()
            return False
        return True

    def adj_cnt(self, cnt1, cnt2):
        x0, y0 = cnt1[0][0][:2]
        dis = sys.maxsize
        ti = 0
        for i in range(len(cnt2)):
            tx, ty = cnt2[i][0][:2]
            l = math.sqrt((x0 - tx) ** 2 + (y0 - ty) ** 2)
            if l < dis:
                ti = i
                dis = l
        a = cnt2[:ti]
        b = cnt2[ti:]
        if len(a) == 0:
            cnt2 = b
        elif len(b) == 0:
            cnt2 = a
        else:
            cnt2 = np.concatenate((b, a))

        x1, y1 = cnt1[100][0][:2]
        x2, y2 = cnt2[100][0][:2]
        x3, y3 = cnt2[::-1][100][0][:2]
        l1 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        l2 = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
        if l1 > l2:
            cnt2 = cnt2[::-1]
        return cnt1, cnt2

    def eval_point(self, img_contrast, img_mask, type=0):
        def cnt_len(cnt):
            area = cv2.arcLength(cnt, True)
            return area

        def cnt_area(cnt):
            area = cv2.contourArea(cnt)
            return area

        th = 127

        img_c = copy.deepcopy(img_contrast)
        mask = copy.deepcopy(img_mask)
        mask[mask > 0] = 255
        mask_not = cv2.bitwise_not(mask)

        score = []
        if type == 0:
            masked = cv2.bitwise_and(img_c, img_c, mask=mask)
            b = np.sum(mask > th)
            if b == 0:
                return 0
            return np.sum(masked > th) / np.sum(mask > th)

        dconts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        dconts = list(dconts)
        dconts.sort(key=cnt_area, reverse=True)
        max_dcont_area = cnt_area(dconts[0])

        for i in range(len(dconts)):
            dcont = dconts[i]
            if cnt_area(dcont) * 10 < max_dcont_area:
                continue

            morp = np.zeros(mask.shape[:2], np.uint8)
            cv2.drawContours(morp, dconts, i, 255, -1)

            if type == 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
                morp = cv2.morphologyEx(morp, cv2.MORPH_DILATE, kernel, iterations=13) \
                       - cv2.morphologyEx(morp, cv2.MORPH_DILATE, kernel, iterations=3)
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
                morp = cv2.morphologyEx(morp, cv2.MORPH_ERODE, kernel, iterations=3) \
                       - cv2.morphologyEx(morp, cv2.MORPH_ERODE, kernel, iterations=13)

            morp = cv2.bitwise_and(morp, mask_not, mask=mask_not)

            cont, hier = cv2.findContours(morp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            cont_arr = []
            for i in range(len(cont)):
                if hier[0][i][3] != -1:
                    continue
                l = cnt_len(cont[i])
                if l > 100 and hier[0][i][2] > -1:
                    tmp = []
                    for j in range(len(hier[0])):
                        if hier[0][j][3] == i:
                            tmp.append(cont[j])
                    tmp.sort(key=cnt_area, reverse=True)
                    if len(tmp[0]) > 100:
                        cont_arr.append([cont[i], tmp[0]])

            for cont in cont_arr:
                cont[0], cont[1] = self.adj_cnt(cont[0], cont[1])
                step = 10
                s1 = math.ceil(len(cont[0]) / step)
                s2 = math.ceil(len(cont[1]) / step)
                for i in range(step):
                    x1, y1 = cont[0][i * s1][0][:2]
                    x2, y2 = cont[1][i * s2][0][:2]
                    cv2.line(morp, (x1, y1), (x2, y2), 0, 5)
            cont, hier = cv2.findContours(morp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cont_black = []

            avg_area = 0
            for i in range(len(cont)):
                if hier[0][i][3] != -1:
                    cont_black.append(cont[i])
                    continue
                avg_area += cnt_area(cont[i])
            avg_area /= len(cont)

            for i in range(len(cont)):
                if cnt_area(cont[i]) < avg_area / 2 or hier[0][i][3] != -1:
                    continue
                ret = np.zeros(mask.shape[:2], np.uint8)
                cv2.drawContours(ret, cont, i, (255), -1)
                cv2.drawContours(ret, cont_black, -1, (0), -1)

                st_x, st_y, st_w, st_h = cv2.boundingRect(cont[i])
                t_mask = ret[st_y:st_y + st_h, st_x:st_x + st_w]
                t_img = img_c[st_y:st_y + st_h, st_x:st_x + st_w]
                masked = cv2.bitwise_and(t_img, t_img, mask=t_mask)
                b = np.sum(t_mask > th)
                if b == 0:
                    return 0
                s = np.sum(masked > th) / b
                score.append(s)

        if type == 1:
            return max(score)
        else:
            return min(score)

    def get_score(self, avg_score, out_score):
        if out_score == 0:
            return 100
        x = avg_score / out_score
        ymin = math.log2(2)
        ymax = math.log2(200)

        y = math.log2(x)
        if y >= ymax:
            return 100
        elif y <= ymin:
            return 0
        return round((y - ymin) / (ymax - ymin) * 100)

    def predict(self, src):
        im = np.array([])
        if not self.is_init:
            if not self.init_model():
                return False, im, []

        # src = np.squeeze(src)

        im = uity.ij_auto_contrast(src)
        if im.dtype != 'uint8':
            im = uity.ij_16_to_8(im)

        im_contrast = copy.deepcopy(im)

        im = uity.down_sample(im, (512, 512))

        im = self.model.predict(np.expand_dims(np.array([im / 255.0]), axis=3))[0]
        im = np.uint8(im * 255)
        im = np.squeeze(im)

        thea = filters.threshold_li(im)
        im[im < thea] = 0
        im[im >= thea] = 255

        # _, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        im = uity.up_sample(im, src.shape[:2])
        im[im > 0] = 1

        # in_score = self.eval_point(im_contrast, im, 2)
        avg_score = self.eval_point(im_contrast, im, 0)
        out_score = self.eval_point(im_contrast, im, 1)

        return True, im, [out_score, avg_score, self.get_score(avg_score, out_score)]
