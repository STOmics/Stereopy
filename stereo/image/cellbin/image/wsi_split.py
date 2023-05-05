from math import ceil

from tqdm import tqdm
import numpy as np


class SplitWSI(object):
    def __init__(self, img, win_shape, overlap=0, batch_size=0,
                 need_fun_ret=False, need_combine_ret=True, editable=False, tar_dtype=np.uint8):
        """
        update by cenweixuan on 2023/3/07
        help split the img and run the function piece by piece then combine the pieces into img
        :param img:(ndarry)
        :param win_shape:(tuple)pieces shape
        :param overlap:(int)
        :param need_fun_ret: fun's batch return
        :param need_combine_ret: if need combine ret
        :param editable:True to overwrite the img with dst
        :param batch_size:>0 your fun must support to input a list
        :param tar_dtype:output dtype
        """
        self._img = img
        self._win_shape = win_shape
        self._overlap = overlap
        self._editable = editable
        self._batch_size = batch_size
        self._tar_dtype = tar_dtype

        self._need_fun_ret = need_fun_ret
        self._need_combine_ret = need_combine_ret

        self._box_lst = []
        self._dst = np.array([])
        self._fun_ret = []

        self._y_nums = 0
        self._x_nums = 0

        self._runfun = None
        self._runfun_args = None
        self._is_set_runfun = 0

        self._prefun = None
        self._prefun_args = None
        self._is_set_prefun = 0

        self._fusion = None
        self._fusion_args = None
        self._is_set_fusion_fun = 0

        self._f_init()

    def _f_init(self):
        if self._need_combine_ret:
            if self._editable:
                self._dst = self._img
            else:
                self._dst = np.zeros(self._img.shape, self._tar_dtype)

    def get_nums(self):
        return self._x_nums, self._y_nums

    def f_set_run_fun(self, fun, *args):
        self._runfun = fun
        self._runfun_args = args
        self._is_set_runfun = 1

    def f_set_pre_fun(self, fun, *args):
        self._prefun = fun
        self._prefun_args = args
        self._is_set_prefun = 1

    def f_set_fusion_fun(self, fun, *args):
        self._fusion = fun
        self._fusion_args = args
        self._is_set_fusion_fun = 1

    def _f_split(self):
        h, w = self._img.shape[:2]
        win_h, win_w = self._win_shape[:2]
        self._y_nums = ceil(h / (win_h - self._overlap))
        self._x_nums = ceil(w / (win_w - self._overlap))
        for y_temp in range(self._y_nums):
            for x_temp in range(self._x_nums):
                x_begin = int(max(0, x_temp * (win_w - self._overlap)))
                y_begin = int(max(0, y_temp * (win_h - self._overlap)))
                x_end = int(min(x_begin + win_w, w))
                y_end = int(min(y_begin + win_h, h))
                if y_begin >= y_end or x_begin >= x_end:
                    continue
                self._box_lst.append([y_begin, y_end, x_begin, x_end])
        return

    def _f_get_batch_input(self, batch_box):
        batch_input = []
        for box in batch_box:
            y_begin, y_end, x_begin, x_end = box
            img_win = self._img[y_begin: y_end, x_begin: x_end]
            if self._is_set_prefun:
                img_win = self._prefun(img_win, *self._prefun_args)
            batch_input.append(img_win)
        return batch_input

    def _f_set_img(self, box, img_win):
        h, w = self._dst.shape[:2]
        win_h, win_w = img_win.shape[:2]
        win_y_begin, win_x_begin = 0, 0
        y_begin, y_end, x_begin, x_end = box
        if self._overlap != 0:
            if y_begin != 0:
                y_begin = min(y_begin + self._overlap // 2, h - 1)
                win_y_begin = min(win_y_begin + self._overlap // 2, win_h - 1)
            if x_begin != 0:
                x_begin = min(x_begin + self._overlap // 2, w - 1)
                win_x_begin = min(win_x_begin + self._overlap // 2, win_w - 1)
            if y_end != h:
                y_end = y_end - self._overlap // 2
            if x_end != w:
                x_end = x_end - self._overlap // 2
        if self._is_set_fusion_fun:
            self._dst[y_begin: y_end, x_begin: x_end, ...] = self._fusion(
                self._dst[y_begin: y_end, x_begin: x_end, ...],
                img_win[win_y_begin: win_y_begin + y_end - y_begin, win_x_begin: win_x_begin + x_end - x_begin, ...],
                *self._fusion_args)
        else:
            self._dst[y_begin: y_end, x_begin: x_end, ...] = \
                img_win[win_y_begin: win_y_begin + y_end - y_begin, win_x_begin: win_x_begin + x_end - x_begin, ...]
        return

    def _f_run(self):
        #for i in range(0, len(self._box_lst), self._batch_size):
        for i in tqdm(range(0, len(self._box_lst), self._batch_size)):
            batch_box = self._box_lst[i:min(i + self._batch_size, len(self._box_lst))]
            batch_input = self._f_get_batch_input(batch_box)
            batch_output = []
            if self._batch_size > 1:
                batch_output = self._runfun(batch_input, *self._runfun_args)
            else:
                batch_output = [self._runfun(batch_input[0], *self._runfun_args)]

            if self._need_fun_ret:
                self._fun_ret.append(batch_output)

            if self._need_combine_ret:
                for box, pred in zip(batch_box, batch_output):
                    self._f_set_img(box, pred)
        return

    def f_split2run(self):
        self._f_split()
        if self._is_set_runfun and (self._runfun is not None):
            self._f_run()
        return self._box_lst, self._fun_ret, self._dst
