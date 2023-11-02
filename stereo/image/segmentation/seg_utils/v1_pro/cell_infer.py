import multiprocessing as mp
import os
import time

import cv2
import glog
import numpy as np
import torch
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from skimage import filters
from tqdm import tqdm

from stereo import logger
from .dataset import data_batch2
from .resnet_unet import EpsaResUnet
from .utils import split_preproc

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_transforms():
    list_transforms = []
    list_transforms.extend([])
    list_transforms.extend([ToTensorV2()])
    list_trfms = Compose(list_transforms)
    return list_trfms


def cellInfer(file, size, overlap=100):
    # split -> predict -> merge
    if isinstance(file, list):
        file_list = file
    else:
        file_list = [file]

    result = []

    model_path = os.path.join(os.path.split(__file__)[0], 'model')
    model_dir = os.path.join(model_path, 'best_model.pth')
    logger.info(f'CellCut_model infer path {model_dir}...')
    model = EpsaResUnet(out_channels=6)
    glog.info('Load model from: {}'.format(model_dir))
    model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage), strict=True)
    model.eval()
    logger.info('Load model ok.')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        glog.info('GPU type is {}'.format(torch.cuda.get_device_name(0)))
    glog.info(f"using device: {device}")
    model.to(device)
    for idx, image in enumerate(file_list):
        logger.info(image.shape)

        t1 = time.time()
        if torch.cuda.is_available():
            import cupy
            from utils import cuda_kernel
            from cucim.skimage.morphology import disk
            logger.info('median filter using gpu')
            image_cp = cupy.asarray(image)
            # Accelerate median using specific cuda kernel function
            median_image = cupy.empty(image.shape, dtype=cupy.uint8)
            (height, width) = image.shape
            cuda_kernel.median_filter_kernel(
                ((width + 15) // 16, (height + 15) // 16),
                (16, 16),
                (image_cp, median_image, width, height, disk(50))
            )

            median_image = np.asarray(median_image.get())
            images = cv2.subtract(image, median_image)
        else:
            logger.info('median filter using cpu')

            image_list, m_x_list, m_y_list = split_preproc(image, 1000, 100)
            images = np.zeros(image.shape, dtype=np.uint8)
            images.fill(0)
            median_filter_in_pool_parallel(image_list, images, m_x_list, m_y_list)

        t2 = time.time()
        logger.info('median filter: {}'.format(t2 - t1))

        # accelerate data loader
        overlap = 100
        dataset = data_batch2(images, 256, overlap)

        merge_label = image
        merge_label.fill(0)
        x_list, y_list, ori_size = dataset.get_list()
        test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=20)
        img_idx = 0
        for batch in tqdm(test_dataloader, ncols=80):
            img = batch
            img = img.type(torch.FloatTensor)
            img = img.to(device)

            pred_mask = model(img)
            bacth_size = len(pred_mask)
            pred_mask = torch.sigmoid(pred_mask).detach().cpu().numpy()
            pred = pred_mask[:, 0, :, :]
            pred[:] = (pred[:] < 0.55) * 255
            pred1 = pred.astype(np.uint8)
            for i in range(bacth_size):
                temp_img = pred1[i][:ori_size[i + img_idx][0], :ori_size[i + img_idx][1]]
                info = [x_list[i + img_idx], y_list[i + img_idx]]
                h, w = temp_img.shape
                if int(info[0]) == 0 or int(info[1]) == 0:
                    x_begin = int(info[0])
                    y_begin = int(info[1])
                    temp_data = temp_img[1: - 1, 1: - 1]
                    merge_label[int(x_begin): int(x_begin) + h - 2, int(y_begin): int(y_begin) + w - 2] = temp_data
                else:
                    x_begin = int(info[0]) + overlap // 2
                    y_begin = int(info[1]) + overlap // 2
                    temp_data = temp_img[overlap // 2: - overlap // 2, overlap // 2: - overlap // 2]
                    merge_label[int(x_begin): int(x_begin) + h - overlap,
                    int(y_begin): int(y_begin) + w - overlap] = temp_data  # noqa
            img_idx += 20

        result.append(merge_label)

    return result


def s_median_filter(image):
    from skimage.morphology import disk
    m_image = filters.median(image, disk(50))
    m_image = cv2.subtract(image, m_image)
    return m_image


def median_filter_in_pool(image_list, images):
    with mp.Pool(processes=20) as p:
        for i in image_list:
            median_image = p.apply_async(s_median_filter, (i,))
            images.append(median_image)
        p.close()
        p.join()


def median_filter_in_pool_parallel(image_list, images, x_list, y_list):
    import queue
    q = queue.Queue()

    def worker():
        idx = 0
        while True:
            item = q.get()
            if item == 'STOP':
                q.task_done()
                break
            item = item.get()

            x = x_list[idx]
            y = y_list[idx]
            h, w = item.shape
            images[x: x + h - 2, y: y + w - 2] = item[1:-1, 1:-1]
            idx += 1
            # del item

            q.task_done()

    import threading
    threading.Thread(target=worker, daemon=True).start()

    with mp.Pool(processes=20) as p:
        for i in image_list:
            median_image = p.apply_async(s_median_filter, (i,))
            q.put(median_image)
        p.close()
        p.join()
        q.put('STOP')

    q.join()
