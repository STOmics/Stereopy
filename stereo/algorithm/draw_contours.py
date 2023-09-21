import os
import random

import cv2
import numpy as np
import pandas as pd
# from tqdm import tqdm
from skimage import measure

from stereo.log_manager import logger
from stereo.utils.time_consume import log_consumed_time


class DrawContours:

    def __init__(self, adjusted_data: pd.DataFrame, out_dir="./"):
        self.data = adjusted_data.loc[:, ['x', 'y', 'label']].to_numpy()
        self.out_path = os.path.join(out_dir, 'outline.txt')

    def get_cells_centroids(self, cell_points: dict):
        centroids = {}
        ct_label = {}
        for cell, cell_point in cell_points.items():
            centroid = self.get_centroid(cell_point)
            centroids[cell] = centroid
            ct_label[tuple(centroid)] = cell
        return centroids, ct_label

    def get_centroid(self, cell_point: np.ndarray):
        if not isinstance(cell_point, np.ndarray):
            cell_point = np.array(cell_point, dtype=np.uint32)
        x, y = np.sum(cell_point, axis=0)
        centroid = [int(x / len(cell_point)), int(y / len(cell_point))]
        return centroid

    def init_cell_points(self):
        logger.info("init data...")
        cell_points = {}
        for row in self.data:
            x, y, label = row
            if label in cell_points:
                cell_points[label].append([x, y])
            else:
                cell_points[label] = [[x, y]]
        return cell_points

    def create_canvas(self, cell_points: dict):
        logger.info("create canvas...")
        # for label, cell_point in tqdm(cell_points.items(), desc="fill poly on canvas"):
        max_x, max_y, max_label = self.data.max(axis=0)
        canvas = np.zeros([max_y + 1, max_x + 1], dtype=np.uint8)
        centroids = {}
        ct_label = {}
        for label, cell_point in cell_points.items():
            cell_point = np.array(cell_point, dtype=np.int32)
            hull = cv2.convexHull(cell_point)
            hull = np.asarray([i[0].tolist() for i in hull])
            cv2.fillPoly(canvas, [hull], (int(255)))

            centroid = self.get_centroid(cell_point)
            centroids[label] = centroid
            ct_label[tuple(centroid)] = label

        _, thres = cv2.threshold(canvas, 120, 255, cv2.THRESH_BINARY)
        h, w = canvas.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        im_floodfill = thres.copy()
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        canvas = thres | im_floodfill_inv
        return canvas, centroids, ct_label

    def random_color(self):
        b = random.randint(0, 256)
        g = random.randint(0, 256)
        r = random.randint(0, 256)
        return (r, g, b)

    @log_consumed_time
    def get_contours(self):
        cell_points = self.init_cell_points()

        img, centroids, ct_label = self.create_canvas(cell_points)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        ret, sure_fg = cv2.threshold(dist_transform, dist_transform.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        ct_np = np.array(list(ct_label.keys()))
        i = 3
        for label, xy in centroids.items():
            markers[xy[1], xy[0]] = i + label

        markers = cv2.watershed(img, markers)

        img[markers == -1] = [0, 0, 0]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        kernel = np.ones((1, 1), np.uint8)
        img_erode = cv2.erode(img, kernel)
        label = measure.label(img_erode, connectivity=1)
        props = measure.regionprops(label)

        out = []
        logger.info("start to draw contours of cells")
        for i, p in enumerate(props):
            cell = p['image'].astype(np.uint8)

            if cell.shape == (1, 1):
                continue

            bbox_p = p['bbox']
            center = ct_np[np.where((ct_np[:, 0] > bbox_p[1]) & (ct_np[:, 0] < bbox_p[3]))]
            center = center[np.where((center[:, 1] > bbox_p[0]) & (center[:, 1] < bbox_p[2]))]
            if len(center) == 0:
                continue

            elif len(center) == 1:
                center_pt = tuple(center[0].tolist())
            else:
                mid_pt = np.asarray([(bbox_p[1] + bbox_p[3]) // 2, (bbox_p[0] + bbox_p[2]) // 2])
                dis = np.sqrt(np.sum((center - mid_pt) ** 2, axis=1))
                center_pt = tuple(center[np.argmin(dis)].tolist())

            label = int(ct_label[center_pt])

            contours, _ = cv2.findContours(cell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(np.array(contours).tolist()[0]) > 32:
                i = 128 * len(np.array(contours).tolist()[0])
                while True:
                    epsilon = cv2.arcLength(contours[0], True) / i
                    appro = cv2.approxPolyDP(contours[0], epsilon, True)  #
                    contours = [appro]
                    i -= 64
                    if len(np.array(contours).tolist()[0]) <= 32:
                        break

            ctr = np.vstack(contours).squeeze().tolist()
            out.append([label] + [[x + bbox_p[1], y + bbox_p[0]] for x, y in ctr])

        return self.save_result(out)

    def save_result(self, out_data):
        logger.info(f"save the result of drawing contours to {self.out_path}")
        lines = []
        with open(self.out_path, 'w') as f:
            # for row in tqdm(out_data, desc='writing outline'):
            for row in out_data:
                lines.append(str(row[0]) + '\t' + '\t'.join([str(x) + ' ' + str(y) for x, y in row[1:]]) + "\n")
                if len(lines) >= 10000:
                    f.writelines(lines)
                    lines.clear()
            if len(lines) > 0:
                f.writelines(lines)
        return self.out_path
