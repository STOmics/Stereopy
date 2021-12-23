import numpy as np
import os
from scipy import ndimage
from .find_maxima import find_maxima
from skimage import measure, segmentation
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')


# statistic cell property
def grade_stat(props, output_path):
    stat = []
    for idx, obj in enumerate(props):
        prop = {}
        prop['area'] = obj['area']
        prop['mean_intensity'] = obj['mean_intensity']
        prop['eccentricity'] = obj['eccentricity']
        prop['feret_diameter_max'] = obj['feret_diameter_max']
        prop['convex_area'] = obj['convex_area']
        stat.append(prop)
    f = open(os.path.join(output_path, 'cell_stat_auto.txt'), 'w')
    f.write(str(stat))
    f.close()


# calc per cell score
def score_cell(obj):
    AREA_THRE = 150
    INTENSITY_THRE = 150
    SHAPE_THRE = 0.6
    CONVEX_THRE = 0.8

    area = obj['area']
    convex_area = obj['convex_area']
    eccentricity = obj['eccentricity']
    mean_intensity = obj['mean_intensity']
    ratio = convex_area / area

    if area < AREA_THRE:
        area_score = 0.15
        intensity_score = 0
        shape_score = 0
    elif mean_intensity < INTENSITY_THRE:
        intensity_score = 0.25 * mean_intensity / 255
        area_score = 0.25
        shape_score = 0
    else:
        intensity_score = 0.1 + 0.25 * mean_intensity / 255
        area_score = 0.25
        shape_score = 0.25 * eccentricity + 0.25 * ratio

    total_score = area_score + intensity_score + shape_score
    return total_score


# score and watershed
def water_score(input_list):

    mask, image = input_list
    label = measure.label(mask, connectivity=2)
    props = measure.regionprops(label, intensity_image=image)
    shapes = mask.shape
    post_mask = np.zeros(shapes, dtype=np.uint8)
    color_mask_ori = np.zeros(shapes)
    score_list = []
    for idx, obj in enumerate(props):
        intensity_image = obj['intensity_image'] # white image
        bbox = obj['bbox']
        center = obj['centroid']
        count, xpts, ypts = find_maxima(intensity_image)
        if count > 1:
            distance = ndimage.distance_transform_edt(intensity_image)
            markers = np.zeros(intensity_image.shape, dtype=np.uint8)
            for i in range(count):
                markers[ypts[i], xpts[i]] = i + 1
            seg_result = segmentation.watershed(-distance, markers, mask=intensity_image, compactness=10,
                                                watershed_line=True)
            seg_result = np.where(seg_result != 0, 255, 0).astype(np.uint8)  #binary image
            post_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] += seg_result
            label_seg = measure.label(seg_result, connectivity=1)
            props_seg = measure.regionprops(label_seg, intensity_image=intensity_image)
            color_seg = np.zeros(obj['image'].shape)
            for p in props_seg:
                score_temp = score_cell(p)
                bbox_p = p['bbox']
                center = p['centroid']
                color_mask_temp = p['image'] * score_temp * 100
                color_seg[bbox_p[0]: bbox_p[2], bbox_p[1]: bbox_p[3]] += color_mask_temp
                score_list.append([p['label'], center[0], center[1], score_temp])
            color_mask_ori[bbox[0]: bbox[2], bbox[1]: bbox[3]] += color_seg
        else:
            post_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] += (obj['image'] * 255).astype(np.uint8)
            # area_list.append(mean_intensity)
            total_score = score_cell(obj)
            score_list.append([obj['label'], center[0], center[1], total_score])
            color_mask_temp = obj['image'] * total_score * 100
            color_mask_ori[bbox[0]: bbox[2], bbox[1]: bbox[3]] += color_mask_temp
    post_mask = np.where(post_mask != 0, 1, 0).astype(np.uint8)
    color_mask_ori = np.array(np.rint(color_mask_ori), dtype=np.uint8)
    return [post_mask, color_mask_ori]


def score(input_list):
    mask, image = input_list
    label = measure.label(mask, connectivity=2)
    props = measure.regionprops(label, intensity_image=image)
    shapes = mask.shape
    color_mask_ori = np.zeros(shapes)
    score_list = []
    for idx, obj in enumerate(props):
        bbox = obj['bbox']
        center = obj['centroid']
        total_score = score_cell(obj)
        score_list.append([obj['label'], center[0], center[1], total_score])
        color_mask_temp = obj['image'] * total_score * 100
        color_mask_ori[bbox[0]: bbox[2], bbox[1]: bbox[3]] += color_mask_temp
    color_mask_ori = np.array(np.rint(color_mask_ori), dtype=np.uint8)
    return [mask, color_mask_ori]


def watershed_multi(input_list, processes):
    with mp.Pool(processes=processes) as p:
        post_img = p.map(water_score, input_list)
    return post_img


def score_multi(input_list, processes):
    with mp.Pool(processes=processes) as p:
        post_img = p.map(score, input_list)
    return post_img