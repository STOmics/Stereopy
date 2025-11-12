#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/2 09:53
# @Author  : payne
# @File    : boundary.py
# @Description : get boundary coordination when user use poly selection


import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon


class ConcaveHull(object):

    def __init__(self, points, k):
        if isinstance(points, np.core.ndarray):
            self.data_set = points
        elif isinstance(points, list):
            self.data_set = np.array(points)
        else:
            raise ValueError('Please provide an [N,2] numpy array or a list of lists.')

        # Clean up duplicates
        self.data_set = np.unique(self.data_set, axis=0)

        # Create the initial index
        self.indices = np.ones(self.data_set.shape[0], dtype=bool)  # bool of a column of all 1's

        self.k = k

    @staticmethod
    def dist_pt_to_group(a, b):  # a is a (n,2) , b is (1,2) arrays
        return np.sqrt(np.sum(np.square(np.subtract(a, b)), axis=1))

    @staticmethod
    def get_lowest_latitude_index(points):
        return np.argsort(points[:, 1])[0]

    @staticmethod
    def norm_array(v):
        """
        normalize row vectors in an array. observations are rows
        """
        return np.divide(v, np.array(np.sqrt(np.sum(np.square(v), axis=1)), ndmin=2).transpose())

    @staticmethod
    def norm(v):
        """
         normalize a single vector, is there an existing command?
        """
        norms = np.array(np.sqrt(np.sum(np.square(v))))
        return v / norms

    def get_k_nearest(self, ix, k):
        """
        Calculates the k nearest point indices to the point indexed by ix

        :param ix: Index of the starting point
        :param k: Number of neighbors to consider
        :return: Array of indices into the data set array
        """
        ixs = self.indices
        # base_indices is list of linear indicies that are TRUE, ie part of dataset
        base_indices = np.arange(len(ixs))[ixs]
        distances = self.dist_pt_to_group(self.data_set[ixs, :], self.data_set[ix, :])
        sorted_indices = np.argsort(distances)
        kk = min(k, len(sorted_indices))
        k_nearest = sorted_indices[range(kk)]
        return base_indices[k_nearest]

    def clockwise_angles(self, last, ix, ixs, first):
        """
        last needs to be the index of the previous current point
        """
        if first == 1:
            last_norm = np.array([-1, 0], ndmin=2)
        elif first == 0:
            # normalized vector pointing towards previous point
            last_norm = self.norm(np.subtract(self.data_set[last, :], self.data_set[ix, :]))
        # normalized row vectors pointing to set of k nearest neibs
        ixs_norm = self.norm_array(np.subtract(self.data_set[ixs, :], self.data_set[ix, :]))
        ang = np.zeros((ixs.shape[0], 1))
        for j in range(ixs.shape[0]):
            theta = np.arccos(np.dot(last_norm, ixs_norm[j, :]))
            z_comp = np.cross(last_norm, ixs_norm[j, :])
            if z_comp <= 0:
                ang[j, 0] = theta
            elif z_comp > 0:
                ang[j, 0] = 2 * np.pi - theta
        return np.squeeze(ang)

    def recurse_calculate(self):
        """
        Calculates the concave hull using the next value for k while reusing the distances dictionary
        :return: Concave hull
        """
        recurse = ConcaveHull(self.data_set, self.k + 1)
        if recurse.k >= self.data_set.shape[0]:
            print(" max k reached, at k={0}".format(recurse.k))
            return None
        return recurse.calculate()

    def calculate(self):
        """
        Calculates the convex hull of the data set as an array of points
        :return: Array of points (N, 2) with the concave hull of the data set
        """
        if self.data_set.shape[0] < 3:
            return None

        if self.data_set.shape[0] == 3:
            return self.data_set

        # Make sure that k neighbors can be found
        kk = min(self.k, self.data_set.shape[0])

        first_point = self.get_lowest_latitude_index(self.data_set)
        current_point = first_point
        # last_point = current_point # not sure if this is necessary since it wont get used until after step 2

        # Note that hull and test_hull are matrices (N, 2)
        hull = np.reshape(np.array(self.data_set[first_point, :]), (1, 2))
        test_hull = hull

        # Remove the first point
        self.indices[first_point] = False

        step = 2
        stop = 2 + kk

        while ((current_point != first_point) or (step == 2)) and len(
                self.indices[self.indices]) > 0:  # last condition counts number of ones, points in dataset
            if step == stop:
                self.indices[first_point] = True
            # notice how get_k_nearest doesnt take the data set directly as an arg, as it is implicit that it takes
            # self as an imput because we are inside a class:

            # knn = [3,6,2] or [0,2,7] etc indicies into the full
            # dataset (with no points removed)
            knn = self.get_k_nearest(current_point, kk)

            if step == 2:
                angles = self.clockwise_angles(1, current_point, knn, 1)
            else:
                # Calculates the headings between first_point and the knn points
                # Returns angles in the same indexing sequence as in knn
                angles = self.clockwise_angles(last_point, current_point, knn, 0)  # noqa

            # Calculate the candidate indexes (largest angles first). candidates =[0,1,2]  or [2,1,0] etc if kk=3
            candidates = np.argsort(-angles)

            i = 0
            invalid_hull = True

            while invalid_hull and i < len(candidates):
                candidate = candidates[i]

                # Create a test hull to check if there are any self-intersections
                next_point = np.reshape(self.data_set[knn[candidate], :], (1, 2))
                test_hull = np.append(hull, next_point, axis=0)

                line = LineString(test_hull)
                # invalid_hull will remain True for every candidate which creates
                # a line that intersects the hull. as soon as the hull doesnt self intersect, it will become false
                # and the loop will terminate
                invalid_hull = not line.is_simple
                i += 1

            if invalid_hull:
                return self.recurse_calculate()

            last_point = current_point  # record last point for clockwise angles # noqa
            current_point = knn[candidate]  # candidate = 0, 1, or 2 if kk=3
            hull = test_hull
            # we remove the newly found current point from the "mask" indicies so that it wont be passed to
            # get_k_nearest (within the implicit input, self)

            self.indices[current_point] = False
            step += 1

        poly = Polygon(hull)

        count = 0
        total = self.data_set.shape[0]
        for ix in range(total):
            pt = Point(self.data_set[ix, :])
            if poly.intersects(pt) or pt.within(poly):
                count += 1

        if count == total:
            return hull
        else:
            return self.recurse_calculate()
