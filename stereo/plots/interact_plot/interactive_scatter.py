#!/usr/bin/env python3
# coding: utf-8
"""
@author: qindanhua@genomics.cn
@time:2021/08/04
"""
import copy
from typing import Optional
import re

import holoviews as hv
import hvplot.pandas  # noqa
import pandas as pd
import panel as pn
import param
# from colorcet import palette
from holoviews.selection import link_selections
import tifffile as tiff
import numpy as np

from stereo.log_manager import logger
from stereo.stereo_config import stereo_conf
from stereo.tools.tools import make_dirs
from stereo.preprocess.filter import filter_genes

link = link_selections.instance()
pn.param.ParamMethod.loading_indicator = True

colormaps = stereo_conf.linear_colormaps


class InteractiveScatter:
    """
    Interactive scatter
    """

    def __init__(
            self,
            data,
            width: Optional[int] = 500,
            height: Optional[int] = 500,
            bgcolor='#2F2F4F',
    ):
        self.data = data
        self.width = width
        self.height = height
        self.bgcolor = bgcolor
        if self.data.cells.total_counts is None:
            total_counts = self.data.exp_matrix.sum(axis=1).T.A[
                0] if self.data.issparse() else self.data.exp_matrix.sum(axis=1).T
        else:
            total_counts = self.data.cells.total_counts
        self.scatter_df = pd.DataFrame({
            'x': self.data.position[:, 0],
            'y': self.data.position[:, 1] * -1,
            'count': total_counts
        })
        self.scatter_df.reset_index(inplace=True)
        self.selected_exp_data = None
        self.selected_areas_vertex = None
        self.drop_checkbox = pn.widgets.Select(
            # name='method',
            options={'keep selected areas': False, 'drop selected areas': True},
            width=150
        )
        self.bin_select = pn.widgets.Select(
            name='bin size',
            options=[1, 10, 20],
            width=100,
        )
        self.download = pn.widgets.Button(
            name='export',
            button_type="primary",
            width=100
        )
        self.download_message = pn.widgets.StaticText(width=300, height=20)
        self.download.on_click(self._download_callback)
        self.figure = self.interact_scatter()

    def generate_selected_expr_matrix(self, selected_pos, drop=False):
        if selected_pos is not None:
            selected_index = self.scatter_df.index.drop(selected_pos) if drop else selected_pos
            data_temp = copy.deepcopy(self.data)
            self.selected_exp_data = data_temp.sub_by_index(cell_index=selected_index)
            # self.selected_exp_data = self.selected_exp_data.tl.filter_genes(mean_umi_gt=0)
            filter_genes(self.selected_exp_data, mean_umi_gt=0, inplace=True)
        else:
            self.selected_exp_data = None
    
    def get_selected_areas_vertex(self, selection_expr):
        selection_expr_list = str(selection_expr).split('|')
        self.selected_areas_vertex = []
        for sel in selection_expr_list:
            matches = re.findall(r'-?\d+\.\d+|-?\d+', sel)
            if matches is None:
                self.download_message.value = '<font color="red"><b>No area is selected.</b></font>'
                continue
            xmin, xmax, ymin, ymax = [float(m) for m in matches]
            ymin, ymax = -ymax, -ymin
            xmin = int(xmin) - 1
            xmax = int(xmax) + 1
            ymin = int(ymin) - 1
            ymax = int(ymax) + 1
            self.selected_areas_vertex.append((xmin, xmax, ymin, ymax))

    @param.depends(link.param.selection_expr)
    def _download_callback(self, _):
        self.download_message.value = ''
        self.download.loading = True
        selected_pos = hv.Dataset(self.scatter_df).select(link.selection_expr).data.index
        self.generate_selected_expr_matrix(selected_pos, self.drop_checkbox.value)
        self.get_selected_areas_vertex(link.selection_expr)
        # logger.info('generate a new StereoExpData')
        self.download_message.value = '<font color="red"><b>The selected areas have been exported.</b></font>'
        self.download.loading = False

    # @param.depends(link.param.selection_expr)
    # def get_selected_boundary_coors(self) -> list:
    @param.depends(link.param.selection_expr)
    def get_selected_area_coors(self) -> list:
        """
        get selected area exp boundary coords, list contains each x,y
        Returns:

        """

        selected_pos = hv.Dataset(self.scatter_df).select(link.selection_expr).data.index
        self.generate_selected_expr_matrix(selected_pos, self.drop_checkbox.value)
        if not self.selected_exp_data or self.selected_exp_data.shape == self.data.shape:
            raise Exception('Please select the data area first!')
        return self.selected_exp_data.position.tolist()
        # list_poly_selection_exp_coors = list()
        # data_set = set()
        # for label in self.selected_exp_data.position.tolist():
        #     x_y = ','.join([str(label[0]), str(label[1])])
        #     if x_y in data_set:
        #         continue
        #     data_set.add(x_y)
        #     list_poly_selection_exp_coors.append(label)
        # return list_poly_selection_exp_coors

    def export_high_res_area(self, origin_file_path: str, output_path: str) -> str:
        """
        export selected area in high resolution
        Args:
            origin_file_path: origin file path which you read
            output_path: location the high res file storaged
        Returns:
            output_path
        """
        # coors = self.get_selected_area_coors()
        if self.selected_exp_data is None:
            raise Exception("No data has been selected, please click the 'export' button beforehand.")
        coors = self.selected_exp_data.position.tolist()
        print('coors length: %s' % len(coors))
        if not coors:
            raise Exception('Please select the data area first!')

        make_dirs(output_path)
        from gefpy.cgef_adjust_cy import CgefAdjust
        cg = CgefAdjust()
        if self.data.bin_type == 'cell_bins':
            cg.generate_cgef_by_coordinate(origin_file_path, output_path, coors)
        else:
            cg.generate_bgef_by_coordinate(origin_file_path, output_path, coors, self.data.bin_size)

        return output_path

    def export_roi_image(self, origin_file_path: str, output_path: str):
        if self.selected_areas_vertex is None or len(self.selected_areas_vertex) == 0:
            raise Exception("No data has been selected, please click the 'export' button beforehand.")
        drop = self.drop_checkbox.value
        with tiff.TiffFile(origin_file_path) as tif:
            origin_image_data = tif.asarray()
            vmin, vmax = np.min(origin_image_data), np.max(origin_image_data)
            if self.data.bin_type == 'bins' or not drop:
                mask = np.zeros_like(origin_image_data)
            else:
                mask = np.ones_like(origin_image_data)
            if self.data.bin_type == 'bins':
                coors = self.selected_exp_data.position.astype(np.int64)
                for x, y in coors:
                    x_end, y_end = x + self.data.bin_size, y + self.data.bin_size
                    mask[y:y_end, x:x_end] = 1
                new_image_data = origin_image_data * mask
            else:
                for xmin, xmax, ymin, ymax in self.selected_areas_vertex:
                    xmin = max(0, xmin)
                    xmax = min(xmax + 1, origin_image_data.shape[1])
                    ymin = max(0, ymin)
                    ymax = min(ymax + 1, origin_image_data.shape[0])
                    mask[ymin:ymax, xmin:xmax] = 1 if not drop else 0
                new_image_data = origin_image_data * mask
            shaped_metadata = tif.shaped_metadata
            metadata = None
            if shaped_metadata is not None:
                metadata = shaped_metadata[0]
                if 'value_range' not in metadata:
                    metadata['value_range'] = (int(vmin), int(vmax))
            tiff.imwrite(output_path, new_image_data, bigtiff=True, metadata=metadata)

    def interact_scatter(self):
        pn.extension()
        hv.extension('bokeh')
        cmap = pn.widgets.Select(value=colormaps['stereo'], options=colormaps, name='color theme', width=200)
        reverse_colormap = pn.widgets.Checkbox(name='reverse_colormap')
        scatter_df = self.scatter_df
        bgcolor = self.bgcolor
        width, height = self.width, self.height

        @pn.depends(cmap, reverse_colormap)
        def _df_plot(cmap_value, reverse_cm_value):
            cmap_value = cmap_value if not reverse_cm_value else cmap_value[::-1]
            return link(scatter_df.hvplot.scatter(
                x='x', y='y', c='count', cnorm='eq_hist',
                cmap=cmap_value,
                width=width, height=height,
                padding=(0.1, 0.1),
                datashade=True,
                dynspread=True,
                # flip_yaxis=True
            ).opts(
                bgcolor=bgcolor,
                xaxis=None,
                yaxis=None,
                aspect='equal',
            ), selection_mode='union')

        @param.depends(link.param.selection_expr)
        def _selection_table(_):
            return hv.element.Table(hv.Dataset(scatter_df).select(link.selection_expr)).opts(width=300, height=200)

        self.figure = pn.Column(
            pn.Row(cmap, reverse_colormap),
            pn.Row(
                _df_plot,
                pn.Column(
                    pn.Column(
                        # "above in the table is selected points, pick or drop them to generate a new StereoExpData",
                        'export selected data as a new StereoExpData object',
                        pn.Row(self.drop_checkbox, self.download),
                        self.download_message
                    ),
                ))
        )
        return self.figure

    def show(self, inline=True):
        if inline:
            return self.figure
        else:
            self.figure.show()
