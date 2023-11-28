#!/usr/bin/env python3
# coding: utf-8
"""
@author: qindanhua@genomics.cn
@time:2021/08/04
"""
import copy
from typing import Optional

import holoviews as hv
import hvplot.pandas  # noqa
import pandas as pd
import panel as pn
import param
# from colorcet import palette
from holoviews.selection import link_selections

from stereo.log_manager import logger
from stereo.stereo_config import stereo_conf
from stereo.tools.tools import make_dirs

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
        self.drop_checkbox = pn.widgets.Select(
            name='method',
            options={'keep selected point': False, 'drop selected point': True},
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
        self.download.on_click(self._download_callback)
        self.figure = self.interact_scatter()

    def generate_selected_expr_matrix(self, selected_pos, drop=False):
        if selected_pos is not None:
            selected_index = self.scatter_df.index.drop(selected_pos) if drop else selected_pos
            data_temp = copy.deepcopy(self.data)
            self.selected_exp_data = data_temp.sub_by_index(cell_index=selected_index)
            self.selected_exp_data = self.selected_exp_data.tl.filter_genes(mean_umi_gt=0)
        else:
            self.selected_exp_data = None

    @param.depends(link.param.selection_expr)
    def _download_callback(self, _):
        self.download.loading = True
        selected_pos = hv.Dataset(self.scatter_df).select(link.selection_expr).data.index
        self.generate_selected_expr_matrix(selected_pos, self.drop_checkbox.value)
        logger.info('generate a new StereoExpData')
        self.download.loading = False

    @param.depends(link.param.selection_expr)
    def get_selected_boundary_coors(self) -> list:
        """
        get selected area exp boundary coords, list contains each x,y
        Returns:

        """
        if not self.selected_exp_data or self.selected_exp_data.shape == self.data.shape:
            raise Exception('Please select the data area in the picture first!')

        selected_pos = hv.Dataset(self.scatter_df).select(link.selection_expr).data.index
        self.generate_selected_expr_matrix(selected_pos, self.drop_checkbox.value)
        list_poly_selection_exp_coors = list()
        data_set = set()
        for label in self.selected_exp_data.position.tolist():
            x_y = ','.join([str(label[0]), str(label[1])])
            if x_y in data_set:
                continue
            data_set.add(x_y)
            list_poly_selection_exp_coors.append(label)
        return list_poly_selection_exp_coors

    def export_high_res_area(self, origin_file_path: str, output_path: str) -> str:
        """
        export selected area in high resolution
        Args:
            origin_file_path: origin file path which you read
            output_path: location the high res file storaged
        Returns:
            output_path
        """
        coors = self.get_selected_boundary_coors()
        print('coors length: %s' % len(coors))
        if not coors:
            raise Exception('Please select the data area in the picture first!')

        make_dirs(output_path)
        from gefpy.cgef_adjust_cy import CgefAdjust
        cg = CgefAdjust()
        if self.data.bin_type == 'cell_bins':
            cg.generate_cgef_by_coordinate(origin_file_path, output_path, coors)
        else:
            cg.generate_bgef_by_coordinate(origin_file_path, output_path, coors, self.data.bin_size)

        return output_path

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
                        pn.Row(self.drop_checkbox),
                        'export selected data a new StereoExpData object',
                        self.download,
                    ),
                ))
        )
        return self.figure

    def show(self, inline=True):
        if inline:
            return self.figure
        else:
            self.figure.show()
