#!/usr/bin/env python3
# coding: utf-8
"""
@author: qindanhua@genomics.cn
@time:2021/08/04
"""
import pandas as pd
import numpy as np
from colorcet import palette
from holoviews.selection import link_selections
import holoviews as hv
import hvplot.pandas
import panel as pn
import param
import io
from typing import Optional
import holoviews.operation.datashader as hd
from stereo.log_manager import logger
import copy
import matplotlib.colors as mcolors

colormaps = {n: palette[n] for n in ['rainbow', 'fire', 'bgy', 'bgyw', 'bmy', 'gray', 'kbc', 'CET_D4']}
link = link_selections.instance()
pn.param.ParamMethod.loading_indicator = True

stmap_colors = ['#0c3383', '#0a88ba', '#f2d338', '#f28f38', '#d91e1e']
nodes = [0.0, 0.25, 0.50, 0.75, 1.0]
mycmap = mcolors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, stmap_colors)))
color_list = [mcolors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
colormaps['stereo'] = color_list


class InteractiveScatter:
    """
    Interactive scatter
    """
    def __init__(
            self,
            data,
            width: Optional[int] = 680, height: Optional[int] = 500,
            bgcolor='#2F2F4F',
            # bgcolor='#333333'
    ):
        self.data = data
        self.width = width
        self.height = height
        self.bgcolor = bgcolor
        # self.link = link_selections.instance()
        self.scatter_df = pd.DataFrame({
            # 'cell': self.data.cell_names,
            'x': self.data.position[:, 0],
            'y': self.data.position[:, 1],
            # 'count': np.array(self.data.exp_matrix.sum(axis=1))[:, 0],
            'count': np.array(self.data.exp_matrix.sum(axis=1))[:, 0] if self.data.cells.total_counts is None else self.data.cells.total_counts
        })
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
            # disable=True
        )
        self.download = pn.widgets.Button(
            # filename='exp_matrix.csv',
            name='export',
            # loading=True,
            # callback=self._download_callback,
            button_type="primary",
            width=100
        )
        self.download.on_click(self._download_callback)
        self.figure = self.interact_scatter()

    def generate_selected_expr_matrix(self, selected_pos, drop=False):
        if selected_pos is not None:
            # selected_index = np.isin(self.data.cell_names, selected_pos)
            selected_index = self.scatter_df.index.drop(selected_pos) if drop else selected_pos
            data_temp = copy.deepcopy(self.data)
            self.selected_exp_data = data_temp.sub_by_index(
                cell_index=selected_index)
        else:
            self.selected_exp_data = None

    @param.depends(link.param.selection_expr)
    def _download_callback(self, _):
        self.download.loading = True
        sio = io.StringIO()
        # selected_pos = hv.Dataset(self.scatter_df).select(link.selection_expr).data[['cell']].values
        selected_pos = hv.Dataset(self.scatter_df).select(link.selection_expr).data.index

        self.generate_selected_expr_matrix(selected_pos, self.drop_checkbox.value)
        logger.info(f'generate a new StereoExpData')

        # download expression matrix
        # print('tocsv')
        # self.selected_exp_data.to_df().to_csv(sio, index=False)
        # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # sio.seek(0)
        self.download.loading = False
        # return sio

    def interact_scatter(self):
        pn.extension()
        hv.extension('bokeh')
        cmap = pn.widgets.Select(value=colormaps['stereo'], options=colormaps, name='color theme', width=200)
        # alpha = pn.widgets.FloatSlider(value=1)
        reverse_colormap = pn.widgets.Checkbox(name='reverse_colormap')
        scatter_df = self.scatter_df
        # dot_size = self.dot_size
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
                # rasterize=True,
                datashade=True,
                dynspread=True,
                # tools=['undo', 'redo'],

            ).opts(
                bgcolor=bgcolor,
                invert_yaxis=True,
                aspect='equal',
            ), selection_mode='union')

        @param.depends(link.param.selection_expr)
        def _selection_table(_):
            # print(link.selection_expr)
            return hv.element.Table(hv.Dataset(scatter_df).select(link.selection_expr)).opts(width=300, height=200)

        self.figure = pn.Column(
            pn.Row(cmap, reverse_colormap),
            pn.Row(
                _df_plot,
                pn.Column(
                    # pn.panel(pn.bind(random_plot, button), loading_indicator=True),
                    _selection_table,
                    pn.Column(
                        "above in the table is selected points, pick or drop them to generate a new StereoExpData",
                        pn.Row(
                            self.drop_checkbox,
                            # self.bin_select
                        ),
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
