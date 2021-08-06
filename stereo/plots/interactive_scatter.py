#!/usr/bin/env python3
# coding: utf-8
"""
@author: qindanhua
@time:2021/08/04
"""
import pandas as pd
import numpy as np
from colorcet import palette
from holoviews.selection import link_selections
import holoviews as hv
from holoviews.element.tiles import EsriImagery
import hvplot.pandas
import panel as pn
import param
import io
from typing import Optional
import datashader as ds
import holoviews.operation.datashader as hd
from stereo.core.stereo_exp_data import StereoExpData

colormaps = {n: palette[n] for n in ['rainbow', 'fire', 'bgy', 'bgyw', 'bmy', 'gray', 'kbc', 'CET_D4']}
link = link_selections.instance()
pn.param.ParamMethod.loading_indicator = True


class InteractiveScatter:
    """
    Interactive scatter
    """
    def __init__(
            self,
            data: Optional[StereoExpData],
            width: Optional[int] = 700,
            height: Optional[int] = 600,
            bgcolor='#23238E'
    ):
        self.data = data
        self.width = width
        self.height = height
        self.bgcolor = bgcolor
        # self.link = link_selections.instance()
        self.scatter_df = pd.DataFrame({
            'cell': self.data.cell_names,
            'x': self.data.position[:, 0],
            'y': self.data.position[:, 1],
            'count': np.array(self.data.exp_matrix.sum(axis=1))[:, 0]
        })
        self.selected_exp_data = None

    def get_fig_opts(self):
        """
        set figure options
        :return:
        """
        topts = hv.opts.Tiles(
            width=self.width, height=self.height,
            bgcolor=self.bgcolor,
        )
        return EsriImagery().opts(topts)

    def generate_selected_expr_matrix(self, selected_pos):
        if selected_pos is not None:
            self.selected_exp_data = self.data.sub_by_name(cell_name=selected_pos)
        else:
            self.selected_exp_data = None

    @param.depends(link.param.selection_expr)
    def _download_callback(self, _):
        import time
        sio = io.StringIO()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        selected_pos = hv.Dataset(self.scatter_df).select(link.selection_expr).data[['cell']].values
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.generate_selected_expr_matrix(selected_pos)
        print('tocsv')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.selected_exp_data.to_df().to_csv(sio, index=False)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        sio.seek(0)
        return sio

    def interact_scatter(self):
        cmap = pn.widgets.Select(value='rainbow', options=colormaps, name='colormaps')
        # alpha = pn.widgets.FloatSlider(value=1)
        reverse_colormap = pn.widgets.Checkbox(name='reverse_colormap')
        opts = self.get_fig_opts()
        scatter_df = self.scatter_df.copy()

        download = pn.widgets.FileDownload(
            filename='exp_matrix.csv',
            label='pick',
            callback=self._download_callback, button_type="primary",
            width=100
        )
        download2 = pn.widgets.FileDownload(
            filename='exp_matrix.csv',
            label='drop',
            # callback=self._download_callback,
            # button_type="primary",
            width=100
        )

        @pn.depends(cmap, reverse_colormap)
        def _df_plot(cmap, reverse_colormap):
            cmap = cmap if not reverse_colormap else cmap[::-1]
            return link(opts * scatter_df.hvplot.scatter(
                x='x', y='y', c='count', cnorm='eq_hist',
                cmap=cmap,
                # rasterize=True,
                datashade=True,
                dynspread=True,

            ), selection_mode='union')

        @param.depends(link.param.selection_expr)
        def _selection_table(_):
            return hv.element.Table(hv.Dataset(scatter_df).select(link.selection_expr)).opts(width=300, height=200)

        return pn.Column(
            pn.Row(cmap, reverse_colormap),
            pn.Row(
                _df_plot,
                pn.Column(
                    # pn.panel(pn.bind(random_plot, button), loading_indicator=True),
                    _selection_table,
                    pn.Column(
                        "above in the table is selected points, pick or drop them to generate a new expression matrix",
                        download,
                        'or',
                        download2,
                        "self.selected_exp_data will be the new expression matrix object"
                    ),
                ))
        )
