#!/usr/bin/env python3
# coding: utf-8
"""
@author: qindanhua@genomics.cn
@time:2021/09/06
"""

import holoviews as hv
import hvplot.pandas
import colorcet as cc
import numpy as np
import panel as pn
from colorcet import palette
from holoviews import opts

colormaps = {n: palette[n] for n in ['glasbey', 'glasbey_bw', 'glasbey_cool', 'glasbey_warm', 'glasbey_dark',
                                     'glasbey_light', 'glasbey_category10', 'glasbey_hv']}
pn.param.ParamMethod.loading_indicator = True
theme_default = 'glasbey_category10'
color_key = {}


def interact_spatial_cluster(
        df,
        width=700,
        height=500,
):
    """
    spatial distribution color mapped by cluster

    :param df: data frame, eg:
       x  y  group
    0  1  2        0
    1  2  3        1
    2  3  4        2
    :param width: width
    :param height: height

    :return: panel widgets
    """
    # for notebook show
    pn.extension()
    hv.extension('bokeh')
    # default setting
    dot_size_default = int(120000 / len(df))

    bg_colorpicker = pn.widgets.ColorPicker(name='background color', value='#000000', width=200)
    dot_slider = pn.widgets.IntSlider(name='dot size', value=dot_size_default, start=1, end=200, step=1, width=200)
    cs = list(sorted(np.array(list(s for s in set(df['group'])))))

    theme_select = pn.widgets.Select(name='color theme', options=list(colormaps.keys()), value=theme_default, width=200)
    cluster_select = pn.widgets.Select(name='cluster', options=cs, value=cs[0], width=100, loading=False)

    global color_key
    color_key = {k: c for k, c in zip(cs, colormaps[theme_default][0:len(cs)])}
    # print(color_key)
    # print(cs)
    ct_colorpicker = pn.widgets.ColorPicker(name='node color', value=color_key[cs[0]], width=70)

    @pn.depends(bg_colorpicker, dot_slider, theme_select, ct_colorpicker)
    def _df_plot(bg_color, dot_size, color_theme, cluster_color):
        global theme_default
        global color_key
        cluster_name = cluster_select.value
        if color_theme != theme_default:
            color_key = {k: c for k, c in zip(cs, colormaps[color_theme][0:len(cs)])}
            cluster_select.value = cs[0]
            ct_colorpicker.value = color_key[cs[0]]
        else:
            color_key[cluster_name] = cluster_color
        theme_default = color_theme
        sfig = df.hvplot.scatter(
            x='x', y='y',
            by='group',
            size=dot_size,
            muted_alpha=0,
            width=width,
            height=height,
            padding=(0.1, 0.1)
        ).opts(bgcolor=bg_color,
               invert_yaxis=True,
               aspect='equal'
               # legend_muted=True,
               # legend_cols=2
               )
        return sfig.opts(
            hv.opts.Scatter(
                color=hv.dim('group').categorize(color_key)
            )
        )
    coms = pn.Row(
        _df_plot,
        pn.Column(
            bg_colorpicker,
            dot_slider,
            theme_select,
            pn.Row(
                cluster_select,
                ct_colorpicker
            )
        )
    )
    return coms


# def hex_to_rgb(value):
#     value = value.lstrip('#')
#     lv = len(value)
#     return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
#
#
# def rgb_to_hex(rgb):
#     return '#%02x%02x%02x' % rgb
