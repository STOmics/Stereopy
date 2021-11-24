#!/usr/bin/env python3
# coding: utf-8
"""
@author: qindanhua@genomics.cn
@time:2021/09/06
"""

import holoviews as hv
import hvplot.pandas
import panel as pn
import collections
from holoviews import opts
from stereo.config import StereoConfig
from natsort import natsorted

conf = StereoConfig()

colormaps = conf.colormaps
pn.param.ParamMethod.loading_indicator = True
theme_default = 'stereo_30'
color_key = collections.OrderedDict()


def interact_spatial_cluster(
        df,
        width=500,
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
    dot_size_default = 1 if len(df) > 20000 else int(100000 / len(df))

    bg_colorpicker = pn.widgets.ColorPicker(name='background color', value='#ffffff', width=200)
    dot_slider = pn.widgets.IntSlider(name='dot size', value=dot_size_default, start=1, end=200, step=1, width=200)
    cs = natsorted(set(df['group']))

    theme_select = pn.widgets.Select(name='color theme', options=list(colormaps.keys()), value=theme_default, width=200)
    cluster_select = pn.widgets.Select(name='cluster', options=cs, value=cs[0], width=100, loading=False)

    ##
    if len(cs) > len(colormaps[theme_default]):
        colormaps[theme_default] = conf.get_colors(theme_default, n=len(cs))

    global color_key
    color_key = collections.OrderedDict({k: c for k, c in zip(cs, colormaps[theme_default][0:len(cs)])})
    # print(color_key)
    # print(cs)
    ct_colorpicker = pn.widgets.ColorPicker(name='node color', value=color_key[cs[0]], width=70)

    @pn.depends(bg_colorpicker, dot_slider, theme_select, ct_colorpicker)
    def _df_plot(bg_color, dot_size, color_theme, cluster_color):
        global theme_default
        global color_key
        cluster_name = cluster_select.value
        if color_theme != theme_default:
            color_key = collections.OrderedDict({k: c for k, c in zip(cs, colormaps[color_theme][0:len(cs)])})
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
            ))
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
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


if __name__ == '__main__':
    a = [(12, 51, 131), (10, 136, 186), (242, 211, 56), (242, 143, 56), (217, 30, 30)]
    print(rgb_to_hex((12, 51, 131)))

