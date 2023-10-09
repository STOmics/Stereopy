#!/usr/bin/env python3
# coding: utf-8
"""
@author: xujunhao@genomics.cn
@time:2022/12/22
"""
import collections

import holoviews as hv
import hvplot.pandas  # noqa
import panel as pn
from natsort import natsorted

from stereo.stereo_config import StereoConfig

conf = StereoConfig()

colormaps = conf.colormaps
pn.param.ParamMethod.loading_indicator = True
theme_default = 'stereo_30'
color_key = collections.OrderedDict()


def interact_spatial_cluster_annotation(
        data,
        df,
        res_marker_gene,
        res_key,
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

    dot_slider = pn.widgets.IntSlider(name='dot size', value=dot_size_default, start=1, end=200, step=1, width=200)
    cs = natsorted(set(df['group']))
    cluster_select = pn.widgets.Select(name='cluster', options=cs, value=cs[0], width=100, loading=False)

    # marker_gene_select = pn.widgets.DataFrame(res_marker_gene['1.vs.rest'].sort_values(by='scores', ascending=False)[['genes','scores']].set_index('scores').head(100), width=200, height=400) # noqa
    if len(cs) > len(colormaps[theme_default]):
        colormaps[theme_default] = conf.get_colors(theme_default, n=len(cs))

    global color_key
    color_key = collections.OrderedDict({k: c for k, c in zip(cs, colormaps[theme_default][0:len(cs)])})

    cluster_text = pn.widgets.TextInput(name='annotation', width=100)

    global flag
    flag = 1

    # global cluster_name
    # cluster_name = cluster_select.value

    @pn.depends(cluster_select)
    def _df_marker_gene(x):
        marker_cluster_select = x + '.vs.rest'
        marker_gene_data = res_marker_gene[marker_cluster_select].sort_values(by='scores', ascending=False)
        marker_gene_data = marker_gene_data[['genes', 'scores']]
        marker_gene_data.set_index('scores', inplace=True)
        marker_gene_data = marker_gene_data.head(100)
        marker_gene_select = pn.widgets.DataFrame(marker_gene_data, width=200, height=400)
        # marker_gene_select = pn.widgets.DataFrame(res_marker_gene[marker_cluster_select].sort_values(by='scores', ascending=False)[['genes','scores']].set_index('scores').head(100), width=200, height=400) # noqa

        return marker_gene_select

    @pn.depends(dot_slider, cluster_text)
    def _df_plot(dot_size, cluster_text):
        global theme_default
        global color_key
        global flag
        cluster_name = cluster_select.value
        if flag == 1:
            flag += 1
        else:
            df.loc[df['group'] == cluster_name, 'group'] = cluster_text
            color_key[cluster_text] = color_key.pop(cluster_name)

        sfig = df.hvplot.scatter(
            x='x', y='y',
            by='group',
            size=dot_size,
            muted_alpha=0,
            width=width,
            height=height,
            padding=(0.1, 0.1)
        ).opts(bgcolor='#ffffff',
               invert_yaxis=True,
               aspect='equal',
               active_tools=['wheel_zoom']
               )
        return sfig.opts(
            hv.opts.Scatter(
                color=hv.dim('group').categorize(color_key)
            ))

    button_save = pn.widgets.Button(name='Save annotation', width=200)

    def save_annotation(event):
        data.tl.result[res_key] = df[['bins', 'group']]
        key = 'cluster'
        data.tl.reset_key_record(key, res_key)
        gene_cluster_res_key = f'gene_exp_{res_key}'
        from stereo.utils.pipeline_utils import cell_cluster_to_gene_exp_cluster

        gene_exp_cluster_res = cell_cluster_to_gene_exp_cluster(data, res_key)
        if gene_exp_cluster_res is not False:
            data.tl.result[gene_cluster_res_key] = gene_exp_cluster_res
            data.tl.reset_key_record('gene_exp_cluster', gene_cluster_res_key)

    button_save.on_click(save_annotation)

    coms = pn.Row(
        _df_plot,
        pn.Column(
            dot_slider,
            pn.Row(cluster_select, cluster_text),
            button_save,
            _df_marker_gene
        )
    )
    return coms
