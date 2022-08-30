import os
import traceback
import cv2
import holoviews as hv
import hvplot.pandas
import spatialpandas as spd
import pandas as pd
import panel as pn
from bokeh.models import HoverTool
import numpy as np
from natsort import natsorted
import tifffile as tif
from collections import OrderedDict
# from selenium import webdriver
# from chromedriver_py import binary_path
from stereo.config import StereoConfig

conf = StereoConfig()

class PlotCells:
    def __init__(self, data, cluster_res_key='cluster', bgcolor='#2F2F4F', figure_size=500, fg_alpha=0.8, base_image=None):
        self.data = data
        if cluster_res_key in self.data.tl.result:
            res = self.data.tl.result[cluster_res_key]
            self.cluster_res = np.array(res['group'])
            self.cluster_id = natsorted(np.unique(self.cluster_res).tolist())
            n = len(self.cluster_id)
            cmap = conf.get_colors('stereo_30', n)
            self.cluster_color_map = OrderedDict({k: v for k, v in zip(self.cluster_id, cmap)})
        else:
            self.cluster_res = None
            self.cluster_id = []
        
        self.bgcolor = bgcolor
        self.figure_size = figure_size
        self.fg_alpha = fg_alpha
        self.base_image = base_image

        if self.figure_size < 500:
            self.figure_size = 500
        elif self.figure_size > 1000:
            self.figure_size = 1000
        
        if self.fg_alpha < 0.5:
            self.fg_alpha = 0.5
        elif self.fg_alpha > 1:
            self.fg_alpha = 1
        
        if self.base_image is None:
            self.fg_alpha = 1
        
        if self.fg_alpha > 0.5:
            self.hover_fg_alpha = 0.5
        else:
            self.hover_fg_alpha = self.fg_alpha - 0.1 if self.fg_alpha > 0.2 else self.fg_alpha

        self.figure_polygons = None
        self.figure_points = None


    def _create_base_image_points(self):
        assert os.path.exists(self.base_image), f'{self.base_image} is not exists!'

        image_data = tif.imread(self.base_image)
        y, x = np.nonzero(image_data)
        data = image_data[y, x]
        points_detail = pd.DataFrame({
            'x': x,
            'y': y,
            'value': data
        }, dtype='uint32')
        return points_detail
    
    def _create_polygons(self, color_by):
        polygons = []
        color = []
        position = []
        # cell_count = len(self.data.cell_names)
        # cell_idx = []
        for i, cell_border in enumerate(self.data.cells.cell_border):
            cell_border = cell_border[cell_border[:, 0] < 32767] + self.data.position[i]
            cell_border = cell_border.reshape((-1, )).tolist()
            polygons.append([cell_border])
            if color_by == 'total_count':
                color.append(self.data.cells.total_counts[i])
            elif color_by == 'n_genes_by_counts':
                color.append(self.data.cells.n_genes_by_counts[i])
            elif color_by == 'cluster':
                color.append(self.cluster_res[i] if self.cluster_res is not None else self.data.cells.total_counts[i])
            else:
                color.append(self.data.cells.total_counts[i])
            position.append(str(tuple(self.data.position[i].astype(np.uint32))))
            # cell_idx.append(i)
        
        polygons = spd.geometry.PolygonArray(polygons)
        polygons_detail = spd.GeoDataFrame({
            'polygons': polygons,
            'color': color,
            'position': position,
            'total_counts': self.data.cells.total_counts.astype(np.uint32),
            'pct_counts_mt': self.data.cells.pct_counts_mt,
            'n_genes_by_counts': self.data.cells.n_genes_by_counts.astype(np.uint32),
            'cluster_id': np.zeros_like(self.data.cell_names) if self.cluster_res is None else self.cluster_res.astype(np.int16)
        })

        tooltips = [
        ('Position', '@position'),
        ('Total Counts', '@total_counts'),
        ('Pct Counts Mt', '@pct_counts_mt'),
        ('nGenes By Counts', '@n_genes_by_counts'),
        ('Cluster Id', '@cluster_id')]
        hover_tool = HoverTool(tooltips=tooltips)

        vdims = polygons_detail.columns.tolist()
        vdims.remove('polygons')

        return polygons_detail, hover_tool, vdims
    
    def _create_widgets(self):
        self.color_map_key_continuous = pn.widgets.Select(value='stereo', options=list(conf.linear_colormaps.keys()), name='color theme', width=200)
        self.color_map_key_discrete = pn.widgets.Select(value='stereo_30', options=list(conf.colormaps.keys()), name='color theme', width=200)
        if self.cluster_res is None:
            self.color_map_key_discrete.visible = False
        else:
            self.color_map_key_continuous.visible = False
        self.reverse_colormap = pn.widgets.Checkbox(name='reverse colormap', value=False, disabled=False if self.cluster_res is None else True)
        self.size_input = pn.widgets.IntInput(name='size', value=500, start=300, end=1000, step=10, width=200)
        color_by_key = []
        if self.cluster_res is not None:
            default_cluster_id = self.cluster_id[0]
            default_cluster_color = self.cluster_color_map[default_cluster_id]
            self.cluster = pn.widgets.Select(value=default_cluster_id, options=self.cluster_id, name='cluster', width=100)
            self.cluster_colorpicker = pn.widgets.ColorPicker(name='cluster color', value=default_cluster_color, width=70)
            color_by_key.append('cluster')
        else:
            self.cluster = pn.widgets.Select(name='cluster', width=100, disabled=True, visible=False)
            self.cluster_colorpicker = pn.widgets.ColorPicker(name='cluster color', width=70, disabled=True, visible=False)
        color_by_key.extend(['total_count', 'n_genes_by_counts'])
        self.color_by = pn.widgets.Select(name='color by', options=color_by_key, value=color_by_key[0], width=200)
        self.export = pn.widgets.Button(name='export', button_type="primary", width=100)
        self.export.on_click(self._export_callback)
    
    def _export_callback(self, _):
        pass
        # if self.figure_polygons is None:
        #     return
        # file_logger.info("start to export")
        # self.export.loading = True
        # try:
        #     from bokeh.io import export_svg
        #     plot = hv.render(self.figure_polygons)
        #     file_logger.info("aaaaaaaaaaaaaaaaaaa")
        #     plot.output_backend = 'svg'
        #     file_logger.info(binary_path)
        #     driver = webdriver.Chrome(executable_path=binary_path)
        #     export_svg(plot, filename='plot.svg', webdriver=driver)
        #     file_logger.info("bbbbbbbbbbbbbbbbbbb")
        # except:
        #     exception = traceback.format_exc()
        #     file_logger.error(exception)
        # finally:
        #     self.export.loading = False
        # self.export.loading = True
        # try:
        #     file_logger.info("start to export")
        #     hv.save(self.figure_polygons, 'aaa.png', fmt='png', backend='matplotlib', toolbar=False)
        #     # hv.save(self.figure_polygons, 'aaa.png', fmt='png', backend='bokeh', toolbar=False)
        #     file_logger.info("exporting finished")
        # except:
        #     exception = traceback.format_exc()
        #     file_logger.error(exception)
        # finally:
        #     self.export.loading = False

    def show(self):
        assert self.data.cells.cell_border is not None

        pn.param.ParamMethod.loading_indicator = True
        pn.extension()
        hv.extension('bokeh')
        # hv.extension('matplotlib')
        # hv.output(fig='svg')        
        
        self._create_widgets()

        @pn.depends(self.color_map_key_continuous, self.color_map_key_discrete, self.color_by, self.reverse_colormap, self.cluster_colorpicker)
        def _create_figure(cm_key_continuous_value, cm_key_discrete_value, color_by_value, reverse_colormap_value, cluster_colorpicker_value):
            polygons_detail, hover_tool, vdims = self._create_polygons(color_by_value)
            
            if self.cluster_res is None or color_by_value != 'cluster':
                self.cluster.visible = False
                self.cluster_colorpicker.visible = False
                self.reverse_colormap.disabled = False
                if self.color_map_key_discrete.visible is True:
                    cm_key_value = 'stereo'
                    self.color_map_key_continuous.visible = True
                    self.color_map_key_discrete.visible = False
                else:
                    cm_key_value = cm_key_continuous_value

                cmap = conf.linear_colors(cm_key_value, reverse=reverse_colormap_value)
            else:
                self.cluster.visible = True
                self.cluster_colorpicker.visible = True
                if self.color_map_key_continuous.visible is True:
                    cm_key_value = 'stereo_30'
                    self.color_map_key_continuous.visible = False
                    self.color_map_key_discrete.visible = True
                else:
                    cm_key_value = cm_key_discrete_value

                self.cluster_color_map[self.cluster.value] = cluster_colorpicker_value
                cmap = list(self.cluster_color_map.values())
            
            self.figure_polygons = polygons_detail.hvplot.polygons(
                'polygons',
                # c='color' if cluster_res is None or color_by_value != 'cluster' else hv.dim('color').categorize(cluster_color_map),
                # datashade=True,
                # dynspread=True,
                # aggregator='mean',
                hover_cols=vdims
                ).opts(
                    bgcolor=self.bgcolor,
                    color='color' if self.cluster_res is None or color_by_value != 'cluster' else hv.dim('color').categorize(self.cluster_color_map),
                    cnorm='eq_hist',
                    cmap=cmap,
                    colorbar=False,
                    width=self.figure_size,
                    height=self.figure_size,
                    xaxis='bare',
                    yaxis='bare',
                    invert_yaxis=True,
                    line_width=1,
                    line_alpha=0,
                    hover_line_alpha=1,
                    fill_alpha=self.fg_alpha,
                    hover_fill_alpha=self.hover_fg_alpha,
                    active_tools=['wheel_zoom'],
                    # tools=[hover_tool, 'lasso_select']
                    tools=[hover_tool]
                )
            if self.base_image is not None:
                base_image_points_detail = self._create_base_image_points()
                self.figure_points = base_image_points_detail.hvplot.scatter(
                    x='x', y='y',
                    c='value', cmap='gray', cnorm='eq_hist',
                    datashade=True, dynspread=True
                    ).opts(
                        bgcolor=self.bgcolor,
                        width=self.figure_size,
                        height=self.figure_size,
                        xaxis='bare',
                        yaxis='bare',
                        invert_yaxis=True
                    )
                figure = self.figure_points * self.figure_polygons
            else:
                figure = self.figure_polygons
            return figure
        
        return pn.Row(
            pn.Column(_create_figure),
            pn.Column(
                self.color_map_key_continuous, self.color_map_key_discrete, self.color_by, self.reverse_colormap,
                pn.Row(self.cluster, self.cluster_colorpicker),
                # self.export
                )
        )