import os
from collections import OrderedDict

import holoviews as hv
import hvplot.pandas  # noqa
import hvplot.xarray  # noqa
import numpy as np
import panel as pn
import spatialpandas as spd
import tifffile as tiff
import xarray as xr
from bokeh.models import HoverTool
from natsort import natsorted

from stereo.stereo_config import stereo_conf


class PlotCells:
    def __init__(
            self,
            data,
            color_by='total_count',
            color_key=None,
            # cluster_res_key='cluster',
            bgcolor='#2F2F4F',
            width=None,
            height=None,
            fg_alpha=0.5,
            base_image=None,
            base_im_to_gray=False
    ):
        self.data = data
        # if cluster_res_key in self.data.tl.result:
        #     res = self.data.tl.result[cluster_res_key]
        #     self.cluster_res = np.array(res['group'])
        #     self.cluster_id = natsorted(np.unique(self.cluster_res).tolist())
        #     n = len(self.cluster_id)
        #     cmap = stereo_conf.get_colors('stereo_30', n)
        #     self.cluster_color_map = OrderedDict({k: v for k, v in zip(self.cluster_id, cmap)})
        # else:
        #     self.cluster_res = None
        #     self.cluster_id = []
        if color_by != 'cluster':
            self.cluster_res = None
            self.cluster_id = []
            if color_by == 'gene':
                if not np.any(np.isin(self.data.genes.gene_name, color_key)):
                    raise ValueError(f'The gene {color_key} is not found.')
        else:
            if color_key in self.data.tl.result:
                res = self.data.tl.result[color_key]
                self.cluster_res = np.array(res['group'])
                self.cluster_id = natsorted(np.unique(self.cluster_res).tolist())
                n = len(self.cluster_id)
                cmap = stereo_conf.get_colors('stereo_30', n)
                self.cluster_color_map = OrderedDict({k: v for k, v in zip(self.cluster_id, cmap)})
            else:
                self.cluster_res = None
                self.cluster_id = []

        self.color_by_input = color_by
        self.color_key = color_key
        self.bgcolor = bgcolor
        self.width, self.height = self._set_width_and_height(width, height)
        self.fg_alpha = fg_alpha
        self.base_image = base_image

        if self.fg_alpha < 0:
            self.fg_alpha = 0.3
        elif self.fg_alpha > 1:
            self.fg_alpha = 1

        if self.base_image is None:
            self.fg_alpha = 1

        self.hover_fg_alpha = self.fg_alpha / 2
        self.figure_polygons = None
        self.figure_points = None
        self.base_im_to_gray = base_im_to_gray

    def _set_width_and_height(self, width, height):
        if width is None or height is None:
            # width = 500
            # min_position = np.min(self.data.position, axis=0)
            # max_position = np.max(self.data.position, axis=0)
            # p_width, p_height = max_position - min_position
            # height = int(np.ceil(width / (p_width / p_height)))
            if width is None and height is not None:
                width = height
            elif width is not None and height is None:
                height = width
            else:
                width, height = 500, 500
            return width, height
        return width, height

    def _get_base_image_boundary(self, image_data: np.ndarray):
        min_x, max_x, min_y, max_y = -1, -1, -1, -1
        if len(image_data.shape) == 3:
            image_data = image_data.sum(axis=2)
        col_sum = image_data.sum(axis=0)
        nonzero_idx = np.nonzero(col_sum)[0]
        if nonzero_idx.size > 0:
            min_x, max_x = np.min(nonzero_idx), np.max(nonzero_idx)

        row_sum = image_data.sum(axis=1)
        nonzero_idx = np.nonzero(row_sum)[0]
        if nonzero_idx.size > 0:
            min_y, max_y = np.min(nonzero_idx), np.max(nonzero_idx)

        return min_x, max_x, min_y, max_y

    def _create_base_image_xarray(self):
        assert os.path.exists(self.base_image), f'{self.base_image} is not exists!'

        image_xarray = None
        with tiff.TiffFile(self.base_image) as tif:
            # image_data = tiff.imread(self.base_image)
            image_data = tif.asarray()
            if len(image_data.shape) == 3 and self.base_im_to_gray:
                from cv2 import cvtColor, COLOR_BGR2GRAY
                image_data = cvtColor(image_data[:, :, [2, 1, 0]], COLOR_BGR2GRAY)
            if len(image_data.shape) == 3 and image_data.dtype == np.uint16:
                # from stereo.image.tissue_cut.tissue_cut_utils.tissue_seg_utils import transfer_16bit_to_8bit
                # image_data = transfer_16bit_to_8bit(image_data)
                from matplotlib.colors import Normalize
                if tif.shaped_metadata is not None:
                    metadata = tif.shaped_metadata[0]
                    if 'value_range' in metadata:
                        vmin, vmax = metadata['value_range']
                    else:
                        vmin, vmax = np.min(image_data), np.max(image_data)
                else:
                    vmin, vmax = np.min(image_data), np.max(image_data)
                image_data = Normalize(vmin, vmax)(image_data).data
            min_x, max_x, min_y, max_y = self._get_base_image_boundary(image_data)
            if min_x == -1 or max_x == -1 or min_y == -1 or max_y == -1:
                raise Exception("the base image is empty.")
            max_x += 1
            max_y += 1
            min_x = max(0, min_x - 100)
            max_x = min(max_x + 100, image_data.shape[1])
            min_y = max(0, min_y - 100)
            max_y = min(max_y + 100, image_data.shape[0])
            image_data = image_data[min_y:max_y, min_x:max_x]
            if len(image_data.shape) == 2:
                image_xarray = xr.DataArray(data=image_data, coords=[range(min_y, max_y), range(min_x, max_x)], dims=['y', 'x'])
            elif len(image_data.shape) == 3:
                bg_pixel = np.array([0, 0, 0], dtype=image_data.dtype)
                if image_data.dtype == np.uint8:
                    bg_value = 255
                else:
                    bg_value = 1.0
                bg_mask = np.where(image_data == bg_pixel, bg_value, 0)
                image_data += bg_mask
                image_xarray = xr.DataArray(data=image_data, coords=[range(min_y, max_y), range(min_x, max_x), range(0, image_data.shape[2])], dims=['y', 'x', 'channel'])
        return image_xarray

    def _create_polygons(self, color_by):
        polygons = []
        color = []
        position = []
        if color_by == 'gene':
            in_bool = np.isin(self.data.genes.gene_name, self.color_key)
        for i, cell_border in enumerate(self.data.cells.cell_border):
            cell_border = cell_border[cell_border[:, 0] < 32767] + self.data.position[i]
            cell_border = cell_border.reshape((-1,)).tolist()
            polygons.append([cell_border])
            if color_by == 'total_count':
                color.append(self.data.cells.total_counts[i])
            elif color_by == 'n_genes_by_counts':
                color.append(self.data.cells.n_genes_by_counts[i])
            elif color_by == 'gene':
                color.append(self.data.exp_matrix[i, in_bool].sum())
            elif color_by == 'cluster':
                color.append(self.cluster_res[i] if self.cluster_res is not None else self.data.cells.total_counts[i])
            else:
                color.append(self.data.cells.total_counts[i])
            position.append(str(tuple(self.data.position[i].astype(np.uint32))))

        polygons = spd.geometry.PolygonArray(polygons)
        polygons_detail = spd.GeoDataFrame({
            'polygons': polygons,
            'color': color,
            'position': position,
            'total_counts': self.data.cells.total_counts.astype(np.uint32),
            'pct_counts_mt': self.data.cells.pct_counts_mt,
            'n_genes_by_counts': self.data.cells.n_genes_by_counts.astype(np.uint32),
            'cluster_id': np.zeros_like(self.data.cell_names) if self.cluster_res is None else self.cluster_res
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
        self.color_map_key_continuous = pn.widgets.Select(
            value='stereo', options=list(stereo_conf.linear_colormaps.keys()), name='color theme', width=200
        )
        self.color_map_key_discrete = pn.widgets.Select(
            value='stereo_30', options=list(stereo_conf.colormaps.keys()), name='color theme', width=200
        )
        if self.cluster_res is None:
            self.color_map_key_discrete.visible = False
        else:
            self.color_map_key_continuous.visible = False
        self.reverse_colormap = pn.widgets.Checkbox(
            name='reverse colormap', value=False, disabled=False if self.cluster_res is None else True
        )
        self.size_input = pn.widgets.IntInput(name='size', value=500, start=300, end=1000, step=10, width=200)
        color_by_key = []
        if self.cluster_res is not None:
            default_cluster_id = self.cluster_id[0]
            default_cluster_color = self.cluster_color_map[default_cluster_id]
            self.cluster = pn.widgets.Select(
                value=default_cluster_id, options=self.cluster_id, name='cluster', width=100
            )
            self.cluster_colorpicker = pn.widgets.ColorPicker(
                name='cluster color', value=default_cluster_color, width=70
            )
            color_by_key.append('cluster')
        else:
            self.cluster = pn.widgets.Select(name='cluster', width=100, disabled=True, visible=False)
            self.cluster_colorpicker = pn.widgets.ColorPicker(
                name='cluster color', width=70, disabled=True, visible=False
            )
        color_by_key.extend(['total_count', 'n_genes_by_counts', 'gene'])
        # self.color_by = pn.widgets.Select(name='color by', options=color_by_key, value=color_by_key[0], width=200)
        self.color_by = pn.widgets.Select(name='color by', options=color_by_key, value=self.color_by_input, width=200)
        # if self.color_by_input == 'gene':
        #     gene_names_selector_value = [self.color_key] if isinstance(self.color_key, str) else self.color_key
        # else:
        #     i = np.argmax(self.data.genes.n_counts)
        #     gene_names_selector_value = [self.data.genes.gene_name[i]]
        # if isinstance(gene_names_selector_value, np.ndarray):
        #     gene_names_selector_value = gene_names_selector_value.tolist()
        # self.gene_names = pn.widgets.MultiSelect(name='gene names', value=gene_names_selector_value, options=self.data.genes.gene_name.tolist(), size=10, width=200)
        if self.color_by_input == 'gene':
            gene_names_selector_value = self.color_key
        else:
            i = np.argmax(self.data.genes.n_counts)
            gene_names_selector_value = self.data.genes.gene_name[i]
        self.gene_names = pn.widgets.Select(name='gene names', value=gene_names_selector_value, 
                                            options=self.data.genes.gene_name.tolist(), size=10, width=200)


    def show(self):
        assert self.data.cells.cell_border is not None

        pn.param.ParamMethod.loading_indicator = True
        pn.extension()
        hv.extension('bokeh')

        self._create_widgets()

        @pn.depends(self.color_map_key_continuous, self.color_map_key_discrete, self.color_by, self.reverse_colormap,
                    self.cluster_colorpicker, self.gene_names)
        def _create_figure(cm_key_continuous_value, cm_key_discrete_value, color_by_value, reverse_colormap_value,
                           cluster_colorpicker_value, gene_names_value):
            if color_by_value == 'gene':
                self.color_key = gene_names_value
                self.gene_names.visible = True
                # self.gene_names.disabled = False
            else:
                self.gene_names.visible = False
                # self.gene_names.disabled = True
            
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

                cmap = stereo_conf.linear_colors(cm_key_value, reverse=reverse_colormap_value)
            else:
                self.cluster.visible = True
                self.cluster_colorpicker.visible = True
                if self.color_map_key_continuous.visible is True:
                    self.color_map_key_continuous.visible = False
                    self.color_map_key_discrete.visible = True

                self.cluster_color_map[self.cluster.value] = cluster_colorpicker_value
                cmap = list(self.cluster_color_map.values())

            if self.cluster_res is None or color_by_value != 'cluster':
                color = 'color'
                show_legend = False
                colorbar = True
            else:
                color = hv.dim('color').categorize(self.cluster_color_map)
                show_legend = True
                colorbar = False
            self.figure_polygons = polygons_detail.hvplot.polygons(
                'polygons', hover_cols=vdims
            ).opts(
                bgcolor=self.bgcolor,
                color=color,
                cnorm='eq_hist',
                cmap=cmap,
                width=self.width,
                height=self.height,
                xaxis='bare',
                yaxis='bare',
                invert_yaxis=True,
                line_width=1,
                line_alpha=0,
                hover_line_alpha=1,
                fill_alpha=self.fg_alpha,
                hover_fill_alpha=self.hover_fg_alpha,
                active_tools=['wheel_zoom'],
                tools=[hover_tool],
                show_legend=show_legend,
                colorbar=colorbar 
            )

            if self.base_image is not None:
                base_image_points_detail = self._create_base_image_xarray()
                if len(base_image_points_detail.shape) == 2:
                    self.figure_points = base_image_points_detail.hvplot(
                        cmap='gray', cnorm='eq_hist', hover=False, colorbar=False,
                        # datashade=True, dynspread=True
                        rasterize=True, aggregator='mean', dynspread=True
                    ).opts(
                        bgcolor=self.bgcolor,
                        width=self.width,
                        height=self.height,
                        xaxis='bare',
                        yaxis='bare',
                        invert_yaxis=True
                    )
                else:
                    self.figure_points = base_image_points_detail.hvplot.rgb(
                        x='x', y='y', bands='channel', hover=False, 
                        rasterize=True, aggregator='mean', dynspread=True
                    ).opts(
                        bgcolor=self.bgcolor,
                        width=self.width,
                        height=self.height,
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
                self.color_map_key_continuous,
                self.color_map_key_discrete,
                self.color_by,
                self.reverse_colormap,
                pn.Row(self.cluster, self.cluster_colorpicker),
                self.gene_names
            )
        )
