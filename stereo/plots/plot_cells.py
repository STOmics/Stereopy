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
from stereo.core.stereo_exp_data import StereoExpData
from stereo.preprocess.qc import cal_total_counts, cal_n_genes_by_counts, cal_pct_counts_mt


class PlotCells:
    def __init__(
            self,
            data,
            color_by='total_counts',
            color_key=None,
            # cluster_res_key='cluster',
            bgcolor='#2F2F4F',
            palette=None,
            width=None,
            height=None,
            fg_alpha=0.5,
            base_image=None,
            base_im_to_gray=False,
            use_raw=True
    ):
        self.data: StereoExpData = data
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
            self.palette = 'stereo' if palette is None else palette
            assert isinstance(self.palette, str), f'The palette must be a name of palette when color_by is {color_by}'
            self.cluster_res = None
            self.cluster_id = []
            if color_by == 'gene':
                if not np.any(np.isin(self.data.genes.gene_name, color_key)):
                    raise ValueError(f'The gene {color_key} is not found.')
        else:
            if palette is None:
                self.palette = 'stereo_30'
            elif isinstance(palette, str):
                self.palette = palette
            else:
                self.palette = 'custom'
            # self.palette = 'stereo_30' if palette is None else 'custom'
            if color_key in self.data.tl.result:
                res = self.data.tl.result[color_key]
                self.cluster_res = np.array(res['group'])
                self.cluster_id = natsorted(np.unique(self.cluster_res).tolist())
                n = len(self.cluster_id)
                # if isinstance(palette, str):
                #     cmap = stereo_conf.get_colors(self.palette, n)
                #     self.cluster_color_map = OrderedDict({k: v for k, v in zip(self.cluster_id, cmap)})
                if isinstance(palette, (list, np.ndarray)):
                    stereo_conf.palette_custom = list(palette)
                elif isinstance(palette, dict):
                    stereo_conf.palette_custom = [palette[k] for k in self.cluster_id if k in palette]
                cmap = stereo_conf.get_colors(self.palette, n)
                self.cluster_color_map = OrderedDict({k: v for k, v in zip(self.cluster_id, cmap)})
            else:
                self.cluster_res = None
                self.cluster_id = []

        self.last_cm_key_continuous = None
        self.last_cm_key_discrete = None
        self.color_by_input = color_by
        self.color_key = color_key
        self.bgcolor = bgcolor
        self.width, self.height = self._set_width_and_height(width, height)
        self.fg_alpha = fg_alpha
        self.base_image = base_image
        self.base_image_points = None

        if self.fg_alpha < 0:
            self.fg_alpha = 0.3
        elif self.fg_alpha > 1:
            self.fg_alpha = 1

        if self.base_image is None:
            self.fg_alpha = 1

        self.hover_fg_alpha = self.fg_alpha / 2
        self.figure_polygons = None
        self.figure_points = None
        self.figure_colorbar_legend = None
        self.colorbar_or_legend = None
        self.base_im_to_gray = base_im_to_gray
        self.use_raw = use_raw and self.data.raw is not None
        self.rangexy_stream = None
        self.x_range = None
        self.y_range = None
        self.firefox_path = None
        self.driver_path = None

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
            image_data = tif.asarray()
            if len(image_data.shape) == 3 and self.base_im_to_gray:
                from cv2 import cvtColor, COLOR_BGR2GRAY
                image_data = cvtColor(image_data[:, :, [2, 1, 0]], COLOR_BGR2GRAY)
            if len(image_data.shape) == 3 and image_data.dtype == np.uint16:
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
        if self.use_raw:
            if self.data.shape != self.data.raw.shape:
                cells_isin = np.isin(self.data.raw.cell_names, self.data.cell_names)
                genes_isin = np.isin(self.data.raw.gene_names, self.data.gene_names)
                exp_matrix = self.data.raw.exp_matrix[cells_isin][:, genes_isin]
                gene_names = self.data.raw.gene_names[genes_isin]
            else:
                exp_matrix = self.data.raw.exp_matrix
                gene_names = self.data.raw.gene_names
        else:
            exp_matrix = self.data.exp_matrix
            gene_names = self.data.gene_names
        total_counts = cal_total_counts(exp_matrix)
        n_genes_by_counts = cal_n_genes_by_counts(exp_matrix)
        pct_counts_mt = cal_pct_counts_mt(exp_matrix, gene_names)
        if color_by == 'gene':
            in_bool = np.isin(self.data.genes.gene_name, self.color_key)
        for i, cell_border in enumerate(self.data.cells.cell_border):
            cell_border = cell_border[cell_border[:, 0] < 32767] + self.data.position[i]
            cell_border = cell_border.reshape((-1,)).tolist()
            polygons.append([cell_border])
            if color_by == 'total_counts':
                # color.append(self.data.cells.total_counts[i])
                color.append(total_counts[i])
            elif color_by == 'n_genes_by_counts':
                # color.append(self.data.cells.n_genes_by_counts[i])
                color.append(n_genes_by_counts[i])
            elif color_by == 'gene':
                color.append(exp_matrix[i, in_bool].sum())
            elif color_by == 'cluster':
                # color.append(self.cluster_res[i] if self.cluster_res is not None else self.data.cells.total_counts[i])
                color.append(self.cluster_res[i] if self.cluster_res is not None else total_counts[i])
            else:
                # color.append(self.data.cells.total_counts[i])
                color.append(total_counts[i])
            position.append(str(tuple(self.data.position[i].astype(np.uint32))))

        polygons = spd.geometry.PolygonArray(polygons)
        polygons_detail = spd.GeoDataFrame({
            'polygons': polygons,
            'color': color,
            'position': position,
            # 'total_counts': self.data.cells.total_counts.astype(np.uint32),
            # 'pct_counts_mt': self.data.cells.pct_counts_mt,
            # 'n_genes_by_counts': self.data.cells.n_genes_by_counts.astype(np.uint32),
            'total_counts': total_counts,
            'pct_counts_mt': pct_counts_mt,
            'n_genes_by_counts': n_genes_by_counts,
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
        # self.color_map_key_continuous = pn.widgets.Select(
        #     value='stereo', options=list(stereo_conf.linear_colormaps.keys()), name='color theme', width=200
        # )
        # self.color_map_key_discrete = pn.widgets.Select(
        #     value='stereo_30', options=list(stereo_conf.colormaps.keys()), name='color theme', width=200
        # )
        self.color_map_key_continuous = pn.widgets.Select(
            value=self.palette, options=sorted(list(stereo_conf.linear_colormaps.keys())), name='color theme', width=200
        )
        self.color_map_key_discrete = pn.widgets.Select(
            value=self.palette, options=sorted(list(set(stereo_conf.colormaps.keys()) | {self.palette})), name='color theme', width=200
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
        color_by_key.extend(['total_counts', 'n_genes_by_counts', 'gene'])
        self.color_by = pn.widgets.Select(name='color by', options=color_by_key, value=self.color_by_input, width=200)
        if self.color_by_input == 'gene':
            gene_names_selector_value = self.color_key
        else:
            i = np.argmax(self.data.genes.n_counts)
            gene_names_selector_value = self.data.genes.gene_name[i]
        gene_idx_sorted = np.argsort(self.data.genes.n_counts * -1)
        gene_names_sorted = self.data.genes.gene_name[gene_idx_sorted].tolist()
        self.gene_names = pn.widgets.Select(name='genes', value=gene_names_selector_value, 
                                            options=gene_names_sorted, size=10, width=200)
        
        self.save_title = pn.widgets.StaticText(name='', value='<b>Save Plot</b>', width=200)
        self.save_file_name = pn.widgets.TextInput(name='file name(.png, .svg or .pdf)', width=200)
        self.save_file_width = pn.widgets.IntInput(name='width', value=self.width, width=95)
        self.save_file_hight = pn.widgets.IntInput(name='height', value=self.height, width=95)
        self.save_button = pn.widgets.Button(name='save', button_type="primary", width=100)
        self.save_only_in_view = pn.widgets.Checkbox(name='only in view', value=False, width=100)
        self.with_base_image = pn.widgets.Checkbox(name='with base image', value=False, width=100)
        self.save_button.on_click(self._save_button_callback)
        self.save_message = pn.widgets.StaticText(name='', value='', width=400)
    
    def _set_firefox_and_driver_path(self):
        from sys import executable
        from os import environ
        import platform

        os_type = platform.system().lower()
        executable_dir = os.path.dirname(executable)
        environ_path = environ['PATH'].split(os.pathsep)
        if executable_dir not in environ_path:
            environ_path = [executable_dir] + environ_path
        if os_type == 'windows':
            bin_paths = [
                executable_dir,
                os.path.join(executable_dir, 'Scripts'),
                os.path.join(executable_dir, 'Library', 'bin'),
                os.path.join(executable_dir, 'Library', 'mingw-w64', 'bin'),
                os.path.join(executable_dir, 'Library', 'usr', 'bin'),
                os.path.join(executable_dir, 'bin')
            ]
            for bin_path in bin_paths:
                if bin_path not in environ_path:
                    environ_path = [bin_path] + environ_path
                a_path = os.path.join(bin_path, 'firefox.exe')
                if os.path.exists(a_path) and self.firefox_path is None:
                    self.firefox_path = a_path
                a_path = os.path.join(bin_path, 'geckodriver.exe')
                if os.path.exists(a_path) and self.driver_path is None:
                    self.driver_path = a_path
        elif os_type == 'linux':
            if self.firefox_path is None:
                self.firefox_path = os.path.join(executable_dir, 'firefox')
            if self.driver_path is None:
                self.driver_path = os.path.join(executable_dir, 'geckodriver')
        else:
            raise ValueError(f'The operating system {os_type} is not supported.')
        environ['PATH'] = os.pathsep.join(environ_path)

    def save_plot(
        self,
        save_file_name: str,
        save_width: int = None,
        save_height: int = None,
        save_only_in_view: bool = False,
        with_base_image: bool = False
    ):
        """
        Save the plot to a PNG, SVG or PDF file depending on the extension of the file name.

        :param save_file_name: the name of the file to save the plot.
        :param save_width: the width of the saved plot, defaults to be the same as the plot.
        :param save_height: the height of the saved plot, defaults to be the same as the plot.
        :param save_only_in_view: only save the plot in the view, defaults to False.
        :param with_base_image: whether to save the plot with the base image, defaults to False.
                                Currently, the dpi of the saved base image may not be high.

        """
        self._set_firefox_and_driver_path()

        from selenium import webdriver
        from selenium.webdriver import FirefoxOptions, FirefoxService
        from bokeh.io import export_png, export_svg
        from bokeh.io.export import get_svg
        from bokeh.layouts import row, Row
        from cairosvg import svg2pdf

        if save_file_name == '':
            raise ValueError('Please input the file name.')
        
        if not save_file_name.lower().endswith('.png') and \
            not save_file_name.lower().endswith('.svg') and \
            not save_file_name.lower().endswith('.pdf'):
            raise ValueError('Only PNG, SVG and PDF files are supported.')

        save_width = self.width if save_width is None else save_width
        save_height = self.height if save_height is None else save_height
        if with_base_image:
            figure_points = self._create_base_image_figure()
        else:
            figure_points = None
        with_base_image = with_base_image and figure_points is not None

        file_name_prefix, file_name_suffix = os.path.splitext(save_file_name)
        save_file_name = f"{file_name_prefix}_cells_plotting{file_name_suffix}"

        current_x_range = self.x_range
        current_y_range = self.y_range
        try:
            if save_only_in_view:
                if self.x_range is not None and self.y_range is not None:
                    figure_polygons = self.figure_polygons[self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]]
                    if with_base_image:
                        figure_points = figure_points.select(x=self.x_range, y=self.y_range)
                else:
                    figure_polygons = self.figure_polygons
            else:
                figure_polygons = self.figure_polygons

            if not with_base_image:
                output_render = hv.render(figure_polygons, backend='bokeh')
                for renderer in output_render.renderers:
                    renderer.glyph.fill_alpha = 1
            else:
                output_render = hv.render(figure_points, backend='bokeh')
                polygons_render = hv.render(figure_polygons, backend='bokeh')
                output_render.renderers.extend(polygons_render.renderers)
            output_render.output_backend = 'svg'
            output_render.toolbar_location = None
            output_render.border_fill_color = None
            output_render.outline_line_color = 'gray'
            output_render.xaxis.visible = False
            output_render.yaxis.visible = False
            output_render.width = save_width
            output_render.height = save_height
            
            if save_height == self.height:
                figure_colorbar_legend = self.figure_colorbar_legend
            else:
                figure_colorbar_legend = self._create_colorbar_or_legend(
                    self.colorbar_or_legend,
                    self.figure_polygons.opts['cmap'],
                    self.figure_polygons.data,
                    save_height
                )

            if isinstance(figure_colorbar_legend, Row):
                for f in figure_colorbar_legend.children:
                    f.output_backend = 'svg'
                    f.height = save_height if save_height > 500 else 500
            else:
                figure_colorbar_legend.output_backend = 'svg'
                figure_colorbar_legend.height = save_height if save_height > 500 else 500

            to_save_instance = row(output_render, figure_colorbar_legend)

            opts = FirefoxOptions()
            opts.add_argument("--headless")
            opts.binary_location = self.firefox_path
            service = FirefoxService(executable_path=self.driver_path)
            with webdriver.Firefox(options=opts, service=service) as driver:
                if save_file_name.lower().endswith('png'):
                    export_png(to_save_instance, filename=save_file_name, webdriver=driver, timeout=86400)
                elif save_file_name.lower().endswith('svg'):
                    export_svg(to_save_instance, filename=save_file_name, webdriver=driver, timeout=86400)
                elif save_file_name.lower().endswith('pdf'):
                    svg = get_svg(to_save_instance, driver=driver, timeout=86400)[0]
                    svg2pdf(bytestring=svg, write_to=save_file_name)
        except Exception as e:
            raise e
        finally:
            self.rangexy_stream.event(x_range=current_x_range, y_range=current_y_range)
            if figure_colorbar_legend is self.figure_colorbar_legend:
                if isinstance(figure_colorbar_legend, Row):
                    for f in figure_colorbar_legend.children:
                        f.height = self.height if self.height > 500 else 500
                else:
                    figure_colorbar_legend.height = self.height if self.height > 500 else 500
        return save_file_name
            

    def _save_button_callback(self, _):
        """
        apt-get install libgtk-3-dev libasound2-dev
        conda install -c conda-forge selenium firefox geckodriver cairosvg
        """
        self.save_button.loading = True
        self.save_message.value = ''
        try:
            save_file_name = self.save_plot(
                self.save_file_name.value,
                self.save_file_width.value,
                self.save_file_hight.value,
                save_only_in_view=self.save_only_in_view.value, 
                with_base_image=self.with_base_image.value,
            )
            self.save_message.value = f'<font color="red"><b>The plot has been saved to {save_file_name}.</b></font>'
        except ValueError as e:
            self.save_message.value = f'<font color="red"><b>{str(e)}</b></font>'
        except Exception as e:
            raise e
        finally:
            self.save_button.loading = False


    def _create_colorbar_or_legend(self, type, cmap, plot_data=None, figure_height=None):

        from bokeh.plotting import figure as bokeh_figure
        from bokeh.models import (
            Legend, LegendItem, ColorBar, 
            EqHistColorMapper, BinnedTicker,
            ColumnDataSource
        )
        from bokeh.layouts import row

        figure_height = self.height if figure_height is None else figure_height
        figure_height = 500 if figure_height < 500 else figure_height
        if type == 'colorbar':
            figures = 1
        else:
            legend_items_in_col = int(figure_height / 25)
            legend_counts = len(cmap.keys())
            legend_cols, legend_left = divmod(legend_counts, legend_items_in_col)
            if legend_left > 0:
                legend_cols += 1
            figures = legend_cols

        figure_list = []
        for i in range(figures):
            f = bokeh_figure(width=80, height=figure_height, toolbar_location=None, x_axis_type=None, y_axis_type=None)
            f.outline_line_color = None
            f.xgrid.grid_line_color = None
            f.ygrid.grid_line_color = None
            figure_list.append(f)

        if type == 'colorbar':
            min_value = min(plot_data['color'])
            max_value = max(plot_data['color'])
            ticks_num = 100
            ticks_interval = (max_value - min_value) / (ticks_num - 1)
            ticks = [min(min_value + i * ticks_interval, max_value) for i in range(ticks_num)]
            color_mapper = EqHistColorMapper(palette=cmap, low=min_value, high=max_value)

            data_source = ColumnDataSource(data={
                'x': [0] * len(ticks),
                'y': [0] * len(ticks),
                'color': ticks
            })
            figure_list[0].circle(x='x', y='y', color={'field': 'color', 'transform': color_mapper}, size=0, source=data_source)
            ticker = BinnedTicker(mapper=color_mapper, num_major_ticks=8)
            color_bar = ColorBar(color_mapper=color_mapper, location='center_left' if self.height >= 500 else 'top_left',
                                    orientation='vertical', height=int(figure_height // 1.5), width=int(figure_height / 500 * 20),
                                    major_tick_line_color='black', major_label_text_font_size=f'{int(figure_height / 500 * 11)}px',
                                    major_tick_in=int(figure_height / 500 * 5), major_tick_line_width=int(figure_height / 500 * 1),
                                    ticker=ticker)
            figure_list[0].width = int(figure_height / 500 * 150)
            figure_list[0].add_layout(color_bar, 'center')
            fig = figure_list[0]
        elif type == 'legend':
            legend_labels = list(cmap.keys())
            labels_len = [len(label) for label in legend_labels]
            max_label_len = max(labels_len)
            figure_width = 13 * max_label_len
            if figure_width < 80:
                figure_width = 80
            figure_width = int(figure_height / 500 * figure_width)
            for i in range(legend_cols):
                # legend = Legend(location='top_left', orientation='vertical', border_line_color=None,
                #                 label_text_font_size=f'{int(figure_height / 500 * 13)}px',
                #                 label_height=int(figure_height / 500 * 20), label_width=int(figure_height / 500 * 20))
                legend = Legend(location='top_left', orientation='vertical', border_line_color=None)
                labels = legend_labels[i * legend_items_in_col: (i + 1) * legend_items_in_col]
                for label in labels:
                    color = cmap[label]
                    legend.items.append(LegendItem(label=label, renderers=[figure_list[i].circle(x=[0], y=[0], color=color, size=0)]))
                figure_list[i].add_layout(legend, 'left')
                figure_list[i].width = figure_width
            if len(figure_list) == 1:
                fig = figure_list[0]
            else:
                fig = row(*figure_list, sizing_mode='fixed')
        
        self.colorbar_or_legend = type

        return fig
    
    def _create_base_image_figure(self):
        figure = None
        if self.base_image is not None:
            if self.base_image_points is None:
                self.base_image_points = self._create_base_image_xarray()
            if len(self.base_image_points.shape) == 2:
                figure = self.base_image_points.hvplot(
                    cmap='gray', cnorm='eq_hist', hover=False, colorbar=False,
                    datashade=True, aggregator='mean', dynspread=True
                    # rasterize=True, aggregator='mean', dynspread=True
                ).opts(
                    bgcolor=self.bgcolor,
                    width=self.width,
                    height=self.height,
                    xaxis=None,
                    yaxis=None,
                    invert_yaxis=True
                )
            else:
                figure = self.base_image_points.hvplot.rgb(
                    x='x', y='y', bands='channel', hover=False,
                    datashade=True, aggregator='mean', dynspread=True
                    # rasterize=True, aggregator='mean', dynspread=True
                ).opts(
                    bgcolor=self.bgcolor,
                    width=self.width,
                    height=self.height,
                    xaxis='bare',
                    yaxis='bare',
                    invert_yaxis=True
                )
        return figure
    
    def __rangexy_callback(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range

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
                    cm_key_value = 'stereo' if self.last_cm_key_continuous is None else self.last_cm_key_continuous
                    # cm_key_value = self.palette if self.last_cm_key_continuous is None else self.last_cm_key_continuous
                    self.color_map_key_continuous.visible = True
                    self.color_map_key_continuous.value = cm_key_value
                    self.color_map_key_discrete.visible = False
                else:
                    cm_key_value = cm_key_continuous_value

                cmap = stereo_conf.linear_colors(cm_key_value, reverse=reverse_colormap_value)
                self.last_cm_key_continuous = cm_key_value
            else:
                self.cluster.visible = True
                self.cluster_colorpicker.visible = True
                self.reverse_colormap.disabled = True
                if self.color_map_key_continuous.visible is True:
                    # cm_key_value = 'stereo_30' if self.last_cm_key_discrete is None else self.last_cm_key_discrete
                    cm_key_value = self.palette if self.last_cm_key_discrete is None else self.last_cm_key_discrete
                    self.color_map_key_continuous.visible = False
                    self.color_map_key_discrete.visible = True
                    self.color_map_key_discrete.value = cm_key_value
                else:
                    cm_key_value = cm_key_discrete_value
                
                if cm_key_value != self.last_cm_key_discrete:
                    n = len(self.cluster_id)
                    colors = stereo_conf.get_colors(cm_key_value, n)
                    self.cluster_color_map = OrderedDict({k: v for k, v in zip(self.cluster_id, colors)})
                    default_cluster_id = self.cluster_id[0]
                    default_cluster_color = self.cluster_color_map[default_cluster_id]
                    self.cluster.value = default_cluster_id
                    self.cluster_colorpicker.value = default_cluster_color
                else:
                    self.cluster_color_map[self.cluster.value] = cluster_colorpicker_value
                # cmap = list(self.cluster_color_map.values())
                cmap = self.cluster_color_map
                self.last_cm_key_discrete = cm_key_value

            if self.cluster_res is None or color_by_value != 'cluster':
                self.figure_colorbar_legend = self._create_colorbar_or_legend('colorbar', cmap, polygons_detail)
            else:
                self.figure_colorbar_legend = self._create_colorbar_or_legend('legend', cmap)

            self.figure_polygons = polygons_detail.hvplot.polygons(
                'polygons', hover_cols=vdims
            ).opts(
                bgcolor=self.bgcolor,
                color='color',
                cnorm='eq_hist',
                cmap=cmap,
                width=self.width,
                height=self.height,
                xaxis=None,
                yaxis=None,
                invert_yaxis=True,
                line_width=1,
                line_alpha=0,
                hover_line_alpha=1,
                fill_alpha=self.fg_alpha,
                hover_fill_alpha=self.hover_fg_alpha,
                active_tools=['wheel_zoom'],
                tools=[hover_tool],
                show_legend=False,
                colorbar=False
            )

            if self.base_image is not None:
                if self.figure_points is None:
                    self.figure_points = self._create_base_image_figure()
                figure = self.figure_points * self.figure_polygons
            else:
                figure = self.figure_polygons
            
            self.rangexy_stream = hv.streams.RangeXY(source=self.figure_polygons)
            self.rangexy_stream.add_subscriber(self.__rangexy_callback)

            return pn.Row(figure, self.figure_colorbar_legend)

        return pn.Row(
            pn.Column(_create_figure),
            pn.Column(
                self.color_map_key_continuous,
                self.color_map_key_discrete,
                self.color_by,
                self.reverse_colormap,
                pn.Row(self.cluster, self.cluster_colorpicker),
                self.gene_names,
                '',
                self.save_title,
                self.save_file_name,
                pn.Row(self.save_file_width, self.save_file_hight),
                self.save_only_in_view,
                # self.with_base_image,
                self.save_button,
                self.save_message,
            )
        )
