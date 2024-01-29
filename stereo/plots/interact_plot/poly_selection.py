import collections
import copy
from typing import Optional

import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import panel as pn
from holoviews.element.selection import spatial_select
from holoviews.util.transform import dim
import tifffile as tiff
import cv2
from shapely.geometry import MultiPoint

from stereo.stereo_config import stereo_conf
from stereo.tools.tools import make_dirs
from stereo.preprocess import filter_genes

pn.extension()
hv.extension('bokeh')

pn.param.ParamMethod.loading_indicator = True
pa = hv.annotate.instance()


class PolySelection(object):
    """
    test
    """

    def __init__(
            self,
            data,
            width: Optional[int] = 500, height: Optional[int] = 500,
            bgcolor='#2F2F4F'
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
            'y': self.data.position[:, 1],
            'count': total_counts
        })
        self.scatter_df.reset_index(inplace=True)
        self.scatter = self._plot()
        self.download = pn.widgets.Button(
            name='export',
            button_type="primary",
            width=100
        )
        self.download_message = pn.widgets.StaticText(width=300, height=20)
        # self.download_message = pn.pane.Markdown('This is a message')
        self.drop_checkbox = pn.widgets.Select(
            # name='method',
            options={'keep selected area': False, 'drop selected area': True},
            width=150
        )
        self.add = pn.widgets.Button(
            name='add',
            button_type="primary",
            width=100
        )
        self.add_message = pn.widgets.StaticText(width=300, height=20)
        self.list_poly_selection_coors = []
        self.download.on_click(self._download_callback)
        self.add.on_click(self._add_selected_area)
        self.selected_exp_data = None
        self.selected_pos = None
        self.figure = self.show()

    # @param.depends(pa.param.annotator)
    def _download_callback(self, _):
        self.download.loading = True
        self.download_message.value = ''
        if len(pa.selected.data):
            selected_point = pa.selected.data[0]
        else:
            # import collections
            # selected_point = collections.OrderedDict({'x': [], 'y': []})
            selected_point = None
            # print('selections not found, please choose a selection area')
            self.download_message.value = '<font color="red"><b>No area is selected, you have to click the selected area beforehand.</b></font>'
        if selected_point is not None:
            selected_point_array = np.array([selected_point['x'], selected_point['y']])
            points = selected_point_array.T
            selection_expr = dim('x', spatial_select, dim('y'), geometry=points)
            selected_pos = hv.Dataset(self.scatter_df).select(selection_expr).data.index
            # if self.selected_pos is None:
            #     self.selected_pos = selected_pos.to_numpy()
            # else:
            #     self.selected_pos = np.union1d(self.selected_pos, selected_pos)
            self.selected_exp_data = self.generate_selected_exp_data(selected_pos, drop=self.drop_checkbox.value)
            self.download_message.value = '<font color="red"><b>Selected area has been exported.</b></font>'
        self.download.loading = False

    def add_selected_area(self):
        """
        push selected area coords to list
        Returns:
        """
        if not pa.selected.data:
            raise Exception("The selected area's data cannot be empty!")

        self.list_poly_selection_coors.append(pa.selected.data)
        print(f"{len(self.list_poly_selection_coors)} areas have been added into queue.")

    def _add_selected_area(self, _):
        """
        push selected area coords to list
        Returns:
        """
        self.add.loading = True
        self.add_message.value = ''
        if not pa.selected.data:
            # raise Exception("The selected area's data cannot be empty!")
            self.add_message.value = '<font color="red"><b>No area is selected, you have to click the selected area beforehand.</b></font>'
        else:
            self.list_poly_selection_coors.append(pa.selected.data)
            area_count = len(self.list_poly_selection_coors)
            if area_count > 1:
                self.add_message.value = f'<font color="red"><b>{area_count} areas have been added.</b></font>'
            else:
                self.add_message.value = f'<font color="red"><b>{area_count} area has been added.</b></font>'
        self.add.loading = False

    # def get_selected_boundary_coors(self) -> list:
    def get_selected_area_coors(self, drop=False) -> list:
        selected_exp_data = self.get_selected_areas(drop)
        if selected_exp_data is not None:
            return selected_exp_data.position.tolist()
        return []
    
    def get_selected_areas(self, drop=False):
        """
        get selected area exp boundary coords, list contains each x,y
        Returns:

        """
        if len(self.list_poly_selection_coors) == 0:
            raise Exception('Please select the data area in the picture first!')

        print("processing selected {} area".format(len(self.list_poly_selection_coors)))
        # list_poly_selection_exp_coors = []
        # data_set = set()
        selected_pos = None
        for each_polygon in self.list_poly_selection_coors:
            if len(each_polygon):
                selected_point = each_polygon[0]
            else:
                selected_point = collections.OrderedDict({'x': [], 'y': []})
                print('selections not found, please choose a selection area')
                continue

            selected_point_array = np.array([selected_point['x'], selected_point['y']])
            points = selected_point_array.T
            selection_expr = dim('x', spatial_select, dim('y'), geometry=points)
            if selected_pos is None:
                selected_pos = hv.Dataset(self.scatter_df).select(selection_expr).data.index.to_numpy()
            else:
                selected_pos = np.union1d(
                    selected_pos, 
                    hv.Dataset(self.scatter_df).select(selection_expr).data.index.to_numpy()
                )
        if selected_pos is not None:
            return self.generate_selected_exp_data(selected_pos, drop=drop)
        #     data_temp = copy.deepcopy(self.data)
        #     self.selected_exp_data = data_temp.sub_by_index(cell_index=selected_pos)
        #     # self.selected_exp_data = self.selected_exp_data.tl.filter_genes(mean_umi_gt=0)
        #     self.selected_exp_data = filter_genes(self.selected_exp_data, mean_umi_gt=0)
        #     list_poly_selection_exp_coors = self.selected_exp_data.position.tolist()
        #     if self.selected_exp_data.shape[0] * self.selected_exp_data.shape[1] == 0:
        #         return []
        #     if self.selected_exp_data.shape == self.data.shape:
        #         raise Exception('Please select the data area in the picture first!')
        # return list_poly_selection_exp_coors
        return None

    def export_high_res_area(self, origin_file_path: str, output_path: str, drop: bool = False) -> str:
        """
        export selected area in high resolution
        Args:
            origin_file_path: origin file path which you read
            output_path: location the high res file storaged
        Returns:
            output_path
        """
        coors = self.get_selected_area_coors(drop)
        # print('coors length: %s' % len(coors))
        if not coors or len(coors) == 0:
            raise Exception('Please select the data area in the picture first!')

        make_dirs(output_path)
        from gefpy.cgef_adjust_cy import CgefAdjust
        cg = CgefAdjust()
        if self.data.bin_type == 'cell_bins':
            cg.generate_cgef_by_coordinate(origin_file_path, output_path, coors)
        else:
            cg.generate_bgef_by_coordinate(origin_file_path, output_path, coors, self.data.bin_size)

        return output_path

    def export_roi_image(self, origin_file_path: str, output_path: str, drop: bool = False):
        if len(self.list_poly_selection_coors) == 0:
            raise Exception('Please select the data area in the picture first!')
        
        # origin_image_data = tiff.imread(origin_file_path)
        with tiff.TiffFile(origin_file_path) as tif:
            origin_image_data = tif.asarray()
            vmin, vmax = np.min(origin_image_data), np.max(origin_image_data)
            if not drop:
                mask = np.zeros_like(origin_image_data)
            else:
                mask = np.ones_like(origin_image_data)
            if self.data.bin_type == 'bins':
                coors = self.get_selected_area_coors()
                if not coors or len(coors) == 0:
                    raise Exception('Please select the data area in the picture first!')
                coors = np.array(coors, dtype=np.int64)
                for x, y in coors:
                    x_end, y_end = x + self.data.bin_size, y + self.data.bin_size
                    mask[y:y_end, x:x_end] = 1 if not drop else 0
                new_image_data = origin_image_data * mask
            else:
                polygons = []
                for each_polygon in self.list_poly_selection_coors:
                    if len(each_polygon) == 0:
                        continue
                    selected_point = each_polygon[0]
                    points = [point for point in zip(selected_point['x'], selected_point['y'])]
                    mlp = MultiPoint(points)
                    one_polygon = []
                    for x, y in points:
                        x = int(x) if x < mlp.centroid.x else int(x) + 1
                        y = int(y) if y < mlp.centroid.y else int(y) + 1
                        one_polygon.append([x, y])
                    one_polygon = np.array(one_polygon, dtype=np.int64)
                    polygons.append(one_polygon)
                if len(polygons) == 0:
                    raise Exception('Please select the data area in the plot first!')
                if not drop:
                    if len(mask.shape) == 2:
                        color = 1
                    else:
                        color = [1, 1, 1]
                    mask = cv2.fillPoly(mask, polygons, color)
                else:
                    if len(mask.shape) == 2:
                        color = 0
                    else:
                        color = [0, 0, 0]
                    mask = cv2.fillPoly(mask, polygons, color)
                new_image_data = origin_image_data * mask
            shaped_metadata = tif.shaped_metadata
            metadata = None
            if shaped_metadata is not None:
                metadata = shaped_metadata[0]
                if 'value_range' not in metadata:
                    metadata['value_range'] = (int(vmin), int(vmax))
            tiff.imwrite(output_path, new_image_data, bigtiff=True, metadata=metadata)


    def generate_selected_exp_data(self, selected_pos, drop=False):
        import copy
        if selected_pos is not None:
            selected_index = self.scatter_df.index.drop(selected_pos) if drop else selected_pos
            data_temp = copy.deepcopy(self.data)
            selected_exp_data = data_temp.sub_by_index(cell_index=selected_index)
            # self.selected_exp_data = self.selected_exp_data.tl.filter_genes(mean_umi_gt=0)
            selected_exp_data = filter_genes(selected_exp_data, mean_umi_gt=0)
        else:
            selected_exp_data = None
        return selected_exp_data

    def _plot(self):
        scatter = self.scatter_df.hvplot.scatter(
            x='x', y='y', c='count', cnorm='eq_hist',
            cmap=stereo_conf.linear_colors('stereo'),
            width=self.width, height=self.height,
            padding=(0.1, 0.1),
            dynspread=True,
            datashade=True,
        ).opts(
            bgcolor=self.bgcolor,
            xaxis=None,
            yaxis=None,
            aspect='equal',
            invert_yaxis=True
        )
        return scatter

    def add_annotator(self):
        poly_layout = pa(
            hv.Polygons([]).opts(alpha=0.5),
            show_vertices=True,
            annotations=['index'],
            vertex_style={'nonselection_alpha': 1, 'size': 10, 'color': 'blue'},
            table_opts={'editable': False, 'width': 300}
        )
        poly_scatter, anno_table = hv.annotate.compose(self.scatter, poly_layout)
        return poly_scatter, anno_table

    def show(self):
        poly_scatter, anno_table = self.add_annotator()
        self.figure = pn.Row(
            poly_scatter,
            pn.Column(
                anno_table.Table.PolyAnnotator_Vertices,
                '',
                pn.Row(self.download, self.drop_checkbox),
                self.download_message,
                '',
                self.add,
                self.add_message
            ))
        return self.figure
