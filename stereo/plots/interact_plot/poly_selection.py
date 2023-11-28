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

from stereo.stereo_config import stereo_conf
from stereo.tools.tools import make_dirs

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
            'y': self.data.position[:, 1] * -1,
            'count': total_counts
        })
        self.scatter_df.reset_index(inplace=True)
        self.scatter = self._plot()
        self.download = pn.widgets.Button(
            name='export',
            button_type="primary",
            width=100
        )
        self.drop_checkbox = pn.widgets.Select(
            name='method',
            options={'keep selected point': False, 'drop selected point': True},
            width=150
        )
        self.list_poly_selection_coors = []
        self.download.on_click(self._download_callback)
        self.selected_exp_data = None
        self.figure = self.show()

    # @param.depends(pa.param.annotator)
    def _download_callback(self, _):
        self.download.loading = True
        if len(pa.selected.data):
            selected_point = pa.selected.data[0]
        else:
            import collections
            selected_point = collections.OrderedDict({'x': [], 'y': []})
            print('selections not found, please choose a selection area')
        selected_point_array = np.array([selected_point['x'], selected_point['y']])
        points = selected_point_array.T
        selection_expr = dim('x', spatial_select, dim('y'), geometry=points)
        selected_pos = hv.Dataset(self.scatter_df).select(selection_expr).data.index
        self.generate_selected_expr_matrix(selected_pos, drop=self.drop_checkbox.value)
        self.download.loading = False

    def add_selected_area(self):
        """
        push selected area coords to list
        Returns:
        """
        if not pa.selected.data:
            raise Exception("The selected area's data cannot be empty!")

        self.list_poly_selection_coors.append(pa.selected.data)

    def get_selected_boundary_coors(self) -> list:
        """
        get selected area exp boundary coords, list contains each x,y
        Returns:

        """
        if not self.selected_exp_data or self.selected_exp_data.shape == self.data.shape:
            raise Exception('Please select the data area in the picture first!')

        print("processing selected {} area".format(len(self.list_poly_selection_coors)))
        list_poly_selection_exp_coors = []
        data_set = set()
        for each_polygon in self.list_poly_selection_coors:
            if len(each_polygon):
                selected_point = each_polygon[0]
            else:
                selected_point = collections.OrderedDict({'x': [], 'y': []})
                print('selections not found, please choose a selection area')

            selected_point_array = np.array([selected_point['x'], selected_point['y']])
            points = selected_point_array.T
            selection_expr = dim('x', spatial_select, dim('y'), geometry=points)
            selected_pos = hv.Dataset(self.scatter_df).select(selection_expr).data.index
            data_temp = copy.deepcopy(self.data)
            self.selected_exp_data = data_temp.sub_by_index(cell_index=selected_pos)
            self.selected_exp_data = self.selected_exp_data.tl.filter_genes(mean_umi_gt=0)
            exp_matrix_data = self.selected_exp_data.position.tolist()
            if self.selected_exp_data.shape[0] * self.selected_exp_data.shape[1] == 0:
                return []
            for label in exp_matrix_data:
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

    def generate_selected_expr_matrix(self, selected_pos, drop=False):
        import copy
        if selected_pos is not None:
            selected_index = self.scatter_df.index.drop(selected_pos) if drop else selected_pos
            data_temp = copy.deepcopy(self.data)
            self.selected_exp_data = data_temp.sub_by_index(cell_index=selected_index)
            self.selected_exp_data = self.selected_exp_data.tl.filter_genes(mean_umi_gt=0)
        else:
            self.selected_exp_data = None

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
                '',
                pn.Row(self.drop_checkbox, self.download)
            ))
        return self.figure
