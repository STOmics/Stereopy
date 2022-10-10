import collections
import copy

import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
import panel as pn
import param
from holoviews.element.selection import spatial_select
from holoviews.util.transform import dim
from stereo.config import StereoConfig
from typing import Optional
from stereo.tools.boundary import ConcaveHull

pn.extension()
hv.extension('bokeh')

conf = StereoConfig()

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
        # self.poly_annotate = hv.annotate.instance()
        self.scatter_df = pd.DataFrame({
            # 'cell': self.data.cell_names,
            'x': self.data.position[:, 0],
            'y': self.data.position[:, 1] * -1,
            # 'count': np.array(self.data.exp_matrix.sum(axis=1))[:, 0],
            'count': np.array(self.data.exp_matrix.sum(axis=1))[:,
                                                                0] if self.data.cells.total_counts is None else self.data.cells.total_counts
        }).reset_index()
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
        self.list_poly_selection_exp_coors = []
        self.download.on_click(self._download_callback)
        self.selected_exp_data = None
        self.figure = self.show()

    # @param.depends(pa.param.annotator)
    def _download_callback(self, _):
        self.download.loading = True
        # sio = io.StringIO()
        # selected_pos = hv.Dataset(self.scatter_df).select(link.selection_expr).data[['cell']].values
        # print(pa.selected.data)
        if len(pa.selected.data):
            selected_point = pa.selected.data[0]
        else:
            import collections
            from stereo.log_manager import logger
            selected_point = collections.OrderedDict({'x': [], 'y': []})
            print('selections not found, please choose a selection area')
        selected_point_array = np.array([selected_point['x'], selected_point['y']])
        points = selected_point_array.T
        selection_expr = dim('x', spatial_select, dim('y'), geometry=points)
        # print(selection_expr)
        selected_pos = hv.Dataset(self.scatter_df).select(selection_expr).data.index
        self.generate_selected_expr_matrix(selected_pos, drop=self.drop_checkbox.value)
        # logger.info(f'generate a new StereoExpData')

        self.download.loading = False

    def add_selected_area(self):
        """
        push selected area coords to list
        Returns:
        """
        self.list_poly_selection_coors.append(pa.selected.data)

    def get_selected_boundary_coors(self) -> list:
        """
        get selected area exp boundary coords, list contains each x,y
        Returns:

        """
        print("processing selected {} area".format(len(self.list_poly_selection_coors)))
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
            exp_matrix_data = self.selected_exp_data.position.tolist()
            init = ConcaveHull(exp_matrix_data, 3)
            concave_hull = init.calculate().tolist()
            concave_hull = [int(i) for k in concave_hull for i in k]
            self.list_poly_selection_exp_coors.append(concave_hull)
        return self.list_poly_selection_exp_coors

    def export_high_res_area(self, origin_file_path: str, output_path: str) -> str:
        """
        export selected area in high resolution
        Args:
            origin_file_path: origin file path which you read
            next: location the high res file storaged

        Returns:
            output_path
        """
        coors = self.get_selected_boundary_coors()
        from gefpy.cgef_adjust_cy import CgefAdjust
        cg = CgefAdjust()
        cg.create_Region_Bgef(origin_file_path, output_path, coors)
        return output_path

    def generate_selected_expr_matrix(self, selected_pos, drop=False):
        import copy
        if selected_pos is not None:
            # selected_index = np.isin(self.data.cell_names, selected_pos)
            selected_index = self.scatter_df.index.drop(selected_pos) if drop else selected_pos
            data_temp = copy.deepcopy(self.data)
            self.selected_exp_data = data_temp.sub_by_index(cell_index=selected_index)
        else:
            self.selected_exp_data = None

    def _plot(self):
        scatter = self.scatter_df.hvplot.scatter(
            x='x', y='y', c='count', cnorm='eq_hist',
            cmap=conf.linear_colors('stereo'),
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
        # return pn.Row(poly_scatter, pn.Column(anno_table, self.download))
        return self.figure
