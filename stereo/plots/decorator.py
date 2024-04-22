from functools import partial
from functools import wraps

import panel as pn
from matplotlib.figure import Figure

from stereo.utils.data_helper import reorganize_data_coordinates


def plot_scale(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        from stereo.plots.plot_collection import PlotCollection
        from stereo.plots.plot_base import PlotBase
        pc_object = args[0]
        data = None
        if isinstance(pc_object, PlotCollection):
            data = pc_object.data
        elif isinstance(pc_object, PlotBase):
            data = pc_object.stereo_exp_data
        if data:
            if (data.attr is None) or (data.resolution is None) or (data.resolution <= 0) or (data.bin_size is None) \
                    or (data.bin_size <= 0):
                kwargs['show_plotting_scale'] = False
            else:
                kwargs.setdefault('show_plotting_scale', True)
                data_resolution = data.resolution if data.bin_type == 'cell_bins' else data.resolution * data.bin_size
                data_bin_offset = 1 if data.bin_type == 'cell_bins' else data.bin_size
                kwargs.setdefault('data_resolution', data_resolution)
                kwargs.setdefault('data_bin_offset', data_bin_offset)
        return func(*args, **kwargs)

    return wrapped


def download(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        out_path = None
        dpi = 100
        if 'out_path' in kwargs:
            out_path = kwargs['out_path']
            del kwargs['out_path']
        if 'out_dpi' in kwargs:
            dpi = kwargs['out_dpi']
            del kwargs['out_dpi']
        fig: Figure = func(*args, **kwargs)
        if type(fig) is not Figure:
            return fig
        if out_path is None:
            pn.extension()
            file_name_input = pn.widgets.TextInput(name='file name', placeholder='Enter a file name...', width=200)
            format_select = pn.widgets.Select(name='file format', value='png', options=['png', 'pdf'], width=60)
            dpi_input = pn.widgets.IntInput(name='dpi', placeholder='Enter the dip...', width=200, value=100, step=1,
                                            start=0)
            export_button = pn.widgets.Button(name='export', button_type="primary", width=100)
            static_text = pn.widgets.StaticText(width=800)

            def _action(_, figure: Figure):
                export_button.loading = True
                static_text.value = ""
                try:
                    out_path = file_name_input.value
                    file_format = format_select.value
                    dpi = dpi_input.value if dpi_input.value > 0 else 100
                    if out_path is not None and len(out_path) > 0:
                        out_path = f"{out_path}_{func.__name__}.{file_format}"
                        figure.savefig(out_path, bbox_inches='tight', dpi=dpi)
                        static_text.value = f'the plot has already been saved in the same directory as this notebook ' \
                                            f'and named as <font color="red"><b>{out_path}</b></font>'
                finally:
                    export_button.loading = False

            action = partial(_action, figure=fig)
            export_button.on_click(action)
            return pn.Column(
                '<font size="3"><br>Exporting the plot.</br></font>',
                pn.Row(file_name_input, format_select, dpi_input),
                pn.Row(export_button, static_text)
            )
        else:
            fig.savefig(out_path, bbox_inches='tight', dpi=dpi)

    return wrapped


def download_only(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        out_path = None
        dpi = 100
        if 'out_path' in kwargs:
            out_path = kwargs['out_path']
            del kwargs['out_path']
        if 'out_dpi' in kwargs:
            dpi = kwargs['out_dpi']
            del kwargs['out_dpi']
        fig: Figure = func(*args, **kwargs)
        if type(fig) is Figure and out_path is not None:
            fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
        return fig

    return wrapped


def reorganize_coordinate(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        from stereo.plots.plot_collection import PlotCollection
        from stereo.plots.plot_base import PlotBase
        pc_object = args[0]
        data = None
        if isinstance(pc_object, PlotCollection):
            data = pc_object.data
        elif isinstance(pc_object, PlotBase):
            data = pc_object.stereo_exp_data
        if data is not None and data.merged:
            reorganize_coordinate = False
            horizontal_offset_additional = 0
            vertical_offset_additional = 0
            if 'reorganize_coordinate' in kwargs:
                reorganize_coordinate = kwargs['reorganize_coordinate']
                del kwargs['reorganize_coordinate']
            if 'horizontal_offset_additional' in kwargs:
                horizontal_offset_additional = kwargs['horizontal_offset_additional']
                del kwargs['horizontal_offset_additional']
                if horizontal_offset_additional < 0:
                    horizontal_offset_additional = 0
            if 'vertical_offset_additional' in kwargs:
                vertical_offset_additional = kwargs['vertical_offset_additional']
                del kwargs['vertical_offset_additional']
                if vertical_offset_additional < 0:
                    vertical_offset_additional = 0
            if reorganize_coordinate:
                data.position, data.position_offset, data.position_min = \
                    reorganize_data_coordinates(
                        data.cells.batch, data.position, data.position_offset, data.position_min,
                        reorganize_coordinate, horizontal_offset_additional, vertical_offset_additional
                    )
        res = func(*args, **kwargs)
        data.reset_position()
        return res

    return wrapped

# def reorganize_coordinate(func):
#     @wraps(func)
#     def wrapped(*args, **kwargs):
#         from stereo.plots.plot_collection import PlotCollection
#         from stereo.plots.plot_base import PlotBase
#         pc_object = args[0]
#         data = None
#         if isinstance(pc_object, PlotCollection):
#             data = pc_object.data
#         elif isinstance(pc_object, PlotBase):
#             data = pc_object.stereo_exp_data
#         if data is not None and data.merged:
#             reorganize_coordinate = False
#             horizontal_offset_additional = 0
#             vertical_offset_additional = 0
#             if 'reorganize_coordinate' in kwargs:
#                 reorganize_coordinate = kwargs['reorganize_coordinate']
#                 del kwargs['reorganize_coordinate']
#             if 'horizontal_offset_additional' in kwargs:
#                 horizontal_offset_additional = kwargs['horizontal_offset_additional']
#                 del kwargs['horizontal_offset_additional']
#                 if horizontal_offset_additional < 0:
#                     horizontal_offset_additional = 0
#             if 'vertical_offset_additional' in kwargs:
#                 vertical_offset_additional = kwargs['vertical_offset_additional']
#                 del kwargs['vertical_offset_additional']
#                 if vertical_offset_additional < 0:
#                     vertical_offset_additional = 0
#             if reorganize_coordinate:
#                 batches = natsorted(np.unique(data.cells.batch))
#                 data_count = len(batches)
#                 position_row_count = ceil(data_count / reorganize_coordinate)
#                 position_column_count = reorganize_coordinate
#                 max_xs = [0] * (position_column_count + 1)
#                 max_ys = [0] * (position_row_count + 1)
#                 for i, bno in enumerate(batches):
#                     idx = np.where(data.cells.batch == bno)[0]
#                     data.position[idx] -= data.position_offset[bno] if data.position_offset is not None else 0
#                     position_row_number = i // reorganize_coordinate
#                     position_column_number = i % reorganize_coordinate
#                     max_x = data.position[idx][:, 0].max()
#                     max_y = data.position[idx][:, 1].max()
#                     if max_x > max_xs[position_column_number + 1]:
#                         max_xs[position_column_number + 1] = max_x
#                     if max_y > max_ys[position_row_number + 1]:
#                         max_ys[position_row_number + 1] = max_y

#                 data.position_offset = {}
#                 for i, bno in enumerate(batches):
#                     idx = np.where(data.cells.batch == bno)[0]
#                     position_row_number = i // reorganize_coordinate
#                     position_column_number = i % reorganize_coordinate
#                     x_add = max_xs[position_column_number]
#                     y_add = max_ys[position_row_number]
#                     if position_column_number > 0:
#                         x_add += sum(max_xs[0:position_column_number]) + horizontal_offset_additional * position_column_number  # noqa
#                     if position_row_number > 0:
#                         y_add += sum(max_ys[0:position_row_number]) + vertical_offset_additional * position_row_number
#                     # position_offset = np.repeat([[x_add, y_add]], repeats=len(idx), axis=0).astype(np.uint32)
#                     position_offset = np.array([x_add, y_add], dtype=data.position.dtype)
#                     data.position[idx] += position_offset
#                     data.position_offset[bno] = position_offset
#         return func(*args, **kwargs)
#     return wrapped
