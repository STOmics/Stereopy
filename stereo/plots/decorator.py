from functools import wraps, partial
from matplotlib.figure import Figure
import panel as pn

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
            if (data.attr is None) or \
                ('resolution' not in data.attr) or (data.attr['resolution'] <= 0):
                kwargs['show_plotting_scale'] = False
            else:
                kwargs.setdefault('show_plotting_scale', True)
                data_resolution = data.attr['resolution'] if data.bin_type == 'cell_bins' else data.attr['resolution'] * data.bin_size
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
        if fig is None:
            return None
        if out_path is None:
            pn.extension()
            file_name_input = pn.widgets.TextInput(name='file name', placeholder='Enter a file name...', width=200)
            dpi_input = pn.widgets.IntInput(name='dpi', placeholder='Enter the dip...', width=200, value=100, step=1, start=0)
            export_button = pn.widgets.Button(name='export', button_type="primary", width=100)
            static_text = pn.widgets.StaticText(width=800)
            def _action(_, figure: Figure):
                export_button.loading = True
                static_text.value = ""
                try:
                    out_path = file_name_input.value
                    dpi = dpi_input.value if dpi_input.value > 0 else 100
                    if out_path is not None and len(out_path) > 0:
                        out_path = f"{out_path}_{func.__name__}.png"
                        figure.savefig(out_path, bbox_inches='tight', dpi=dpi)
                        static_text.value = f'the plot has alrady been saved in the same directory as this notebook and named as <font color="red"><b>{out_path}</b></font>'
                finally:
                    export_button.loading = False
            action = partial(_action, figure=fig)
            export_button.on_click(action)
            return pn.Column(
                '<font size="3"><br>Exporting the plot.</br></font>',
                pn.Row(file_name_input, dpi_input),
                pn.Row(export_button, static_text)
            )
        else:
            fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
    return wrapped