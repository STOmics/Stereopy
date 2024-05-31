from IPython.display import HTML
from IPython.display import display

from stereo.plots.plot_base import PlotBase


class ShowBatchQcReport(PlotBase):
    def show_batch_qc_report(self, res_key='batch_qc'):
        if res_key not in self.pipeline_res:
            raise ValueError(f"The result specified by {res_key} is not exists.")

        report_path = self.pipeline_res[res_key]['report_path']
        with open(report_path, 'r') as fp:
            content = fp.read()
        display(HTML(content))
