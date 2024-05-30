from typing import Literal, Optional

# from stereo.algorithm.st_gears.visual import RegisPlotter
# from stereo.constant import PLOT_SCATTER_SIZE_FACTOR

from .ms_plot_base import MSDataPlotBase


class PlotStGears(MSDataPlotBase):
    
    def __init__(self, ms_data, pipeline_res=None):
        super().__init__(ms_data, pipeline_res)
        from stereo.algorithm.st_gears.visual import RegisPlotter
        self.__plotter = RegisPlotter(num_cols=3, dpi_val=100)
        self.__scatter_size_factor = 11000
    

    def scatter_for_st_gears(
        self,
        ctype: Literal['cell_label', 'num_anchors', 'pi_max', 'strong_anch', 'weight', 'sum_gene'] = 'cell_label',
        lay_type: Literal['to_pre', 'to_next'] = 'to_pre',
        spatial_key: str = 'spatial',
        width: Optional[float] = None,
        height: Optional[float] = None,
        cols: Optional[int] = None,
        dot_size: Optional[float] = None,
        invert_y: Optional[bool] = True,
    ):
        """_summary_

        :param ctype: _description_, defaults to 'cell_label'
        :param lay_type: _description_, defaults to 'to_pre'
        :param spatial_key: _description_, defaults to 'spatial'
        :param width: _description_, defaults to None
        :param height: _description_, defaults to None
        :param cols: _description_, defaults to None
        :param dot_size: _description_, defaults to None
        :return: _description_
        """
        anndata_list = [self.ms_data.data_list[i].adata for i in self.pipeline_res['st_gears']['regis_ilist']]
        if cols is not None:
            self.__plotter.num_cols = cols
        if dot_size is None:
            mean_dot_count = sum([adata.shape[0] for adata in anndata_list]) / len(anndata_list)
            dot_size = self.__scatter_size_factor / mean_dot_count
        nrows, ncols = self.__plotter._define_cells(len(anndata_list))
        if width is None:
            width = ncols * 5
        if height is None:
            height = nrows * 5
        return self.__plotter.plot_scatter_by_grid(
            slicesl=anndata_list,
            anncell_cid=self.pipeline_res['st_gears']['anncell_cid'],
            ali=self.pipeline_res['st_gears']['ali'],
            bli=self.pipeline_res['st_gears']['bli'],
            pili=self.pipeline_res['st_gears']['pili'],
            tyscoreli=self.pipeline_res['st_gears']['tyscoreli'],
            alphali=self.pipeline_res['st_gears']['alphali'],
            figsize=(width, height),
            lay_type=lay_type,
            ctype=ctype,
            spatype=spatial_key,
            filter_by_label=self.pipeline_res['st_gears']['parametres']['filter_by_label'],
            label_col=self.pipeline_res['st_gears']['parametres']['cluster_res_key'],
            sdir=None,
            size=dot_size,
            invert_y=invert_y
        )

