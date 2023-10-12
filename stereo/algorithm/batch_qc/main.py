from multiprocessing import cpu_count
from typing import Union

from stereo.algorithm.algorithm_base import AlgorithmBase
from .batchqc_raw import batchqc_raw


class BatchQc(AlgorithmBase):
    def main(
            self,
            n_neighbors: int = 100,
            condition: Union[str, list, None] = None,
            count_key: str = "total_counts",
            cluster_res_key: Union[str, None] = None,
            report_path: str = "./batch_qc",
            gpu: Union[str, int] = "0",
            data_loader_num_workers: int = 10,
            num_threads: int = -1,
            res_key: str = 'batch_qc'
    ):
        """_summary_

        :param n_neighbors: Calculate the nearest neighbors of a local area, defaults to 100.
        :param condition: Label the experimental conditions. By default, the experimental conditions for each data are different, defaults to None.
        :param count_key: total_counts or n_genes_by_counts, defaults to "total_counts".
        :param cluster_res_key: The key which specifies the clustering result in data.tl.result, defaults to None.
        :param report_path: The path to save the reports of result, defaults to "./batch_qc".
        :param gpu: The gpu on which running this function, defaults to "0", it will run on cpu automatically if the machine doesn't have gpu.
        :param res_key: Set a key to store the result to data.tl.result, defaults to 'batch_qc'.
        :param data_loader_num_workers: 'int',  will create `data_loader_num_workers` num of multiprocessing to work.
        :param num_threads: 'int',  will create `num_threads` num of threads to work.
        """  # noqa
        if num_threads <= 0 or num_threads > cpu_count():
            num_threads = cpu_count()

        if data_loader_num_workers <= 0 or data_loader_num_workers > cpu_count():
            data_loader_num_workers = cpu_count()

        self.pipeline_res[res_key] = batchqc_raw(
            self.stereo_exp_data,
            n_neighbors=n_neighbors,
            condition=condition,
            count_key=count_key,
            celltype_key=cluster_res_key,
            report_path=report_path,
            gpu=gpu,
            data_loader_num_workers=data_loader_num_workers,
            num_threads=num_threads
        )
