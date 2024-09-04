import os
from joblib import (
    Parallel,
    delayed,
    cpu_count
)

import numpy as np

from stereo.log_manager import logger
from stereo.core import StPipeline
from stereo.core.result import MSDataPipeLineResult
from stereo.plots.decorator import download, download_only


class _scope_slice(object):

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer, str, np.str_)):
            return [item]
        else:
            return item


class MSDataPipeLine(object):
    ATTR_NAME = 'tl'
    BASE_CLASS = StPipeline

    def __init__(self, _ms_data):
        super().__init__()
        self.ms_data = _ms_data
        self._result = MSDataPipeLineResult(self.ms_data)
        self._result_keys = dict()
        self._key_record = dict()
        self.__mode = "integrate"
        self.__scope = slice(None)

    @property
    def result(self):
        return self._result

    # @result.setter
    # def result(self, new_result):
    #     self._result = new_result

    @property
    def key_record(self):
        return self._key_record

    @key_record.setter
    def key_record(self, key_record):
        self._key_record = key_record
    
    @property
    def result_keys(self):
        return self._result_keys
    
    @result_keys.setter
    def result_keys(self, result_keys):
        self._result_keys = self._reset_result_keys(result_keys)
    
    @property
    def mode(self):
        return self.__mode
    
    @mode.setter
    def mode(self, mode):
        self.__mode = mode
    
    @property
    def scope(self):    
        return self.__scope
    
    @scope.setter
    def scope(self, scope):
        self.__scope = scope
    
    def _reset_result_keys(self, origin_result_keys: dict = None):
        result_keys = {}
        for scope_key, scope_result_keys in origin_result_keys.items():
            result_keys[scope_key] = []
            for rk in scope_result_keys:
                if rk in self.result[scope_key]:
                    result_keys[scope_key].append(rk)
        return result_keys

    def _use_integrate_method(self, item, *args, **kwargs):
        if "mode" in kwargs:
            del kwargs["mode"]

        scope = kwargs.get("scope", slice(None))
        del kwargs["scope"]

        # if item in {"cal_qc", "filter_cells", "filter_genes", "sctransform", "log1p", "normalize_total",
        #             "scale", "raw_checkpoint", "batches_integrate"}:
        #     if scope != slice(None):
        #         raise Exception(f'{item} use integrate should use all sample')
        #     ms_data_view = self.ms_data
        # elif scope == slice(None):
        if len(self.ms_data[scope]) == len(self.ms_data):
            ms_data_view = self.ms_data
            if ms_data_view.merged_data is None:
                ms_data_view.integrate()
        else:
            ms_data_view = self.ms_data[scope]

        scope_key = self.ms_data.generate_scope_key(ms_data_view._names)
        self.ms_data.scopes_data[scope_key] = ms_data_view.merged_data

        # def set_result_key_method(key):
        #     self.result_keys.setdefault(scope_key, [])
        #     if key in self.result_keys[scope_key]:
        #         self.result_keys[scope_key].remove(key)
        #     self.result_keys[scope_key].append(key)
        
        # ms_data_view.merged_data.tl.result.set_result_key_method = set_result_key_method

        new_attr = self.__class__.BASE_CLASS.__dict__.get(item, None)
        if new_attr is None:
            if self.__class__.ATTR_NAME == "tl":
                from stereo.algorithm.algorithm_base import AlgorithmBase
                merged_data = ms_data_view.merged_data
                new_attr = AlgorithmBase.get_attribute_helper(item, merged_data, merged_data.tl.result)
                if new_attr:
                    logger.info(f'register algorithm {item} to {type(merged_data)}-{id(merged_data)}')
                    return new_attr(*args, **kwargs)
            else:
                from stereo.plots.plot_base import PlotBase
                merged_data = ms_data_view.merged_data
                new_attr = download(PlotBase.get_attribute_helper(item, merged_data, merged_data.tl.result))
                if new_attr:
                    logger.info(f'register plot_func {item} to {type(merged_data)}-{id(merged_data)}')
                    return new_attr(*args, **kwargs)

        logger.info(f'data_obj(idx=0) in ms_data start to run {item}')
        return new_attr(
            ms_data_view.merged_data.__getattribute__(self.__class__.ATTR_NAME),
            *args,
            **kwargs
        )

    def _run_isolated_method(self, item, *args, **kwargs):
        if "mode" in kwargs:
            del kwargs["mode"]
        ms_data_view = self.ms_data[kwargs["scope"]]
        if "scope" in kwargs:
            del kwargs["scope"]

        new_attr = self.__class__.BASE_CLASS.__dict__.get(item, None)
        if self.__class__.ATTR_NAME == 'tl':
            n_jobs = min(len(ms_data_view.data_list), cpu_count())
        else:
            n_jobs = 1
        if new_attr:
            def log_delayed_task(idx, *arg, **kwargs):
                logger.info(f'data_obj(idx={idx}) in ms_data start to run {item}')
                if self.__class__.ATTR_NAME == 'plt':
                    out_path = kwargs.get('out_path', None)
                    if out_path is not None:
                        path_name, ext = os.path.splitext(out_path)
                        kwargs['out_path'] = f'{path_name}_{idx}{ext}'
                new_attr(*arg, **kwargs)

            Parallel(n_jobs=n_jobs, backend='threading', verbose=100)(
                delayed(log_delayed_task)(idx, obj.__getattribute__(self.__class__.ATTR_NAME), *args, **kwargs)
                for idx, obj in enumerate(ms_data_view.data_list)
            )
        else:
            if self.__class__.ATTR_NAME == 'tl':
                from stereo.algorithm.algorithm_base import AlgorithmBase
                base = AlgorithmBase
            else:
                from stereo.plots.plot_base import PlotBase
                base = PlotBase

            def log_delayed_task(idx, obj, *arg, **kwargs):
                logger.info(f'data_obj(idx={idx}) in ms_data start to run {item}')
                new_attr = base.get_attribute_helper(item, obj, obj.tl.result)
                if base == PlotBase:
                    out_path = kwargs.get('out_path', None)
                    if out_path is not None:
                        path_name, ext = os.path.splitext(out_path)
                        kwargs['out_path'] = f'{path_name}_{idx}{ext}'
                    new_attr = download_only(new_attr)
                if new_attr:
                    new_attr(*arg, **kwargs)
                else:
                    raise Exception

            Parallel(n_jobs=n_jobs, backend='threading', verbose=100)(
                delayed(log_delayed_task)(idx, obj, *args, **kwargs)
                for idx, obj in enumerate(ms_data_view.data_list)
            )

    def __getattr__(self, item):
        dict_attr = self.__dict__.get(item, None)
        if dict_attr:
            return dict_attr

        # start with __ may not be our algorithm function, and will cause import problem
        if item.startswith('__'):
            raise AttributeError

        if self.__class__.ATTR_NAME == 'tl':
            from stereo.algorithm.ms_algorithm_base import MSDataAlgorithmBase
            run_method = MSDataAlgorithmBase.get_attribute_helper(item, self.ms_data, self.result)
            if run_method:
                return run_method
        elif self.__class__.ATTR_NAME == 'plt':
            from stereo.plots.ms_plot_base import MSDataPlotBase
            run_method = MSDataPlotBase.get_attribute_helper(item, self.ms_data, self.ms_data.tl.result)
            if run_method:
                return download(run_method)

        def temp(*args, **kwargs):
            if "scope" not in kwargs:
                # kwargs["scope"] = slice_generator[:]
                kwargs["scope"] = self.__scope
            if "mode" not in kwargs:
                kwargs["mode"] = self.__mode

            if kwargs["mode"] == "integrate":
                return self._use_integrate_method(item, *args, **kwargs)
            elif kwargs["mode"] == "isolated":
                self._run_isolated_method(item, *args, **kwargs)
            else:
                raise Exception("`mode` should be one of [`integrate`, `isolated`]")

        return temp
    
    def set_scope_and_mode(
        self,
        scope: slice = slice(None),
        mode: str = "integrate"
    ):
        """
        Set the `scope` and `mode` globally for Multi-slice analysis.

        :param scope: the scope, defaults to slice(None)
        :param mode: the mode, defaults to "integrate"
        """
        assert mode in ("integrate", "isolated"), 'mode should be one of [`integrate`, `isolated`]'
        self.__mode = mode
        self.__scope = scope
        if self.__class__.ATTR_NAME == 'tl':
            self.ms_data.plt.scope = scope
            self.ms_data.plt.mode = mode
        elif self.__class__.ATTR_NAME == 'plt':
            self.ms_data.tl.scope = scope
            self.ms_data.tl.mode = mode
        else:
            pass


slice_generator = _scope_slice()
