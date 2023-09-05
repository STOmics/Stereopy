from joblib import Parallel, delayed, cpu_count

from stereo import logger
from stereo.core import StPipeline
from stereo.plots.decorator import download


class _scope_slice(object):

    def __getitem__(self, item):
        if type(item) is slice:
            return item


class MSDataPipeLine(object):
    ATTR_NAME = 'tl'
    BASE_CLASS = StPipeline

    def __init__(self, _ms_data):
        super().__init__()
        self.ms_data = _ms_data
        self._result = dict()

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, new_result):
        self._result = new_result

    def _use_integrate_method(self, item, *args, **kwargs):
        # if item == "batches_integrate":
        #     raise AttributeError
        
        if "mode" in kwargs:
            del kwargs["mode"]

        scope = kwargs.get("scope", slice(None))
        del kwargs["scope"]

        if item in {"cal_qc", "filter_cells", "filter_genes", "sctransform", "log1p", "normalize_total",
                    "scale", "raw_checkpoint", "batches_integrate"}:
            if scope != slice(None):
                raise Exception(f'{item} use integrate should use all sample')
            ms_data_view = self.ms_data
        elif scope == slice(None):
            ms_data_view = self.ms_data
        else:
            ms_data_view = self.ms_data[scope]
        if not ms_data_view.merged_data:
            ms_data_view.integrate(result=self.ms_data.tl.result)

        def callback_func(key, value):
            key_name = "scope_[" + ",".join(
                [str(self.ms_data._names.index(name)) for name in ms_data_view._names]) + "]"
            self.ms_data.tl.result.setdefault(key_name, {})
            self.ms_data.tl.result[key_name][key] = value

        ms_data_view._merged_data.tl.result.set_item_callback = callback_func

        def get_item_method(name):
            key_name = "scope_[" + ",".join(
                [str(self.ms_data._names.index(name)) for name in ms_data_view._names]) + "]"
            scope_result = self.ms_data.tl.result.get(key_name, None)
            if scope_result is None:
                raise KeyError
            method_result = scope_result.get(name, None)
            # if method_result is None:
            #     raise KeyError
            # return method_result
            return method_result

        ms_data_view._merged_data.tl.result.get_item_method = get_item_method

        def contain_method(item):
            key_name = "scope_[" + ",".join(
                [str(self.ms_data._names.index(name)) for name in ms_data_view._names]) + "]"
            scope_result = self.ms_data.tl.result.get(key_name, None)
            if scope_result is None:
                return False
            method_result = scope_result.get(item, None)
            if method_result is None:
                return False
            return True

        ms_data_view._merged_data.tl.result.contain_method = contain_method

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

        # def log_delayed_task(idx, *arg, **kwargs):
        #     logger.info(f'data_obj(idx={idx}) in ms_data start to run {item}')
        #     return new_attr(*arg, **kwargs)
        
        # return log_delayed_task(
        #     0,
        #     ms_data_view.merged_data.__getattribute__(self.__class__.ATTR_NAME),
        #     *args,
        #     **kwargs
        # )

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
            run_method = MSDataPlotBase.get_attribute_helper(item, self.ms_data, self.result)
            if run_method:
                return download(run_method)

        def temp(*args, **kwargs):
            if "scope" not in kwargs:
                kwargs["scope"] = slice_generator[:]
            if "mode" in kwargs:
                if kwargs["mode"] == "integrate":
                    if self.ms_data.merged_data:
                        return self._use_integrate_method(item, *args, **kwargs)
                    else:
                        raise Exception(
                            "`mode` integrate should merge first, using `ms_data.integrate`"
                        )
                elif kwargs["mode"] == "isolated":
                    self._run_isolated_method(item, *args, **kwargs)
                else:
                    raise Exception("`mode` should be one of [`integrate`, `isolated`]")
            else:
                if self.ms_data.merged_data:
                    return self._use_integrate_method(item, *args, **kwargs)
                else:
                    raise Exception(
                        "`mode` integrate should merge first, using `ms_data.integrate`"
                    )

        return temp


slice_generator = _scope_slice()
