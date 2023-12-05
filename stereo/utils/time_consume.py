import time
import uuid
from functools import wraps

from stereo.log_manager import logger


class TimeConsume:
    def __init__(self):
        self.start_time_map = {}
        self.accumulate = False
        self.time_accumulated = 0

    def start(self, key=None):
        if key is None:
            key = uuid.uuid1()
        self.start_time_map[key] = time.time()
        return key

    def start_to_accumulate(self):
        self.accumulate = True

    def get_time_consumed(self, key, restart=True, unit='s'):
        now_time = time.time()
        start_time = self.start_time_map.get(key)
        if start_time is not None:
            del self.start_time_map[key]
            time_consumed = (now_time - start_time)
            time_consumed_raw = time_consumed
            if unit.lower() == 'ms':
                time_consumed *= 1000
            elif unit.lower() == 'm':
                time_consumed = time_consumed / 1000 / 60
        else:
            time_consumed = None

        if self.accumulate and time_consumed is not None:
            self.time_accumulated += time_consumed_raw

        if restart:
            self.start(key)

        return time_consumed

    def get_time_accumulated(self):
        time_accumulated = self.time_accumulated
        self.time_accumulated = 0
        self.accumulate = False
        return time_accumulated


def log_consumed_time(outer_func=None, unit='s'):
    def log(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            tc = TimeConsume()
            logger.info('start to run {}...'.format(func.__name__))
            tk = tc.start()
            res = func(*args, **kwargs)
            logger.info('{} end, consume time {:.4f}{}.'.format(
                func.__name__, tc.get_time_consumed(key=tk, restart=False, unit=unit), unit))
            return res

        return wrapped

    if outer_func is not None:
        return log(outer_func)
    else:
        return log
