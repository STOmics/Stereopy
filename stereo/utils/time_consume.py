import time
import uuid

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
