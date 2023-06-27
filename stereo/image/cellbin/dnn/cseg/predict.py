from queue import Queue
from threading import Thread, Event

import numpy as np


class CellPredict(object):
    def __init__(self, model, f_preformat, f_postformat):
        self._model = model
        self._f_preformat = f_preformat
        self._f_postformat = f_postformat
        self._t_queue_maxsize = 100
        self._t_workdone = Event()
        self._t_queue = Queue(maxsize=self._t_queue_maxsize)

    def _f_productor(self, img_lst):
        self._t_workdone.set()
        for img in img_lst:
            val_sum = np.sum(img)
            if val_sum <= 0.0:
                pred = np.zeros(img.shape, np.uint8)
            else:
                pred = self._model.f_predict(self._f_preformat(img))
            self._t_queue.put([pred, val_sum], block=True)
        self._t_workdone.clear()
        return

    def _f_consumer(self, pred_lst):
        while (self._t_workdone.is_set()) or (not self._t_queue.empty()):
            pred, val_sum = self._t_queue.get(block=True)
            if val_sum > 0:
                pred = self._f_postformat(pred)
            pred_lst.append(pred)
        return

    def _f_clear(self):
        self._t_queue = Queue(maxsize=self._t_queue_maxsize)

    def _run_batch(self, img_lst):
        self._f_clear()
        pred_lst = []
        t_productor = Thread(target=self._f_productor, args=(img_lst,))
        t_consumer = Thread(target=self._f_consumer, args=(pred_lst,))
        t_productor.start()
        t_consumer.start()
        t_productor.join()
        t_consumer.join()
        self._f_clear()
        return pred_lst

    def f_predict(self, img_lst):
        img = img_lst

        if isinstance(img_lst, list):
            return self._run_batch(img_lst)

        if np.sum(img) < 1:
            pred = np.zeros(img.shape, np.uint8)
        else:
            pred = self._model.f_predict(self._f_preformat(img))
            pred = self._f_postformat(pred)
        return pred
