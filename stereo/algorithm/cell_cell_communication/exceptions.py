# -*- coding: utf-8 -*-
# @Time    : 2022/12/23 13:41
# @Author  : liuxiaobin
# @File    : exceptions.py
# @Version：V 0.1
# @desc :


class ProcessMetaException(Exception):
    def __init__(self):
        super(ProcessMetaException, self).__init__('Error processing Meta data')


class ParseCountsException(Exception):
    def __init__(self, description: str = None, hint: str = None):
        super(ParseCountsException, self).__init__('Invalid Counts data')
        self.description = description
        self.hint = hint


class ThresholdValueException(Exception):
    def __init__(self, threshold_value):
        super(ThresholdValueException, self).__init__(
            'Threshold value ({}) is not valid. Accepted range: 0<=threshold<=1'.format(threshold_value))


class AllCountsFilteredException(Exception):
    def __init__(self, description: str = None, hint: str = None):
        super(AllCountsFilteredException, self).__init__('All counts filtered')
        self.description = description
        self.hint = hint


class NoInteractionsFound(Exception):
    def __init__(self, description: str = None, hint: str = None):
        super(NoInteractionsFound, self).__init__('No CellPhoneDB interacions found in this input.')
        self.description = description
        self.hint = hint
