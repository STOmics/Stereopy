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


class InvalidDatabase(Exception):
    def __init__(self, description: str = None):
        if description is None:
            description = 'Invalid database. Please choose from cellphonedb, liana and celltalkdb, ' \
                          'or input a path of database.'
        super(InvalidDatabase, self).__init__(description)


class PipelineResultInexistent(Exception):
    def __init__(self, res_key: str = None):
        if res_key is not None:
            description = f"The result specified by {res_key} is not exists."
        else:
            description = "The result is not exists."
        super(PipelineResultInexistent, self).__init__(description)


class InvalidSpecies(Exception):
    def __init__(self, species: str = None):
        if species is None:
            description = "Invalid species, please choose from HUMAN and MOUSE."
        else:
            description = f"Species {species.upper()} is invalid, please choose from HUMAN and MOUSE."
        super(InvalidSpecies, self).__init__(description)


class InvalidMicroEnvInput(Exception):
    def __init__(self, description: str = None):
        super(InvalidMicroEnvInput, self).__init__(description)
        self.description = description


class InvalidNicheMethod(Exception):
    def __init__(self, method: str = None):
        if method is None:
            description = "Invalid niche method, please choose from fixed and adaptive."
        else:
            description = f"Niche method {method} is invalid, please choose from fixed and adaptive."
        super(InvalidNicheMethod, self).__init__(description)
