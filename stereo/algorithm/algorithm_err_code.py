from enum import Enum


class ErrorCode(Enum):
    Success, SuccessMsg = 0, 'All steps done, algorithm run perfectly! ($_$)'
    Failed, FailedMsg = -1, 'Failed, examine the log and do not be depressed! (^_^)'
