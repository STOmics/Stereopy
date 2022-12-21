from ...log_manager import logger
from ..algorithm_base import AlgorithmBase


class LeetCodeDaily(AlgorithmBase):

    def main(self, nums, limit, goal):
        """
        Try this:
        https://leetcode.cn/problems/minimum-elements-to-add-to-form-a-given-sum/

        :param nums:
        :param limit:
        :param goal:
        :return:
        """
        logger.info(f'result is: {self.min_elements(nums, limit, goal)}')

    def min_elements(self, nums, limit, goal):
        """
        :type nums: List[int]
        :type limit: int
        :type goal: int
        :rtype: int
        """
        n = 0
        delta = goal - sum(nums)
        while delta:
            if abs(delta) > limit:
                n += abs(delta) / limit
                delta = abs(delta) % limit
            else:
                n += 1
                delta = 0
        return n

    def test_demo1(self):
        nums = [2, 2, 2, 5, 1, -2]
        limit = 5
        goal = 126614243
        logger.info(f'result is: {self.min_elements(nums, limit, goal)}')

    def test_demo2(self):
        nums = [1, -1, 1]
        limit = 3
        goal = -4
        logger.info(f'result is: {self.min_elements(nums, limit, goal)}')

    def test_demo3(self):
        nums = [1, -10, 9, 1]
        limit = 100
        goal = 0
        logger.info(f'result is: {self.min_elements(nums, limit, goal)}')
