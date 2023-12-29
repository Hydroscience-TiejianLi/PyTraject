# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import sys
import numpy as np
from datetime import datetime

from ..utils import hour2time, time2hour
from ..config import Config


class Points_load(object):

    def __init__(self, points_file):

        points = np.load(points_file, allow_pickle=True)

        self.times = points[:, 0]
        self.lats = points[:, 1]
        self.lons = points[:, 2]
        self.precip = points[:, 3]

    #生成要计算的任务
    def gen_task_monthly(self):
        start = min(self.times)
        end   = max(self.times)
        times = np.array([time2hour(row, Config.TIME_BASELINE) for row in self.times])
        for year in range(start.year, end.year + 1):
            ms, me = 1, 12
            if year == start.year:
                ms = start.month
            if year == end.year:
                me = end.month

            for month in range(ms, me + 1):

                t_start = datetime(year, month, 1)
                if month == 12:
                    t_end = datetime(year+1, 1, 1)
                else:
                    t_end = datetime(year, month + 1, 1)
                selected = np.logical_and(self.times >= t_start, self.times < t_end)
                # yield 类似于 return 的作用，返回
                yield {
                    'year': year,
                    'month': month,
                    'count': selected.sum(),
                    'times': times[selected],
                    'lats': self.lats[selected],
                    'lons': self.lons[selected],
                    'precip': self.precip[selected]
                }