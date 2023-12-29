# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import sys
import calendar
import numpy as np
import shapefile
from shapely.geometry import shape, Point
from contextlib import closing
from datetime import datetime
from netCDF4 import Dataset

from ..utils import hour2time, time2hour, convert_time_base
from ..config import Config


def thres_filter(threshold):
    def f(rain, lats, lons, times):
        rain[rain < threshold] = np.ma.masked
        return rain
    return f

class RainFilter(object):

    def __init__(self, rain_root, filter_func, grid_size=0.25, shape_bound=None):
        
        self.rain_root = rain_root
        self.filter = filter_func

        self.lats = np.arange(90.0,-90-grid_size,-grid_size)
        self.lons = np.arange(0,360,grid_size)
        mask = np.zeros((1, len(self.lats), len(self.lons)))
  
        # 如果给定shp文件，就按照shp文件来
        if shape_bound:

            with shapefile.Reader(shape_bound) as shpfile:
                geo_json = shpfile.shape().__geo_interface__
                bound = shape(geo_json)
                left, bot, right, top = bound.bounds
                left_idx = np.searchsorted(self.lons, left)
                right_idx = np.searchsorted(self.lons, right)
                top_idx = -np.searchsorted(self.lats[::-1], top)
                bot_idx = -np.searchsorted(self.lats[::-1], bot)
                for i in range(top_idx, bot_idx):
                    for j in range(left_idx, right_idx):
                        lat, lon = self.lats[i], self.lons[j]
                        if bound.contains(Point(lon, lat)):
                            mask[0, i, j] = 1

        # 如果不给定shp文件，就默认60°S~60°N
        else:

            mask[0, (len(self.lat)-1)//6:-(len(self.lat)-1)//6] = 1

        self.mask = np.logical_not(mask.astype('bool'))
        self.times = None
        self.precip = None
        self.idx = None
    #生成每月要计算的任务
    def gen_task_monthly(self, start: datetime, end: datetime):
        t_min = time2hour(start)
        t_max = time2hour(end)
        for year in range(start.year, end.year + 1):
            ms, me = 1, 12
            if year == start.year:
                ms = start.month
            if year == end.year:
                me = end.month

            for month in range(ms, me + 1):

                nc_file = os.path.join(self.rain_root,  '%d%02d.nc'%(year, month))
                self._read_monthly_file(nc_file)# 读取降雨月文件
                times = self.times[self.idx[:,0]]

                t_start = time2hour(datetime(year, month, 1), Config.TIME_BASELINE)
                t_start = max(t_start, t_min)
                if month == 12:
                    t_end = time2hour(datetime(year+1, 1, 1), Config.TIME_BASELINE) - 1
                else:
                    t_end = time2hour(datetime(year, month + 1, 1), Config.TIME_BASELINE) - 1
                t_end = min(t_end, t_max)
                selected = np.logical_and(times >= t_start, times <= t_end)
                selected_idx = self.idx[selected]
                # yield 类似于 return 的作用，返回
                yield {
                    'year': year,
                    'month': month,
                    'count': selected.sum(),
                    'times': times[selected],
                    'lats': self.lats[selected_idx[:,1]],
                    'lons': self.lons[selected_idx[:,2]],
                    'precip': self.precip[selected]
                }
    # 读取降水文件（一年一个文件存储全年的降水数据）
    # nc 读取路径 cal_rate 表示是否要计算（根据降雨阈值计算出来的）有效降雨点占比例。
    def _read_monthly_file(self, nc, cal_rate=False):
        # del self.times, self.precip, self.idx
        # get rainfall data
        print('Read surface total precipitation from %s ... ' % nc, end='')
        sys.stdout.flush()
        with closing(Dataset(nc, 'r')) as rain_ds:
            times = rain_ds.variables['time'][:].data
            time_units = rain_ds.variables['time'].units
            assert time_units.startswith('hours since')
            time_base = datetime.strptime(time_units[12:31], '%Y-%m-%d %H:%M:%S')
            self.times = convert_time_base(times, time_base, Config.TIME_BASELINE)
            rain = rain_ds.variables['tp'][:]
            _, mask = np.broadcast_arrays(rain.data, self.mask)
            rain.mask = mask

            assert rain_ds.variables['tp'].units == 'm'

        rain *= 1000  # m to mm

        

        # filter
        if cal_rate:
            all_rain = rain.sum()

        # rain[rain < (self.threshold * self.step)] = np.ma.masked # the unit of threshold is mm/h
        rain = self.filter(rain, self.lats, self.lons, self.times)

        if cal_rate:
            used_rain = rain.sum()

        self.idx = np.argwhere(~rain.mask)
        self.precip = rain.compressed()

        print('done!')
        sys.stdout.flush()

        if cal_rate:
            return all_rain, used_rain