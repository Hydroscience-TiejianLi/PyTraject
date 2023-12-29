# -*- coding: utf-8 -*-

from __future__ import division, print_function
from .base import ParallelDataEngine
from ..config import Config
from ..utils import time2hour
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
from multiprocessing.sharedctypes import RawArray
from ..parser import parserNC, parserNC_2d
import ctypes
import numpy as np
import sys
import os


__all__ = ['EraDataEngine', 'era_time_range']


DEBUG_DAYS = 1
default_level_filter = lambda l: l >= Config.PRESSURE_TOP and l <= Config.PRESSURE_BOTTOM
#定义 本程序支持的计算的开始和 结束时间，这里如果有数据更新则需要修改 结束时间即可
era_time_range = Config.TIME_RANGE

'''
def era5_levels():
    levels = '1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000'
    return [int(i) * 100 for i in levels.split('/')]
'''


class EraDataEngine(ParallelDataEngine):

    share_variables = {
        'data': None,
        'data_2d': None,
        'times': None,
        'levels': None,
        'lats': None,
        'lons': None,
        'shape': None,
        'shape_2d': None,
        'valid_length': None
    }

    def __init__(self, dictory, buffer=None, time_baseline=Config.TIME_BASELINE, level_filter=default_level_filter):
        self.first_read = False
        self.built = False
        self.time_baseline = time_baseline

        if dictory != '@':
            self.dictory = dictory
            self.buffer = buffer
            self.buffer_months = buffer[1] - buffer[0] + 1
            self.buffer_date = []
            self.buffer_timeidx = []

    # 准备要计算的数据，实际上是存储两个月数据到一个数组data中去 ；share_variables 填充变量 ；用于为生产任务做准备
    def prepare_for(self, year, month, data_type):
        new_date = []
        new_timeidx = [0]
        for delta in range(self.buffer[0], self.buffer[1] + 1):
            cur_year, cur_month = year,  month + delta
            while cur_month <= 0:
                cur_month += 12
                cur_year -= 1
            while cur_month > 12:
                cur_month -= 12
                cur_year += 1
            cur_date = datetime(cur_year, cur_month, 1)

            if cur_date < era_time_range[0] or cur_date > era_time_range[1]:
                continue

            new_date.append(cur_date)
            if cur_date in self.buffer_date:
                old_idx = self.buffer_date.index(cur_date)
                ob, oe = self.buffer_timeidx[old_idx], self.buffer_timeidx[old_idx+1]
                nb, ne = new_timeidx[-1], new_timeidx[-1] + oe - ob
                new_timeidx.append(ne)
                self.data[nb:ne] = self.data[ob:oe]
                self.data_2d[nb:ne] = self.data_2d[ob:oe]
                self.times[nb:ne] = self.times[ob:oe]

            else:
                nc = os.path.join(self.dictory, cur_date.strftime('%Y%m'))
                nc_2d = os.path.join(self.dictory, cur_date.strftime('%Y%m_2d.nc'))
                t_end = self._read_monthly_file(nc, nc_2d, new_timeidx[-1], data_type)
                new_timeidx.append(t_end)
        self.built = False
        self.buffer_date = new_date
        self.buffer_timeidx = new_timeidx
        EraDataEngine.share_variables['valid_length'] = self.buffer_timeidx[-1]

    # 读取 nc 解析成为 self.data
    def _read_monthly_file(self, compressed_nc, compressed_nc_2d, t_begin, data_type):
        sv = EraDataEngine.share_variables
        print('Reading %s ...' %compressed_nc, end='')
        sys.stdout.flush()

        def load_from_npz(compressed_nc, data_type=None):
            data_inNPZ = np.load(compressed_nc + ('_3d_%s.npz'%data_type), allow_pickle=True)
            data_inNPZ_2d = np.load(compressed_nc + ('_2d_%s.npz'%data_type), allow_pickle=True)
            times_inNC = data_inNPZ['times_inNC']
            levels_inNC = data_inNPZ['levels_inNC']
            lons_inNC = data_inNPZ['lons_inNC']
            lats_inNC = data_inNPZ['lats_inNC']
            variables = data_inNPZ['variables']
            data_inNC = data_inNPZ['data_inNC']
            variables_2d = data_inNPZ_2d['variables_2d']
            data_inNC_2d = data_inNPZ_2d['data_inNC_2d']

            return times_inNC, levels_inNC, lons_inNC, lats_inNC, variables, data_inNC, variables_2d, data_inNC_2d

        if data_type:

            times_inNC, levels_inNC, lons_inNC, lats_inNC, variables, data_inNC, variables_2d, data_inNC_2d = load_from_npz(compressed_nc, data_type)

        else:

            times_inNC, levels_inNC, lons_inNC, lats_inNC, variables, data_inNC = parserNC(compressed_nc)
            _, _, _, variables_2d, data_inNC_2d = parserNC_2d(compressed_nc_2d)

        if not self.first_read:
            sv['lats'] = RawArray(ctypes.c_float, lats_inNC)
            sv['lons'] = RawArray(ctypes.c_float, lons_inNC)
            sv['levels'] = RawArray(ctypes.c_float, levels_inNC)
            self.lats = np.frombuffer(sv['lats'], dtype='float32')
            self.lons = np.frombuffer(sv['lons'], dtype='float32')
            sv['times'] = RawArray(ctypes.c_float, self.buffer_months*31*24//(Config.GRID[1]))
            self.times = np.frombuffer(sv['times'], dtype='float32')
            sv['shape'] = (len(sv['times']), len(sv['levels']), len(sv['lats']), len(sv['lons']), len(variables))
            sv['data'] = RawArray(ctypes.c_float, int(np.product(sv['shape'], dtype='i8')))
            self.data = np.frombuffer(sv['data'], dtype='float32').reshape(sv['shape'])
            
            sv['shape_2d'] = (len(sv['times']), len(sv['lats']), len(sv['lons']), len(variables_2d))
            sv['data_2d'] = RawArray(ctypes.c_float, int(np.product(sv['shape_2d'], dtype='i8')))
            self.data_2d = np.frombuffer(sv['data_2d'], dtype='float32').reshape(sv['shape_2d'])

            self.first_read = True

        ts = [time2hour(t) for t in times_inNC]
        t_end = t_begin + len(ts)
        self.times[slice(t_begin, t_end)] = ts
        #radius = Config.RADIUS
        data = self.data[slice(t_begin, t_end)]
        data[:] = data_inNC
        data_2d = self.data_2d[slice(t_begin, t_end)]
        data_2d[:] = data_inNC_2d
        data_2d *= 100
        # preprocessing
        na = np.newaxis
        lats_rad = self.lats / 180 * np.pi
        # temporal remedy for polar point
        lats_rad[0] = 0
        lats_rad[-1] = 0

        # 下面的操作是将 三个方向的风速由 m/s 转为 经纬度/小时
        # ['V component of wind', 'U component of wind', 'Vertical velocity', 'Z', 'Specific humidity']
        # 风的V分量  0          #风的U分量   1         #垂直速度  2          #地球重力位势 3      #比湿度 4
        # data[times,levels,lats,lons,variables]
        data[:,:,:,:,:3] *= 3600  # seconds to hours
        data[:,:,:,:,3] += Config.RADIUS  # radius 加上地球半径,地势高+地球半径
        # lat' = v' / radius
        data[:,:,:,:,0] /= data[:,:,:,:,3]
        # lon' = u' / (r*cos(lat))
        data[:,:,:,:,1] /= data[:,:,:,:,3] * np.cos(lats_rad)[na,na,:,na]
        data[:,:,:,:,:2] *= 180 / np.pi  # rad to degree
        print('done!')
        sys.stdout.flush()
        return t_end

    @classmethod
    def fork(cls, **kwargs):
        obj = EraDataEngine('@')
        sv  = cls.share_variables
        sv.update(kwargs)
        obj.data = np.frombuffer(sv['data'], dtype='float32').reshape(sv['shape'])
        obj.data_2d = np.frombuffer(sv['data_2d'], dtype='float32').reshape(sv['shape_2d'])
        obj.lats = np.frombuffer(sv['lats'], dtype='float32')
        obj.lons = np.frombuffer(sv['lons'], dtype='float32')
        return obj

    def _build_interpolater(self):
        sv = EraDataEngine.share_variables
        valid = sv['valid_length']
        points = [
            np.frombuffer(sv['times'], dtype='float32')[:valid],
            np.frombuffer(sv['levels'], dtype='float32'),
            self.lats[::-1], # must be ascending
            self.lons
        ]
        points_2d = [
            np.frombuffer(sv['times'], dtype='float32')[:valid],
            self.lats[::-1], # must be ascending
            self.lons
        ]

        data = self.data[:valid, :, ::-1, :, :]
        data_2d = self.data_2d[:valid, ::-1, :, :]
        self.interp_wind = RegularGridInterpolator(points, data[:,:,:,:, :3])
        self.interp_humidity = RegularGridInterpolator(points, data[:,:,:,:,4])
        self.interp_surface_z = RegularGridInterpolator(points_2d, data_2d[:,:,:,0])      # 根据时间、维度、经度插值地表气压高

        self.built = True

    def wind_at(self, lat, lon, z, t):
        if not self.built:
            self._build_interpolater()

        # if near polar point, skip
        if lat > self.lats[1] or lat < self.lats[-2]:
            return [0.0, 0.0, 0.0]

        while lon >= self.lons[-1]:
            lon -= 360
        while lon < self.lons[0]:
            lon += 360

        try:
            if lon <= self.lons[-1]:
                res = self.interp_wind([t, z, lat, lon])
            else:
                left = self.interp_wind([t, z, lat, self.lons[-1]])
                right = self.interp_wind([t, z, lat, self.lons[0]])
                k1 = lon - self.lons[-1]
                k2 = self.lons[0] + 360 - lon
                res = (k2 * left + k1 * right) / (k1 + k2)
                # res = self.interp_wind_extra([t, z, lat, lon])
            return list(res[0])
        except ValueError as e:
            print(e, 'at (t, level, lat, lon)=(%.3f, %.3f, %.3f, %.3f)' %
                  (t, z, lat, lon))
            return [0.0, 0.0, 0.0]

    def humidity_at(self, lat, lon, z, t):
        if not self.built:
            self._build_interpolater()

        # if near polar point, skip
        if lat > self.lats[1] or lat < self.lats[-2]:
            return -999

        while lon >= self.lons[-1]:
            lon -= 360
        while lon < self.lons[0]:
            lon += 360

        try:
            if lon <= self.lons[-1]:
                res = self.interp_humidity([t, z, lat, lon])
            else:
                left = self.interp_humidity([t, z, lat, self.lons[-1]])
                right = self.interp_humidity([t, z, lat, self.lons[0]])
                k1 = lon - self.lons[-1]
                k2 = self.lons[0] + 360 - lon
                res = (k2 * left + k1 * right) / (k1 + k2)
                # res = self.interp_humidity_extra([t, z, lat, lon])
            return res[0]
        except ValueError as e:
            print(e, 'at (t, level, lat, lon)=(%.3f, %.3f, %.3f, %.3f)' %
                  (t, z, lat, lon))
            return -999

    def surface_z_at(self, lat, lon, t):
        if not self.built:
            self._build_interpolater()

        # if near polar point, skip
        if lat > self.lats[1] or lat < self.lats[-2]:
            return Config.PRESSURE_BOTTOM

        while lon >= self.lons[-1]:
            lon -= 360
        while lon < self.lons[0]:
            lon += 360

        try:
            if lon <= self.lons[-1]:
                res = self.interp_surface_z([t, lat, lon])
            else:
                left = self.interp_surface_z([t, lat, self.lons[-1]])
                right = self.interp_surface_z([t, lat, self.lons[0]])
                k1 = lon - self.lons[-1]
                k2 = self.lons[0] + 360 - lon
                res = (k2 * left + k1 * right) / (k1 + k2)
            return res[0]
        except ValueError as e:
            print(e, 'at (t, lat, lon)=(%.3f, %.3f, %.3f)' %
                  (t, lat, lon))
            return Config.PRESSURE_BOTTOM
