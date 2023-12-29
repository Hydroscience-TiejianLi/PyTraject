# -*- enconding: utf-8 -*-

from __future__ import division, print_function
from datetime import datetime
import numpy as np
import shapefile
import math
from .config import Config
from .utils import hour2time, time2hour


__all__ = ['TrajectoryBatch', 'Trajectory', 'Saver', 'load_from_shp']


def _list2array(alist, dtype=np.float64):
    return np.array(alist, dtype=dtype, copy=False)


class TrajectoryBatch(object):

    def __init__(self, xyzq, t0, step, precip=None, tsize=None, time_baseline=Config.TIME_BASELINE):
        self.xyzq = xyzq  # batch x tsize x 4
        self.t0 = _list2array(t0) # batch
        self.step = _list2array(step) # batch
        self.time_baseline = Config.TIME_BASELINE
        self.size = xyzq.shape[0]

        #self.duration = (xyzq.shape[1] - 1) * abs(step)
        if precip is None:
            self.precip = None
        else:
            self.precip = _list2array(precip)
        
        if tsize is None:
            self.tsize = np.full((self.size,), self.xyzq.shape[1], dtype='int')
        else:
            self.tsize = _list2array(tsize, 'int')

    def set_precip(self, precip):
        assert len(precip) == self.size
        if self.precip is None:
            self.precip = _list2array(precip)
        else:
            self.precip[:] = precip

    def is_backtracking(self, traj_id):
        return self.step[traj_id] < 0;

    def gen_point(self, traj_filter=None, point_filter=None):
        for i in range(self.size):
            if traj_filter is None or traj_filter(i):
                for j in range(self.tsize[i]):
                    if point_filter is None or point_filter(i, j):
                        yield i, j

    def get_map(self, gridsize, variable):
        assert variable in ['density', 'height', 'humidity', 'sur_precip']

        nlat = math.ceil(180 / gridsize)
        nlon = math.ceil(360 /gridsize)

        amap = np.zeros((nlat, nlon))
        for i in range(self.size):
            for j in range(self.tsize[i]):
                x, y, z, q = self.xyzq[i, j]
                gi = int(90 - x // gridsize) % nlat
                gj = int(y // gridsize) % nlon
                if variable == 'density':
                    amap[gi, gj] += 1
                elif variable == 'height':
                    amap[gi, gj] += z
                elif variable == 'humidity':
                    amap[gi, gj] += q
                elif variable == 'sur_precip':
                    amap[gi, gj] += self.precip[i]
        return amap


def Trajectory(xyzq, t0, tstep, precip=None, time_baseline=Config.TIME_BASELINE):
    if precip is not None:
        precip = [precip]
    return TrajectoryBatch(xyzq[np.newaxis, :], [t0,], [tstep,], precip, None, time_baseline)


class Saver(object):

    def __init__(self, target):
        self.writer = shapefile.Writer(target, shapeType=shapefile.POLYLINEZ)
        self.writer.field('T0_DATE', 'D')
        self.writer.field('T0_HOUR', 'N', decimal=7)
        self.writer.field('T0_ALL_HOUR', 'N', decimal=7)
        self.writer.field('T_BENCHMARK', 'D')
        self.writer.field('TSTEP', 'N', decimal=7)

        self.writer.field('LAT0', 'N', decimal=13)
        self.writer.field('LON0', 'N', decimal=13)
        self.writer.field('PRESSURE0', 'N', decimal=7)
        self.writer.field('SURF_PRECIP', 'N', decimal=7)
        self.writer.field('BACKTRACKING', 'L')
        self.writer.field('DQ0', 'N', decimal=13)

    def save(self, trajbatch):
        for i, traj in enumerate(trajbatch.xyzq):
            self.writer.linez([traj[:,[1,0,2,3]]])
            t0 = trajbatch.t0[i]
            ttime = hour2time(t0)
            t0_date = ttime.date()
            t0_hour = t0 - time2hour(datetime(year=t0_date.year, month=t0_date.month, day=t0_date.day))
            benchmark = Config.TIME_BASELINE.date()
            tstep = trajbatch.step[i]

            #lat0, lon0, pressure0, q0 = traj[0]
            lat0, lon0, pressure0, q0 = traj[1]
            if trajbatch.precip is not None:
                precip = trajbatch.precip[i]
            else:
                precip = -999
            back = tstep < 0
            dq = (traj[2,3] - traj[0,3]) / tstep
            self.writer.record(t0_date, t0_hour, t0, benchmark, tstep, lat0, lon0, pressure0, precip, back, dq)

    def close(self):
        self.writer.close()


def load_from_shp(filename):
    with shapefile.Reader(filename) as shp:
        assert shp.shapeType == shapefile.POLYLINEZ
        count, max_tsize = len(shp), len(shp.shape(0).z)

        xyzq = np.zeros((count, max_tsize, 4), dtype='float64')
        tsize = np.full((count,), max_tsize, dtype='int')
        precip = np.zeros((count,), dtype='float64')
        tstep = np.zeros((count,), dtype='float64')
        t0 = np.zeros((count, ), dtype='float64')

        for i in range(count):
            shprec = shp.shapeRecord(i)
            xyzq[i, :, 2] = shprec.shape.z
            xyzq[i, :, 3] = shprec.shape.m
            xyzq[i, :, 1::-1] = shprec.shape.points
            for j in range(max_tsize):
                if shprec.shape.z[j] < Config.BOUND_TOP:
                    break
            tsize[i] = j + 1;
            precip[i] = shprec.record['SURF_PRECIP'[:10]]
            tstep[i] = shprec.record['TSTEP']
            t0[i] = shprec.record['T0_ALL_HOUR'[:10]]

        return TrajectoryBatch(xyzq, t0, tstep, precip, tsize)
