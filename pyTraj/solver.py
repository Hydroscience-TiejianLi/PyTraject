# -*- conding: utf-8 -*-

from __future__ import division, print_function
from .utils import time2hour
from .config import Config
from .data import ParallelDataEngine, EraDataEngine
from .trajectory import *
from scipy.integrate import solve_ivp
# from collections import Iterable
from multiprocessing.pool import Pool
from contextlib import closing
from datetime import datetime
import numpy as np
import types


__all__ = ['cal_single', 'submit_task', 'init_pool']



def _truncate_pressure(z):
    return min(max(Config.PRESSURE_TOP + 0.1, z), Config.PRESSURE_BOTTOM - 0.1)


def cal_single(data_engine, lat0, lon0, z0, t0, tstep, duration, fast=False):
    if isinstance(t0, datetime):
        t0_hour = time2hour(t0)
    else:
        t0_hour = t0
    backtracking = tstep < 0

    def ode_func(t, y):
        if backtracking:
            t *= -1
        t += t0_hour
        lat, lon, z = y

        cz = _truncate_pressure(z)
        result = data_engine.wind_at(lat, lon, cz, t)

        if backtracking:
            result = [-r for r in result]
        else:
            result = [r for r in result]
        #if z >= data_engine.surface_z_at(lat, lon, t) and result[2] > 0:
        if z >= Config.PRESSURE_BOTTOM and result[2] > 0:
            result[2] = 0
        if z <= Config.PRESSURE_TOP and result[2] < 0:
            # result = [0.0, 0.0, 0.0]
            result[2] = 0
        return result

    '''
    t_span = (0, duration + 0.5*abs(tstep))
    y0 = [lat0, lon0, z0]
    t_eval = np.arange(0, t_span[1], abs(tstep))
    atol, rtol = 1e-5, 1e-8
    if fast:
        atol *= fast
        rtol *= fast
    traj = solve_ivp(ode_func, t_span, y0, t_eval=t_eval,
                     atol=atol, rtol=rtol, method='RK23').y.T
    '''
    t_span = (0, duration + 0.6*abs(tstep))
    y0 = [lat0, lon0, z0]
    t_eval = np.arange(0.5*abs(tstep), t_span[1], abs(tstep))
    atol, rtol = 1e-5, 1e-8
    if fast:
        atol *= fast
        rtol *= fast
    traj1 = solve_ivp(ode_func, t_span, y0, t_eval=t_eval,
                     atol=atol, rtol=rtol, method='RK23').y.T
    traj2 = solve_ivp(ode_func, (0, -0.6*abs(tstep)), y0, t_eval=[0, -0.5*abs(tstep)],
                     atol=atol, rtol=rtol, method='RK23').y.T

    traj = np.vstack((traj2[::-1], traj1))
    t_eval = np.hstack((np.array([-0.5*abs(tstep), 0]), t_eval))

    if backtracking:
        t_eval *= -1
    t_eval += t0_hour
    traj_t = t_eval

    traj_q = np.array([data_engine.humidity_at(lat, lon, _truncate_pressure(
        z), t) for (lat, lon, z), t in zip(traj, traj_t)])
    traj = np.concatenate((traj, traj_q[:, np.newaxis]), axis=1)

    return Trajectory(traj, t0_hour, tstep)


pool = None
data_engine_in_worker = None


def init_pool(data_engine, n_workers=Config.NUM_WORKERS):
    global pool
    if pool:
        pool.close()

    n_processes = min(Config.NUM_WORKERS, n_workers)

    ia = [data_engine.__class__.fork]
    for k, v in data_engine.__class__.share_variables.items():
        ia.append(k)
        ia.append(v)
    pool = Pool(processes=n_processes,
                initializer=_init_worker, initargs=ia)

#根据 经纬度、高度层、时间 提交一个任务到线程池子里进行计算
def submit_task(lat0, lon0, z0, t0, tstep, duration, fast=False, callback=None, error_callback=None):
    assert pool is not None
    async_res = pool.apply_async(
        func=_job,
        args=(lat0, lon0, z0, t0, tstep, duration, fast),
        callback=callback,
        error_callback=error_callback
    )
    return async_res

def end_tasks():

    if pool:
        pool.close()
        pool.join()


def _init_worker(*args):
    global data_engine_in_worker
    fork = args[0]
    dargs = {}
    for i in range(1, len(args), 2):

        dargs[args[i]] = args[i + 1]
    data_engine_in_worker = fork(**dargs)

def _job(*args):
    return cal_single(data_engine_in_worker, *args)
