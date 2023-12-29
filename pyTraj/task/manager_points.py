# -*- coding: utf-8 -*-

from __future__ import division, print_function
from ..data import EraDataEngine, era_time_range
from .points_load import Points_load
from ..solver import init_pool, submit_task, end_tasks
from ..trajectory import Saver
from ..config import Config
from datetime import datetime, timedelta
from contextlib import closing
import calendar
import sys
import os
import traceback
import time
import numpy as np

default_zs = Config.DEFAULT_ZS

class TaskManager_points(object):

    def __init__(self, data_root, points_file, traj_step=-1, duration=10*24, 
                 zs=default_zs, month_buffer=1):
        self.zs = zs
        self.traj_step = traj_step
        self.duration = duration

        mb = month_buffer
        if self.traj_step < 0:
            buffer = [-mb + 1, 0]
        else:
            buffer = [0, mb - 1]

        self.data_engine = EraDataEngine(data_root, buffer)
        self.points_load = Points_load(points_file)

    def run(self, save_root, batch_size=1e4, fast=False, data_type=None):
        gen_task = self.points_load.gen_task_monthly()
        #从生成器中获取每一个月要计算的点
        for tasks in gen_task:
            bs = int(batch_size) # 设定多少条迹线存储一次；一个shapefile 不同存储太多数据；
            year = tasks['year']
            month = tasks['month']
            count = tasks['count']
            print('Calculating trajectories that originates in %d-%d' %(year, month))
            sys.stdout.flush()
            if count == 0:
                print('Skip')
                sys.stdout.flush()
                continue
            self.data_engine.prepare_for(year, month, data_type)
            counts = np.zeros(count, dtype=np.int)
            for i in range(count):
                counts[i] = np.sum(default_zs < self.data_engine.surface_z_at(tasks['lats'][i], tasks['lons'][i], tasks['times'][i]))
            counts_sum = np.sum(counts)
            print('There are %d points and %d tasks in total' % (count, counts_sum))
            sys.stdout.flush()
            init_pool(self.data_engine, counts_sum)
            part = 0
            trajs = []
            # 任务生成器 传入任务id  i 的开始 和 结束 ，
            # 根据 任务id 可以获取到具体的 lat lon time ，这里再循环不同的气压层
            # 每次调用 生成一个 i z
            def task_generator(s, e):
                for i in range(s, e):
                    for z in self.zs[:counts[i+s]]:
                        yield i, z
            #根据任务id和高度层 生成具体任务，并提交任务到线程池子中去；
            def add_task(i, z, task_gen):
                lat = tasks['lats'][i]
                lon = tasks['lons'][i]
                precip = tasks['precip'][i]
                time = tasks['times'][i]
                #如何任务执行成功了，调用 生成器 next 方法 生成下次的任务
                def success(traj):
                    traj.set_precip([precip])
                    trajs.append(traj)
                    try:
                        ni, nz = next(task_gen)
                        add_task(ni, nz, task_gen)
                    except StopIteration:
                        pass
                #如何任务执行失败了，则跳过执行下一个
                def fail(ex):
                    print('Task (lat=%.3f, lon=%.3f, t=%.0f, z=%.0f) failed, pass' % (
                        lat, lon, time, z))
                    traceback.print_stack()
                    try:
                        ni, nz = next(task_gen)
                        add_task(ni, nz, task_gen)
                    except StopIteration:
                        pass
                #提交一次计算任务 传入经、纬度 气压层和时间；
                submit_task(lat, lon, z, time, self.traj_step, self.duration, fast, success, fail)
            #如果任务个数太多超过bs(默认10000)，则任务分组，存储到不同的shapefile中。
            for s in range(0, count, bs):
                e = min(count, s + bs)
                gen = task_generator(s, e)
                for _ in range(Config.NUM_WORKERS * 2):
                    try:
                        i, z = next(gen)
                    except StopIteration:
                        break
                    add_task(i, z, gen)

                while len(trajs) < counts_sum:
                    continue

                end_tasks()

                with closing(Saver(os.path.join(save_root, "%d%02d-%d" % (year, month, part)))) as saver:
                    for traj in trajs:
                        saver.save(traj)

                trajs.clear()
                part += 1

            print('finished!')
