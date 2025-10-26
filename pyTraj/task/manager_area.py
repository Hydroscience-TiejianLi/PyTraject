# -*- coding: utf-8 -*-

from __future__ import division, print_function
from ..data import EraDataEngine, era_time_range
from .rain_era5 import RainFilter, thres_filter
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
from tqdm import tqdm

default_zs = Config.DEFAULT_ZS

class TaskManager_area(object):

    def __init__(self, data_root, rain_root,
                 traj_step=-1, duration=10*24, zs=default_zs,
                 rain_thres=None, month_buffer=1, shape_bound=None):
        self.zs = zs
        self.traj_step = traj_step
        self.duration = duration

        mb = month_buffer
        if self.traj_step < 0:
            buffer = [-mb + 1, 0]
        else:
            buffer = [0, mb - 1]

        self.data_engine = EraDataEngine(data_root, buffer)

        f = thres_filter(rain_thres)
        self.rain_filter = RainFilter(rain_root, f, 0.25, shape_bound)

    def run(self, from_year, from_month, from_day, from_hour, to_year, to_month, to_day, to_hour, save_root, batch_size=1e4, fast=False, data_type=None):
        start_date = datetime(from_year, from_month, from_day, from_hour)
        # 如何是后向追踪，则计算出 开始时间和结束时间
        if self.traj_step < 0:
            min_start = era_time_range[0] + timedelta(hours=self.duration)
            start_date = max(min_start, start_date)
        end_date = datetime(to_year, to_month, to_day, to_hour)
        if self.traj_step > 0:
            max_end = era_time_range[1] - timedelta(hours=self.duration)
            end_date = min(max_end, end_date)

        # 根据降雨过滤条件，每个月生成一个任务，计算当一个月内要计算的栅格数和对应的时间和降雨量,
        # gen_task 是一个generator 生成器
        gen_task = self.rain_filter.gen_task_monthly(start_date, end_date)
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
            
                # 修正：每批的“预期任务数”应为这一批各点的层数之和
                expected = int(np.sum(counts[s:e]))
                if expected == 0:
                    print(f"Part {part + 1}: points [{s}–{e}) has no tasks, skip.")
                    part += 1
                    continue
            
                print(f"Running part {part + 1}: processing points {s}–{e} (Total points: {e}/{count})")
                sys.stdout.flush()
            
                # 生成当前批次的任务；修正：counts[i] 而不是 counts[i+s]
                def task_generator(s0, e0):
                    for i in range(s0, e0):
                        # 这一格要算多少个层，就发多少任务
                        for z in self.zs[:counts[i]]:
                            yield i, z
            
                gen = task_generator(s, e)
            
                # 预热提交，避免工作进程空闲
                for _ in range(Config.NUM_WORKERS * 2):
                    try:
                        i, z = next(gen)
                    except StopIteration:
                        break
                    add_task(i, z, gen)
            
                # 用 tqdm 展示这一批的进度，并加“无进展超时保护”
                bar = tqdm(total=expected, desc=f"Part {part} in progress")
                step = 0          # 连续“无新增结果”的次数
                last_done = 0     # 上次观测到的完成数量
            
                # 注意：这一批的完成定义为 len(trajs) 达到 expected
                while len(trajs) < expected:
                    # 实时推进进度条
                    done_now = len(trajs)
                    bar.update(done_now - last_done)
            
                    if done_now == last_done:
                        step += 1
                    else:
                        step = 0
                        last_done = done_now
            
                    # 每 60 秒检查一次；连续 30 次（~30 分钟）无进展则超时跳出
                    time.sleep(60)
                    print(f"\nMonitoring progress... (no change count: {step})")
                    sys.stdout.flush()
                    if step >= 30:
                        print("Timeout reached — exiting current loop to prevent freeze.")
                        break
            
                # 关闭进度条（如果提前超时也补齐可见进度）
                bar.update(max(0, len(trajs) - last_done))
                bar.close()
            
                print(f"Part {part + 1} completed. Saving results...")
                sys.stdout.flush()
                with closing(Saver(os.path.join(save_root, "%d%02d-%d" % (year, month, part)))) as saver:
                    for traj in trajs:
                        saver.save(traj)
                print("Successfully saved this part.")
                sys.stdout.flush()
            
                # 清空，为下一个批次做准备
                trajs.clear()
                part += 1

            end_tasks()
            print('finished!')


