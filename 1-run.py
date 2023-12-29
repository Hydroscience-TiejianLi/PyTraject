from __future__ import division, print_function
import sys
import os
from pyTraj.task import TaskManager_area
from pyTraj.task import TaskManager_points
import multiprocessing as mp
folder_path = r'/media/being/P/forecast/era5/'
mp.freeze_support()

manager = TaskManager_points(
    folder_path, r'/media/being/P/forecast/era5/North_China2023.npy', traj_step=-1, duration=240, month_buffer=1
)
manager.run(r'/media/being/P/forecast/era5', batch_size=1e5, fast=1000, data_type='float16')

