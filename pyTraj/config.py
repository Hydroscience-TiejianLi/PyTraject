# -*- encoding: utf-8 -*-

from __future__ import division, print_function
from datetime import datetime
import numpy as np


class Config(object):

    # time
    TIME_BASELINE = datetime(1975, 1, 1, 0)
    TIME_RANGE = [datetime(1975, 1, 1), datetime(2021, 9, 30, 23)]
    # vertical range
    PRESSURE_TOP = 20000
    PRESSURE_BOTTOM = 100000
    #BOUND_TOP = 20100
    # 气压分层
    DEFAULT_ZS = np.arange(25000, 100000, 5000)

    # 地球半径和重力加速度，不同的数据集，这两项数据有可能不同
    RADIUS = 6367470 # ERA5 surface and single level and pressure level
    GRAVE_ACC = 9.80665
    # Parallelization，should lower than the number of CPU
    NUM_WORKERS = 100
    # input data grid
    GRID = (0.25, 1) # 0.75°×0.75°，6h
    LATS = np.arange(90.0,-90.25,-0.25)
    LONS = np.arange(0,360,0.25)
    PRESS_LEVELS = np.arange(20000, 105000, 5000)
