import numpy as np
from netCDF4 import *
import os
import datetime
from calendar import monthrange
from .config import Config

# 解析NC 文件 返回不同气压层的风场和比湿场以及地表气压场
# 根据经纬度和气压高确定数据的shape
lats = Config.LATS
lons = Config.LONS
levels = Config.PRESS_LEVELS
variables = ['V component of wind', 'U component of wind', 'Vertical velocity', 'Z', 'Specific humidity']
                 #风的V分量               #风的U分量             #垂直速度            #地球重力位势       #比湿度
variables_2d = ['Surface pressure']
                 #地表气压
# 解析nc 指定的nc 文件，返回解析出来的数据集
def parserNC(ncpath, dtype='float32'):
    dataset_v = Dataset(ncpath+'_v_component_of_wind.nc')
    dataset_u = Dataset(ncpath+'_u_component_of_wind.nc')
    dataset_w = Dataset(ncpath+'_vertical_velocity.nc')
    dataset_z = Dataset(ncpath+'_geopotential.nc')
    dataset_q = Dataset(ncpath+'_specific_humidity.nc')
    #dataset_q_l = Dataset(ncpath+'_specific_cloud_liquid_water_content.nc')
    #dataset_q_i = Dataset(ncpath+'_specific_cloud_ice_water_content.nc')
    #根据nc名称解析出年月并初始化出一个存储nc文件中数据的数组 data
    filename = os.path.basename(ncpath).split('.')[0]
    year = int(filename[:4])
    month = int(filename[4:])
    days = monthrange(year,month)[1]
    #解析出存储的时间 并给出 键值对数组
    t = datetime.datetime(year,month,1)
    times = [t,]
    delta = datetime.timedelta(hours=1)
    for i in range(days*24//Config.GRID[1]-1):
        t = t + delta
        times.append(t)
    
    data = np.zeros((days*24//Config.GRID[1],len(levels),len(lats),len(lons),5),dtype=dtype)

    data[:, :, :, :, 0] = dataset_v.variables['v'][:][:, :, :, :]
    data[:, :, :, :, 1] = dataset_u.variables['u'][:][:, :, :, :]
    data[:, :, :, :, 2] = dataset_w.variables['w'][:][:, :, :, :]
    data[:, :, :, :, 3] = dataset_z.variables['z'][:][:, :, :, :]  /  Config.GRAVE_ACC # z
    data[:, :, :, :, 4] = dataset_q.variables['q'][:][:, :, :, :] 
    #data[:, :, :, :, 4] += dataset_q_l.variables['clwc'][:][:, :, :, :]
    #data[:, :, :, :, 4] += dataset_q_i.variables['ciwc'][:][:, :, :, :]
    data = data[:, ::-1, :, :]
    return times,levels,lons,lats,variables,data

#解析2dnc 指定nc 文件，返回解析出来的数据集
def parserNC_2d(ncpath, dtype='float32'):
    dataset = Dataset(ncpath)
    #根据nc名称解析出年月并初始化出一个存储nc文件中数据的数组 data_2d
    filename = os.path.basename(ncpath).split('.')[0]
    year = int(filename[:4])
    month = int(filename[4:6])
    days = monthrange(year,month)[1]
    #解析出存储的时间 并给出 键值对数组
    t = datetime.datetime(year,month,1)
    times = [t,]
    delta = datetime.timedelta(hours=1)
    for i in range(days*24//Config.GRID[1]-1):
        t = t + delta
        times.append(t)

    data_2d = np.zeros((days*24//Config.GRID[1],len(lats),len(lons),1),dtype=dtype)

    data_2d[:, :, :, 0] = dataset.variables['sp'][:][:, :, :] / 100 # hPa
    data = data[:, ::-1, :, :]
    return times,lons,lats,variables_2d,data_2d
