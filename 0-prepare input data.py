import sys
import time
import numpy as np

from pyTraj.parser import parserNC, parserNC_2d
from pyTraj.data import EraDataEngine

for month in range(8, 9):

    s = time.time()

    compressed_nc = r'D:\Onion\Traj_era5\data\2018%02d'%month
    compressed_nc_2d = r'D:\Onion\Traj_era5\data\2018%02d_2d.nc'%month

    data_type = 'float16'


    print('Reading %s ...'%compressed_nc, end='')
    sys.stdout.flush()

    times_inNC, levels_inNC, lons_inNC, lats_inNC, variables, data_inNC = parserNC(compressed_nc, dtype=data_type)
    times_inNC, lons_inNC, lats_inNC, variables_2d, data_inNC_2d = parserNC_2d(compressed_nc_2d, dtype=data_type)

    np.savez(r'data\2018%02d_3d_%s.npz'%(month, data_type), times_inNC=times_inNC, levels_inNC=levels_inNC, lons_inNC=lons_inNC, lats_inNC=lats_inNC, variables=variables, data_inNC=data_inNC)
    np.savez(r'data\2018%02d_2d_%s.npz'%(month, data_type), times_inNC=times_inNC, lons_inNC=lons_inNC, lats_inNC=lats_inNC, variables_2d=variables_2d, data_inNC_2d=data_inNC_2d)

    print('\n%.3fs, prepare compltete.'%(time.time() - s))


