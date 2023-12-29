# -*- encoding: utf-8 -*-

from __future__ import division, print_function

class DataEngineBase(object):

    def wind_at(self, lat, lon, z, t):
        raise NotImplementedError()

    def humidity_at(self, lat, lon, z, t):
        raise NotImplementedError()

    def surface_z_at(self, lat, lon, t):
        raise NotImplementedError()

    def prepare_for(self, year, month, buffer_early, buffer_late):
        raise NotImplementedError()


class ParallelDataEngine(DataEngineBase):

    share_variables = {}

    @classmethod
    def fork(cls, **kwargs):
        raise NotImplementedError()
