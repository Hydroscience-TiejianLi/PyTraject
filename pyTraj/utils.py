# -*- encoding: utf-8 -*-

from __future__ import division, print_function
from datetime import datetime, timedelta
from .config import Config


def time2hour(time, base=Config.TIME_BASELINE):
    return (time - base).total_seconds() / 3600


def hour2time(t_in_hour, base=Config.TIME_BASELINE):
    return base + timedelta(hours=t_in_hour)


def convert_time_base(thour, from_base: datetime, to_base: datetime):
    delta = (from_base - to_base).total_seconds() / 3600
    return thour + delta
