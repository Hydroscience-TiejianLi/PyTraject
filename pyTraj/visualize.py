# -*- encoding: utf-8 -*-

from __future__ import division, print_function
import math
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from .graph import Graph


def evapor_interval(vmax):
    order = math.floor(math.log10(vmax))
    min_int = 10 ** (order - 2)

    def ifloat(num):
        return math.floor(num / min_int) * min_int

    step = math.ceil(vmax / 56 / min_int) * min_int

    int0 = ifloat(0)
    int3 = ifloat(vmax)
    int2 = ifloat((int0 + int3) / 2)
    int1 = ifloat((int0 + int2) / 2)

    intervals = list(np.arange(int0, int1 - step * 0.1, step * 2))
    intervals += list(np.arange(int1, int2 - step * 0.1, step * 4))
    intervals += list(np.arange(int2, int3, step * 8))

    intervals.insert(1, step)

    return tuple(intervals)


def density_interval(vmax):
    order = math.floor(math.log10(vmax))
    min_int = max(10 ** (order - 2), 1)

    def ifloat(num):
        return math.floor(num / min_int) * min_int

    step = math.ceil(vmax / 56 / min_int) * min_int

    int0 = ifloat(0)
    int2 = ifloat(vmax)
    int1 = ifloat((int0 + int2) / 2)

    intervals = list(np.arange(int0, int1 - step * 0.1, step * 2))
    intervals += list(np.arange(int1, int2, step * 4))

    intervals.insert(1, 1)
    return tuple(intervals)


def draw_basemap(ax, extent=None, gridline=None, ocean=False, river=True):
    crs = ax.projection
    if extent:
        ax.set_extent(extent, crs=crs)
    ax.coastlines(zorder=3)
    if river:
        ax.add_feature(cfeature.LAKES)
        ax.add_feature(cfeature.RIVERS)
    if ocean:
        ax.add_feature(cfeature.OCEAN)

    if gridline:
        x0, x1, y0, y1 = extent
        ax.gridlines(ylocs=range(-90, 91, gridline), xlocs=range(-180, 181, gridline))
        x0 = math.ceil(x0 // gridline) * gridline
        x1 = math.floor(x1 // gridline) * gridline
        y0 = math.ceil(y0 // gridline) * gridline
        y1 = math.floor(y1 // gridline) * gridline
        ax.set_xticks(range(x0, x1+1, gridline), crs=crs)
        ax.set_yticks(range(y0, y1+1, gridline), crs=crs)

        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)


def draw_shaded(ax, amap, lats, lons, vmax=None, intervals='evapor', cmap=None):
    crs = ax.projection
    #intervals = list(np.arange(0.005, 0.07, 0.01)) + list(np.arange(0.07, 0.15, 0.02)) + list(np.arange(0.15, 0.3, 0.04))
    if isinstance(intervals, str):
        if vmax is None:
            vmax = np.nanmax(amap)
        if intervals == 'evapor':
            intervals = evapor_interval(vmax)
            if not cmap:
                cmap = 'gnuplot2_r'
        elif intervals == 'density':
            intervals = density_interval(vmax)
            if not cmap:
                cmap = 'gnuplot_r'
        # print('levels: ', intervals)
    cf = plt.contourf(lons, lats, amap, transform=crs, cmap=cmap, extend='max', levels=intervals)
    cf.collections[0].set_alpha(0)
    cf.collections[1].set_alpha(0.5)
    return cf

def draw_img(ax, img, lats, lons):
    crs = ax.projection

def draw_lines(ax, bound, color='black', **styles):
    ax.add_geometries(bound, ax.projection, edgecolor=color, facecolor='none', **styles)


def make_segments(xy):
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def draw_trajs(ax, trajs, ratio=1, color='blue', lw=0.1, vmax=None, **styles):
    if color in ('height', 'last_height'):
        cmap = matplotlib.cm.get_cmap('jet')
        norm = matplotlib.colors.Normalize(vmin=20000, vmax=100000)
    elif color in ('humidity'):
        cmap = 'gnuplot_r'
        norm = matplotlib.colors.Normalize(vmin=0.004, vmax=0.015)
    elif color in ('const_h', 'linear_h', 'humidity_h'):
        cmap = 'gnuplot_r'
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        graph = Graph()
        graph._trajs = trajs
        graph._cal_traj_precips()
        method = color.split('_')[0]
        if method == 'const':
            h_method = graph._const_h_at
        elif method == 'linear':
            h_method = graph._linear_h_at
        elif method == 'humidity':
            h_method = graph._humidity_h_at()

    mapable = None
    for tid in range(trajs.size):
        if np.random.rand() < ratio:
            size = trajs.tsize[tid]
            xyzq = trajs.xyzq[tid, :size]

            if color == 'height':
                z = xyzq[:, 2]
                segments = make_segments(xyzq[:, 1::-1])
                mapable = LineCollection(segments, array=z, transform=ax.projection, cmap=cmap, norm=norm, lw=lw, **styles)
                ax.add_collection(mapable)

            elif color == 'humidity':
                z = xyzq[:, 3]
                segments = make_segments(xyzq[:, 1::-1])
                mapable = LineCollection(segments, array=z, transform=ax.projection, cmap=cmap, norm=norm, lw=lw, **styles)
                ax.add_collection(mapable)

            elif color in ('const_h', 'linear_h', 'humidity_h'):
                z = np.array([h_method((tid, j)) for j in range(size)])
                segments = make_segments(xyzq[:, 1::-1])
                mapable = LineCollection(segments, array=z, transform=ax.projection, cmap=cmap, norm=norm, lw=lw, **styles)
                ax.add_collection(mapable)

            else:
                if color == 'last_height':
                    c = cmap(norm(xyzq[0, 2]))
                    if not mapable:
                        mapable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                        mapable.set_array([])
                else:
                    c = color
                ax.plot(xyzq[:, 1], xyzq[:, 0], linestyle='-', color=c, transform=ax.projection, lw=lw, **styles)

    return mapable


def draw_graph(graph: Graph, ax, cmap='gnuplot_r', max_width=3, **styles):
    segments = []
    weights = []
    max_weight = 0
    min_weight = 0
    for edge in graph.edges:
        vertex_start = graph.vertice[edge.start_vid]
        vertex_end = graph.vertice[edge.end_vid]
        reverse_edge = vertex_end.edges.get(edge.start_vid, None)
        if reverse_edge and reverse_edge.weight > edge.weight:
            continue
        x1, y1 = vertex_start.lon, vertex_start.lat
        x2, y2 = vertex_end.lon, vertex_end.lat
        if x1 < x2 - 200:
            x2 -= 360
        elif x1 > x2 + 200:
            x2 += 360
        segments.append([[x1, y1],[x2, y2]])
        weights.append(edge.weight)
        max_weight = max(max_weight, edge.weight)
        min_weight = min(min_weight, edge.weight)
    segments = np.array(segments)
    weights = np.array(weights)

    norm = matplotlib.colors.LogNorm(vmin=max(min_weight, max_weight/1000), vmax=max_weight)
    width = weights / weights.max() * max_width
    mapable = LineCollection(segments, array=weights, transform=ax.projection, cmap=cmap, norm=norm, lw=width, **styles)
    ax.add_collection(mapable)
    return mapable