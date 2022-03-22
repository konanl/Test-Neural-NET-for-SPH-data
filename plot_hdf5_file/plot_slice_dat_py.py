#!/usr/bin/env python
# encoding: utf-8
"""This script read a HDF5 file and plots all particles."""

import h5py
from matplotlib import *
from numpy import *
from matplotlib.pyplot import *
from sys import *
import os

"""This is all for formatting the output."""
files = os.listdir('./')
files.sort()

for file in files():
    # Set a font
    rcParams['font.family'] = 'Arial'  # 字体属性
    rcParams['font.size'] = 10.0

    rcParams['legend.handlelength'] = 0.5  # 图例之间的长度,使用字体大小的几分之几表示
    rcParams['legend.frameon'] = False  # 是否在图例外显示外框
    rcParams['legend.numpoints'] = 1  # the number of points in the legend line
    rcParams['legend.scatterpoints'] = 1  # 为散点图图例条目创建的标记点数

    # Adjust axes line width
    rcParams['axes.linewidth'] = 0.5  # 边的宽

    # Adjust ticks
    rcParams['xtick.major.size'] = 4  # 最大刻度大小
    rcParams['xtick.minor.size'] = 2  # 最小刻度大小
    rcParams['ytick.major.size'] = 4
    rcParams['ytick.minor.size'] = 2

    # Adjust Font Size
    rcParams['xtick.labelsize'] = 'x-small'  # 刻度标签字体大小
    rcParams['ytick.labelsize'] = 'x-small'
    rcParams['axes.labelsize'] = 'small'  # x轴和y轴的字体大小

    # Set Up Figure, Single Column MNRAS
    fig = gcf()
    ax = gca()
    fig, ax = subplots(1, 1)

    fig.set_size_inches(8.27 * 0.39, 8.27 * 0.39)

    dot_size = 1.5
    line_width_thin = 0.001
    line_width_thick = 0.1

    """Read the SPH output."""
    filename = os.path.splitext(file)
    # find files which the suffix is hdf5
    if(filename[1]) == '.hdf5':
        path = os.path.join('./', file)
        num = round(int(filename[0][9:]) * 0.05, 2) # 保留两位小数
        bin = h5py.File(path)   # get the path for every file
        pos = bin["PartType0"]["Coordinates"][:]
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        rho = bin["PartType0"]["Density"][:]
        id = bin["PartType0"]["ParticleIDs"][:]

        RE = 6.370e8

        fil = where(abs(z) < 0.1 * RE)
        x_slice_i = []
        y_slice_i = []
        x_slice_j = []
        y_slice_j = []
        x_slice_k = []
        y_slice_k = []
        x_slice_m = []
        y_slice_m = []

        for k, i in enumerate(id):
            if k in fil[0]:
                index = where(id == i)[0][0]
                if 0 < i <= 32178:
                    x_slice_i.append(x[index])
                    y_slice_i.append(y[index])

                if 32178 < i <= 200000000:
                    x_slice_j.append(x[index])
                    y_slice_j.append(y[index])

                if 200000000 < i <= 200067822:
                    x_slice_k.append(x[index])
                    y_slice_k.append(y[index])

                if 200067822 < i:
                    x_slice_m.append(x[index])
                    y_slice_m.append(y[index])

    s1 = ax.scatter(x_slice_i, y_slice_i, s=dot_size, c='y', linewidth=line_width_thin)
    s2 = ax.scatter(x_slice_j, y_slice_j, s=dot_size, c='r', linewidth=line_width_thin)
    s3 = ax.scatter(x_slice_k, y_slice_k, s=dot_size, c='g', linewidth=line_width_thin)
    s4 = ax.scatter(x_slice_m, y_slice_m, s=dot_size, c='b', linewidth=line_width_thin)

    # xlim((-0.3e9,0.3e9))
    # ylim((-0.3e9,0.3e9))
    xlim((-2e9, 2e9))
    ylim((-2e9, 2e9))

    # ax.set_axisbelow(True) #网格显现在图形下方
    # grid()

    xlabel('X [cm]')
    ylabel('Y [cm]')

    ax.text(-1.5e9, 1.5e9, str(num) + 'hr', fontsize='8')

    savefig(file + '.slice.png', dpi=300, bbox_inches='tight')  # facecolor 设置背景颜色
    close()

