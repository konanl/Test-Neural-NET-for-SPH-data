#!/usr/bin/env python
# encoding: utf-8
"""This script read a HDF5 file and plots all particles depend on their density"""
import h5py
from matplotlib import *
from numpy import *
from matplotlib.pyplot import *
from sys import *
import os

"""This is all for formatting the output."""
files = os.listdir('./')
files.sort()

for file in files:
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
    # fig, ax = subplots(1, 1)
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
        bin = h5py.File(path)   # get the path for every file
        RE = 6.370e8

        count = []  # 判断是老数据还是新数据对指标
        for key in bin:
            count.append(key)
        if len(count) == 2:  # old database
            pos = bin["PartType0"]["Coordinates"][:]
            x = pos[:, 0]
            y = pos[:, 1]
            z = pos[:, 2]
            rho = bin["PartType0"]["Density"][:]
            id = bin["PartType0"]["ParticleIDs"][:]

            i = where(abs(z) < 0.1 * RE)
            x_slice = x[i]
            y_slice = y[i]
            z_slice = z[i]
            xlim((-3e9, 3e9))
            ylim((-3e9, 3e9))
            # Density
            scatter(x_slice, y_slice, s=dot_size, c=rho[i], cmap='rainbow', linewidth=line_width_thin)
            colorbar().set_label('Density')
            xlabel('Radius [cm]')
            ylabel('Radius [cm]')
            title('Density Analysiser', y=1.05)
            # We need 300 dpi for the small format and 150 for the large one
            savefig(file + '.slice.gra_col.Density.png', dpi=300, bbox_inches='tight')
            close()
        if len(count) == 4:  # new database
            pos = bin['pos'][:]
            x = pos[0, :]
            y = pos[1, :]
            z = pos[2, :]
            rho = bin['density'][:]
            pha = bin['phase'][:]

            i = where(abs(z) < 0.1 * RE)
            x_slice = x[i]
            y_slice = y[i]
            z_slice = z[i]
            xlim((260, 275))
            ylim((-80, -60))
            # density
            scatter(x_slice, y_slice, s=dot_size, c=rho[i], cmap='rainbow', linewidth=line_width_thin)
            colorbar().set_label('density')
            xlabel('Radius')
            ylabel('Radius')
            title('Density Analysiser', y=1.05)
            savefig(file + '.slice.density.png', dpi=300, bbox_inches='tight')
            close()
