#!/usr/bin/env python
# encoding: utf-8

"""This script read a HDF5 file and plots all particles with gradient color for some value"""

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
        num = round(int(filename[0][9:]) * 0.05, 2) # 保留两位小数
        bin = h5py.File(path)   # get the path for every file
        pos = bin["PartType0"]["Coordinates"][:]
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        rho = bin["PartType0"]["Density"][:]
        id = bin["PartType0"]["ParticleIDs"][:]

        RE = 6.370e8
        i = where(abs(z) < 0.1 * RE)
        print(i)
        x_slice = x[i]
        y_slice = y[i]
        z_slice = z[i]
        xlim((-3e9, 3e9))
        ylim((-3e9, 3e9))

        # Density
        scatter(x_slice, y_slice, s=dot_size, c=rho[i], cmap='rainbow', linewidth=line_width_thin)
        colorbar().set_label('Density')

        # InternalEnergy
        # ie = bin["PartType0"]["InternalEnergy"][:]
        # scatter(x_slice, y_slice, s=dot_size, c=ie[i], cmap='rainbow', linewidth=line_width_thin)
        # colorbar().set_label('InternalEnergy')

        # Masses
        # mass = bin["PartType0"]["Masses"][:]
        # scatter(x_slice, y_slice, s=dot_size, c=mass[i], cmap='rainbow', linewidth=line_width_thin)
        # colorbar().set_label('Masses')

        # Potential
        # pot = bin["PartType0"]["Potential"][:]
        # scatter(x_slice, y_slice, s=dot_size, c=pot[i], cmap='rainbow', linewidth=line_width_thin)
        # colorbar().set_label('Potential')

        # Velocities
        # vel = bin["PartType0"]["Velocities"][:]
        # x vel
        # scatter(x_slice, y_slice, s=dot_size, c=vel[i,0], cmap='rainbow', linewidth=line_width_thin)
        # y val
        # scatter(x_slice, y_slice, s=dot_size, c=vel[i,1], cmap='rainbow', linewidth=line_width_thin)
        # z vel
        # scatter(x_slice, y_slice, s=dot_size, c=vel[i,2], cmap='rainbow', linewidth=line_width_thin)
        # colorbar().set_label('Velocities')


        xlabel('Radius [cm]')
        ylabel('Radius [cm]')
        # show()
        # We need 300 dpi for the small format and 150 for the large one
        savefig(file + '.slice.gra_col.png', dpi=300, bbox_inches='tight')
        close()
