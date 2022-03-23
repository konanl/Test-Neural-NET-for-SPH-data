#!/usr/bin/env python
# encoding: utf-8
"""This script read a HDF5 file and plots all particles depend on their phase"""

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

        count = []  # 判断是老数据还是新数据对指标
        for key in bin:
            count.append(key)
        if(len(count) == 4):
            pos = bin['pos'][:]
            x = pos[0,:]
            y = pos[1,:]
            z = pos[2,:]

            rho = bin['density'][:]
            pha = bin['phase'][:]

            RE = 6.370e8

            i = where(abs(z) < 0.1 * RE)
            print(i)
            x_slice = x[i]
            y_slice = y[i]
            z_slice = z[i]

            # Separate different particle coordinates according to phase
            x_slice1 = []
            y_slice1 = []
            x_slice2 = []
            y_slice2 = []
            x_slice3 = []
            y_slice3 = []
            x_slice4 = []
            y_slice4 = []
            x_slice5 = []
            y_slice5 = []
            x_slice6 = []
            y_slice6 = []
            x_slice7 = []
            y_slice7 = []

            for k in range(len(pha)):
                if(int(pha[k].item()) == 1):
                   x_slice1.append(x[k])
                   y_slice1.append(y[k])
                if (int(pha[k].item()) == 2):
                    x_slice2.append(x[k])
                    y_slice2.append(y[k])
                if (int(pha[k].item()) == 3):
                    x_slice3.append(x[k])
                    y_slice3.append(y[k])
                if (int(pha[k].item()) == 4):
                    x_slice4.append(x[k])
                    y_slice4.append(y[k])
                if (int(pha[k].item()) == 5):
                    x_slice5.append(x[k])
                    y_slice5.append(y[k])
                if (int(pha[k].item()) == 6):
                    x_slice6.append(x[k])
                    y_slice6.append(y[k])
                if (int(pha[k].item()) == 7):
                    x_slice7.append(x[k])
                    y_slice7.append(y[k])

            # phase
            s1 = ax.scatter(x_slice1, y_slice1, s=dot_size, c='blue', linewidth=line_width_thin)
            s2 = ax.scatter(x_slice2, y_slice2, s=dot_size, c='green', linewidth=line_width_thin)
            s3 = ax.scatter(x_slice3, y_slice3, s=dot_size, c='orange', linewidth=line_width_thin)
            s4 = ax.scatter(x_slice4, y_slice4, s=dot_size, c='purple', linewidth=line_width_thin)
            s5 = ax.scatter(x_slice5, y_slice5, s=dot_size, c='slategray', linewidth=line_width_thin)
            s6 = ax.scatter(x_slice6, y_slice6, s=dot_size, c='red', linewidth=line_width_thin)
            s7 = ax.scatter(x_slice7, y_slice7, s=dot_size, c='gray', linewidth=line_width_thin)

            # xlim((200, 300))
            xlim((260, 275))
            ylim(-100, -50)
            # set grid()
            # ax.set_axisbelow(True)
            # grid()
            # xlabel('Radius [cm]')
            # ylabel('Radius [cm]')
            xlabel('X [cm]')
            ylabel('Y [cm]')
            ax.legend([s1,s2,s3,s4,s5,s6,s7],labels=['1', '2', '3', '4', '5', '6', '7'],
                      loc='upper right', fontsize='xx-small', frameon=True)
            title('Phase Analysiser', y=1.05)
            # show()
            # We need 300 dpi for the small format and 150 for the large one
            savefig(file + '.slice.phase.png', dpi=300, bbox_inches='tight')
            # savefig(file + '.slice.density.png', dpi=300, bbox_inches='tight')
            close()
