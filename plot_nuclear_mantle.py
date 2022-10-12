from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib as mpl
from sys import *
import numpy
import os
import h5py
import pynbody

# Chose your filename here
files = os.listdir('./')
files.sort()

# Set a font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10.0

for file in files:
    plt.rcParams['legend.handlelength'] = 0.5
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['legend.scatterpoints'] = 1

    # Adjust axes line width
    plt.rcParams['axes.linewidth'] = 0.5

    # Adjust ticks
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['ytick.minor.size'] = 2

    # Adjust Font Size
    plt.rcParams['xtick.labelsize'] = 'x-small'
    plt.rcParams['ytick.labelsize'] = 'x-small'
    plt.rcParams['axes.labelsize'] = 'small'

    # Set Up Figure, Single Column MNRAS
    fig = plt.gcf()
    ax = plt.gca()
    fig, ax = plt.subplots(1, 1)

    # Plot two plots in one (w,h)
    fig.set_size_inches(8.27 * 0.5, 8.27 * (10. / 13.) * 0.5)
    fig.set_size_inches(8.27 * 0.39, 8.27 * (10. / 13.) * 0.39)

    dot_size = 1.5
    line_width_thin = 0.001
    line_width_thick = 0.1

    filename = os.path.splitext(file)
    
    if filename[1] == '.material':
        # Read the file
        material = numpy.loadtxt(file)[1:]
        for file_ in files:
            if filename[0] == file_:

                bin = pynbody.tipsy.TipsySnap(file_)

                pos = bin['pos']

                N = numpy.size(pos[:, 0])

                # two part 63, 62
                num = material[0]
                part1x, part2x = [], []
                part1y, part2y = [], []
                part1z, part2z = [], []
                pointer = 0
                # pointer-> material, pointer->pos
                while pointer < len(material) and pointer < N:
                    if material[pointer] == num:
                        part1x.append(pos[pointer][0])
                        part1y.append(pos[pointer][1])
                        part1z.append(pos[pointer][2])
                    else:
                        part2x.append(pos[pointer][0])
                        part2y.append(pos[pointer][1])
                        part2z.append(pos[pointer][2])
                    pointer += 1

                ax.set_axisbelow(True)
                # plt.grid()
                plt.xlim((-3e0, 3e0))
                plt.ylim((-3e0, 3e0))
                
                print(len(part1x))
                print(len(part2x))
                
                # zorder 参数是为了消除遮盖问题：核部分会被遮盖，数值越大的会被优先显示，所以part1储存核的坐标，part2储存幔的坐标
                plt.scatter(part1x, part1y, s=dot_size, color='red', linewidth=line_width_thin, zorder=2)
                plt.scatter(part2x, part2y, s=dot_size, color='green', linewidth=line_width_thin, zorder=1)

                plt.xlabel("x [R$_{\oplus}$]")
                plt.ylabel("y [R$_{\oplus}$]")

                plt.title("M-ANEOS")

                plt.savefig(file + '.png', dpi=300, bbox_inches='tight')

