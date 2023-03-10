from __future__ import print_function

from matplotlib.pyplot import *
from matplotlib import *
from sys import *
import numpy
import os
import h5py
import pynbody
import matplotlib.pyplot as plt
import matplotlib as mpl


# Chose your filename here
files = os.listdir('./')
files.sort()

# Set a font
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10.0

for file in files:
    rcParams['legend.handlelength']  = 0.5
    rcParams['legend.frameon']       = False
    rcParams['legend.numpoints']     = 1
    rcParams['legend.scatterpoints'] = 1

    # Adjust axes line width
    rcParams['axes.linewidth']   = 0.5

    # Adjust ticks
    rcParams['xtick.major.size'] = 4
    rcParams['xtick.minor.size'] = 2
    rcParams['ytick.major.size'] = 4
    rcParams['ytick.minor.size'] = 2

    # Adjust Font Size
    rcParams['xtick.labelsize']  = 'x-small'
    rcParams['ytick.labelsize']  = 'x-small'
    rcParams['axes.labelsize']   = 'small'

    # Set Up Figure, Single Column MNRAS
    fig = gcf()
    ax = gca()
    fig, ax = subplots(1,1)
    
    # Plot two plots in one (w,h)
    fig.set_size_inches(8.27*0.5,8.27*(10./13.)*0.5)
    fig.set_size_inches(8.27*0.39,8.27*(10./13.)*0.39)
    
    dot_size = 1.5
    line_width_thin =  0.001
    line_width_thick = 0.1
    
    filename = os.path.splitext(file)
    if filename[1] == '.std':

        bin = pynbody.tipsy.TipsySnap(file)
        
        pos  = bin['pos'] 
        vel  = bin['vel']
        mass = bin['mass']
        den  = bin['rho']*0.368477
        temp  = bin['temp']
        #energy  = bin['energy']
        #press  = bin['press']
        #soundspeed  = bin['soundspeed']
        #mat  = bin['mat']
        
        N = numpy.size(pos[:,0])

        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        den_min = numpy.min(den)
        den_max = numpy.max(den)

        # Sort the particles according to density
        idx = den.argsort()

        print(den[idx])

        plt.grid()
        #ax.set_axisbelow(True)
        #plt.xlim((-15e0,15e0))
        #plt.ylim((-15e0,15e0))

        # Make a scatter plot
        plt.scatter(x[idx], y[idx], s=dot_size, c=den[idx], cmap=plt.cm.rainbow, linewidth=line_width_thin)

        #plt.colorbar()

        #plt.xlabel("x [R$_{\oplus}$]")
        #plt.ylabel("y [R$_{\oplus}$]")

        #plt.title("M-ANEOS")
        #plt.legend(loc='best')

        plt.savefig(filename[0] + ".png", dpi=300, bbox_inches='tight')
        plt.show() 

        #exit(0)

        hdfFile = h5py.File(filename[0]+'.hdf5', 'w')
        hdfFile.create_dataset('pos', data=pos)
        hdfFile.create_dataset('vel', data=vel)
        hdfFile.create_dataset('mass', data=mass)
        hdfFile.create_dataset('density', data=den)
        hdfFile.create_dataset('temp', data=temp)
        #hdfFile.create_dataset('press', data=press)
        hdfFile.close()
