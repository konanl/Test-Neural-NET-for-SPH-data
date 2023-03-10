from __future__ import print_function

# from matplotlib.pyplot import *
# from matplotlib import *
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
    plt.rcParams['legend.handlelength']  = 0.5
    plt.rcParams['legend.frameon']       = False
    plt.rcParams['legend.numpoints']     = 1
    plt.rcParams['legend.scatterpoints'] = 1

    # Adjust axes line width
    plt.rcParams['axes.linewidth']   = 0.5

    # Adjust ticks
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['ytick.minor.size'] = 2

    # Adjust Font Size
    plt.rcParams['xtick.labelsize']  = 'x-small'
    plt.rcParams['ytick.labelsize']  = 'x-small'
    plt.rcParams['axes.labelsize']   = 'small'

    # Set Up Figure, Single Column MNRAS
    fig = plt.gcf()
    ax = plt.gca()
    fig, ax = plt.subplots(1,1)
    
    # Plot two plots in one (w,h)
    fig.set_size_inches(8.27*0.5,8.27*(10./13.)*0.5)
    fig.set_size_inches(8.27*0.39,8.27*(10./13.)*0.39)
    
    dot_size = 1.5
    line_width_thin =  0.001
    line_width_thick = 0.1
    
    filename = os.path.splitext(file)
    #print(filename[1][1:])
    if filename[1][1:].isdigit() or filename[1] == '.std':
        #num = int(filename[1][1:])*0.05
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

        ax.set_axisbelow(True)
        plt.grid()
        plt.xlim((-3e0,3e0))
        plt.ylim((-3e0,3e0))

        # Make a scatter plot
        plt.scatter(x[idx], y[idx], s=dot_size, c=den[idx], cmap=plt.cm.rainbow, linewidth=line_width_thin)
        plt.colorbar()

        plt.xlabel("x [R$_{\oplus}$]")
        plt.ylabel("y [R$_{\oplus}$]")

        plt.title("M-ANEOS")
        #plt.legend(loc='best')
        
        if filename[1] == '.std':
            #ax.text(-0.9e9, 0.9e9, str(num) + 'hr', fontsize='8')
            plt.savefig(filename[0] + ".png", dpi=300, bbox_inches='tight')
            #plt.show() 
            #exit(0)

            hdfFile = h5py.File(filename[0]+'.hdf5', 'w')
        elif filename[1][1:].isdigit():
            #ax.text(-0.9e9, 0.9e9, str(num) + 'hr', fontsize='8')
            plt.savefig(file+'.png',dpi=300, bbox_inches='tight')
            #plt.show() 
            #exit(0)

            hdfFile = h5py.File(file+'.hdf5', 'w')

        hdfFile.create_dataset('pos', data=pos)
        hdfFile.create_dataset('vel', data=vel)
        hdfFile.create_dataset('mass', data=mass)
        hdfFile.create_dataset('density', data=den)
        hdfFile.create_dataset('temp', data=temp)
        #hdfFile.create_dataset('press', data=press)
        hdfFile.close()
    elif filename[1] == '.temp': # text file 
         temp = numpy.loadtxt(file)[1:]
         for file in files:
            # filename_ = os.path.splitext(file)
            if filename[0] == file:
                bin = pynbody.tipsy.TipsySnap(file)
                pos  = bin['pos']

                N = numpy.size(pos[:,0])

                x = pos[:, 0]
                y = pos[:, 1]
                z = pos[:, 2]

                temp_min = numpy.min(temp)
                temp_max = numpy.max(temp)

                # Sort the particles according to temp
                idx = temp.argsort()

                print(temp[idx])
                
                ax.set_axisbelow(True)
                plt.grid()
                plt.xlim((-3e0,3e0))
                plt.ylim((-3e0,3e0))

                # Make a scatter plot
                plt.scatter(x[idx], y[idx], s=dot_size, c=temp[idx], cmap=plt.cm.rainbow, linewidth=line_width_thin)
                plt.colorbar()

                plt.xlabel("x [R$_{\oplus}$]")
                plt.ylabel("y [R$_{\oplus}$]")

                plt.title("M-ANEOS")
                #plt.legend(loc='best')

                #ax.text(-0.9e9, 0.9e9, str(num) + 'hr', fontsize='8')
                plt.savefig(file+'.png',dpi=300, bbox_inches='tight')
                #plt.show() 
                #exit(0)
                hdfFile = h5py.File(file+'.hdf5', 'w')
                hdfFile.create_dataset('temp', data=temp)
                hdfFile.create_dataset('pos', data=pos)
                hdfFile.close()









            