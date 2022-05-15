"""
Example how to use pynbody.
"""

# this code can get parameters from the linux keyboard
# 从键盘输入文件名直接画出相应的图，每次只能画一张图，不能批处理

from __future__ import print_function

import sys
import numpy
import os
import h5py
import pynbody
import matplotlib.pyplot as plt
import matplotlib as mpl

# Chose your filename here
# the filename is from the keyboard
# assert sys.argv[1]
if len(sys.argv) != 2:
    print("Usage: plot_you.py <file>")

print("Read the file name from the command line.")
filename = sys.argv[1]
#print(len(sys.argv))

#print(filename)

# Read tipsy binary file using pynbody
bin = pynbody.tipsy.TipsySnap(filename)

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

#plt.grid()
#ax.set_axisbelow(True)
#plt.xlim((-15e0,15e0))
#plt.ylim((-15e0,15e0))

# Make a scatter plot
plt.scatter(x[idx], y[idx], s=1, c=den[idx], cmap=plt.cm.rainbow)

plt.colorbar()

plt.xlabel("x [R$_{\oplus}$]")
plt.ylabel("y [R$_{\oplus}$]")

plt.title("M-ANEOS")
#plt.legend(loc='best')

plt.savefig(filename+".png", dpi=300, bbox_inches='tight')
plt.show() 
    
#exit(0)

hdfFile = h5py.File(filename+'.hdf5', 'w')
hdfFile.create_dataset('pos', data=pos)
hdfFile.create_dataset('vel', data=vel)
hdfFile.create_dataset('mass', data=mass)
hdfFile.create_dataset('density', data=den)
hdfFile.create_dataset('temp', data=temp)
#hdfFile.create_dataset('press', data=press)

hdfFile.close()