"""
This script read a HDF5 file and plots all particles.
"""
from matplotlib import *
from numpy import *
from matplotlib.pyplot import *
from sys import *
import os
# Used to read tipsy binary files
import pynbody

"""
This is all for formatting the output.
"""
files = os.listdir('./')
files.sort()
for  file in files:

# Set a font
    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 10.0
    rcParams['axes.unicode_minus']=False

    # Lighten labels
    # http://matplotlib.org/users/customizing.html
    # mpl.rcParams['axes.labelcolor']  = dgrey
    # mpl.rcParams['xtick.color']      = dgrey
    # mpl.rcParams['ytick.color']      = dgrey
    # mpl.rcParams['axes.edgecolor']   = dgrey

    # Legend
    # mpl.rcParams['legend.handlelength']  = 2.9
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
    # fig.set_size_inches(8.27*0.5,8.27*(10./13.)*0.5)
    fig.set_size_inches(8.27*0.39/2,8.27*0.39*(10/10.3)/2)

    """
    Read the file name from the command line.
    """
    # if len(argv) != 2:
    # 				print "Usage: plot_you.py <file>"
    # 				exit(1)
    #
    # file = argv[1]
    dot_size = 1.5
    line_width_thin =  0.001
    line_width_thick = 0.1

    """
    Read the SPH output.
    """

    filename = os.path.splitext(file)
    if filename[1] == '.hdf5':
        num = int(filename[0][9:])*0.05
        print num
        bin = pynbody.load(file)

        pos = bin.gas["pos"]
        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]
        rho = bin.gas["rho"]
        id = bin.gas["ParticleIDs"]

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


        xlim((-2e9, 2e9))
        ylim((-2e9, 2e9))
 #       ax.set_axis below(True)
#	grid()
        xlabel('X [cm]')
        ylabel('Y [cm]')
 #       ax.legend([s1,s2,s3,s4],labels=['Core1','Core2','Mantle1','Mantle2'],loc='upper right',fontsize='8')
 #       title('Disk Analysiser',y=1.05)
        ax.text(-1.7e9,1.7e9,str(num)+'hr',fontsize='8')
 #       show()
        savefig(file + 'slice.png', dpi=300, bbox_inches='tight')
        close()

"""
Plot the particles.
"""
#scatter(x,y,s=1,c=rho,cmap='rainbow',linewidth=0.0001)
#colorbar()


"""
Plot a slice through the model.
"""
"""
RE = 6.370e8

i = where(abs(z) < 0.1*RE)
print i
x_slice = x[i]
y_slice = y[i]
z_slice = z[i]

xlim((-3e9,3e9))
ylim((-3e9,3e9))

scatter(x_slice,y_slice,s=dot_size,c=rho[i],cmap='rainbow',linewidth=line_width_thin)
annotate('t=',xy=(.1,.1),xytext=(-2.5,2.5))
colorbar().set_label('colorBar')

xlabel('Radius [cm]')
ylabel('Radius [cm]')

#show()

# We need 300 dpi for the small format and 150 for the large one
savefig(file+'.slice.png', dpi=300, bbox_inches='tight')
close()

"""