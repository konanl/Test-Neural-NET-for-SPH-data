"""
This script read a HDF5 file and plots all particles.
"""
from matplotlib import *
from numpy import *
from matplotlib.pyplot import *
from sys import *

# Used to read tipsy binary files
import pynbody
import os

"""
This is all for formatting the output.
"""
# Set a font
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10.0

# Lighten labels
# http://matplotlib.org/users/customizing.html
# mpl.rcParams['axes.labelcolor']  = dgrey
# mpl.rcParams['xtick.color']      = dgrey
# mpl.rcParams['ytick.color']      = dgrey
# mpl.rcParams['axes.edgecolor']   = dgrey

# Legend
# mpl.rcParams['legend.handlelength']  = 2.9
files = os.listdir('./')
files.sort()
for  file in files:

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
		bin = pynbody.load(file)

		pos = bin.gas["pos"]
		x = pos[:,0]
		y = pos[:,1]
		z = pos[:,2]
		Temperature = bin.gas["Density"]

		"""
		Plot the particles.
		"""
		#scatter(x,y,s=1,c=rho,cmap='rainbow',linewidth=0.0001)
		#colorbar()

		"""
		Plot a slice through the model.
		"""
		RE = 6.37e8

		i = where(abs(z) < 0.1*RE)
		print i
		x_slice = x[i]
		y_slice = y[i]
		z_slice = z[i]

		xlim((-3e9,3e9))
		ylim((-3e9,3e9))
		ax.set_axisbelow(True)
		grid()

		scatter(x_slice,y_slice,s=dot_size,c=Temperature[i],cmap='rainbow',linewidth=line_width_thin)

		cbar = colorbar()
	#	clim(0, 15)

		xlabel('Radius [cm]')
		ylabel('Radius [cm]')
		

		#show()

		# We need 300 dpi for the small format and 150 for the large one
	#	ax.text(-0.9e9, 0.9e9, str(num) + 'hr', fontsize='8')
		savefig(file+'.slice.png', dpi=300, bbox_inches='tight')
		close()
