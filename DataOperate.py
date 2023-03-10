import h5py
import numpy as np
import os

# Get all file 

files = os.listdir('./')
files.sort()

def GetFirstElem(elem):
    return elem[0]

for file in files:
    filename = os.path.splitext(file)    # spilt the filename and the postfix
    # print(filename)
    if filename[1] == '.hdf5':
        path = os.path.join('./'+file)   # get the path of the file
        
        f = h5py.File(file)
        
        g = f['PartType0']
        
        ids = g['ParticleIDs'][:]
        mass = g['Masses'][:]
        pos = g['Coordinates'][:]
        vel = g['Velocities'][:]
        rho = g['Density'][:]
        ie = g['InternalEnergy'][:]
        pot = g['Potential'][:]
        sl = g['SmoothingLength'][:]
        
        data = []
        
        for i in range(len(ids)):
            a = [[ ids[i], 
                   mass[i], 
                   rho[i], 
                   ie[i], 
                   pot[i], 
                   sl[i], 
                   pos[i][0], pos[i][1], pos[i][2], 
                   vel[i][0], vel[i][1], vel[i][2], 
                 ]]
            data.extend(a)
        data.sort(key = GetFirstElem)
        data = np.array(data)
        data = data.astype(np.float32)
            
        dataset = h5py.File('../Datasets/'+file, 'w')
        dataset.flush()

        dataset.create_dataset('dataset', data=data)
        dataset.close()
