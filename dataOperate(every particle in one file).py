import h5py
import numpy as np
import os
import sys
import argparse


# Get information from keyboard
# ===================
# ===== way one =====
# ===================
# if len(sys.argv) != 2:
#     print("Usage: dataOperate(every particle in one file).py <file directory name>")

# print("Read the file directory from the command line.")
# dataDir = sys.argv[1]

# ===================
# ===== way two =====
# ===================
parser = argparse.ArgumentParser()

parser.add_argument('--dataDir', type=str, default='./', help='directory of the file which need to deal with')
parser.add_argument('--numOfPendingFiles', type=int, default=500, help='the numbers of pending files')
parser.add_argument('--numOfPendingParticle', type=int, default=5000, help='the numbers of pending particle')
# parser.add_argument('--numOfParticle', type=int, default=100, help='the numbeers of Particle want to deal with')
parser.add_argument('--saveDataDir', type=str, default='./dataset', help='directory of the file which need to save')

config = parser.parse_args()
print(config)

# Get all file 
#dataDir = "./snapshot1"
files = os.listdir(config.dataDir)
files.sort()

# find the group of the hdf5 file
def get_hdf5_group(hdf5File):
    group = []
    for key in hdf5File:
        group.append(key)
    return group

# print(f[get_hdf5_group("./Datasets/snapshot_000.hdf5")[0]])
# 提取相应粒子ID的粒子（算法时间复杂度O(n^2))
id_ = 1
for i in range(config.numOfPendingParticle):
    data = []
    for file in files:
        filename = os.path.splitext(file)
        if filename[1] == ".hdf5":
            path = os.path.join(config.dataDir + '/' + file)
            f = h5py.File(path)
            group = f[get_hdf5_group(f)[0]]
            #print(get_hdf5_group(path)[0])
            
            if int(group[i][0]) == id_:
                data.append(group[i])
                
    data = np.array(data)
    data = data.astype(np.float32)
    if id_ < 6000:
        dataset = h5py.File(config.saveDataDir + '/train/' + 'particleID' + str(id_) + '.hdf5', 'w')
        dataset.flush()

        dataset.create_dataset('dataset', data=data)
        dataset.close()
    
    if id_ >=6000 and id_ < 7000:
        dataset = h5py.File(config.saveDataDir + '/valid/' + 'particleID' + str(id_) + '.hdf5', 'w')
        dataset.flush()

        dataset.create_dataset('dataset', data=data)
        dataset.close()
    if id_ >= 7000:
        dataset = h5py.File(config.saveDataDir + '/test/' + 'particleID' + str(id_) + '.hdf5', 'w')
        dataset.flush()

        dataset.create_dataset('dataset', data=data)
        dataset.close()
        
    id_ += 1