import h5py
import os


def findNeighborParticles(coordinate, smoothingLength, group, filename): 

    # radius = ((coordinate_x[0]-coordinate_y[0])**2 
    #           + (coordinate_x[1]-coordinate_y[1])**2 
    #           + (coordinate_x[2]-coordinate_y[2])**2)
    # neighbor_Particles = {}
    for i in range(0, coordinate.shape[0]):
        """Find the neighbor particles"""
        key = str(i)
        neighbor_Particles = []
        for j in range(i+1, coordinate.shape[0]):
            #
            radius = ((coordinate[i, 0]-coordinate[j, 0])**2 + 
                      (coordinate[i, 1]-coordinate[j, 1])**2 + 
                      (coordinate[i, 2]-coordinate[j, 2])**2)**0.5
            # print(radius)
            sl = smoothingLength[i, 0]
            # print(sl)     # sl = 24679906.0
            if radius <= sl:
                neighbor_Particles.append(group[j][:])
        
        # save
        dataset = h5py.File('./data/'+filename+'_'+key+'_particle.hdf5', 'w')
        dataset.flush()

        dataset.create_dataset(key, data=neighbor_Particles)
        dataset.close()  


if __name__ == "__main__":
    """test code"""
    file_dir = './'
    files = os.listdir(file_dir)
    for file in files:
        filename = os.path.splitext(file)
        if filename[1] == '.hdf5':
            path = os.path.join(file_dir + '/' + file)
            f = h5py.File(path)
            # group = f[get_hdf5_group(f)[0]]
            group = f['dataset']
            #print(group)
            coordinate = group[:, 6:9]
            smoothingLength = group[:, 5].reshape(-1, 1)
            # print(coordinate.shape, smoothingLength.shape)
            # neighbor_particles_list = findNeighborParticlces()
            neighbor_Particles = findNeighborParticles(coordinate, smoothingLength, group, filename[0])
            print(len(neighbor_Particles))