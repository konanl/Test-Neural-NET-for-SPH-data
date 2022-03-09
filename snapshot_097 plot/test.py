import h5py
import os

s = h5py.File('./snapshot_097.hdf5')

p = s['PartType0/pgrp']

print(p[0])
planet = []
disk = []
escape = []
for i in range(len(p)):
    if p[i] == 2:
        planet.append(p[i])
    if p[i] == 1:
        disk.append(p[i])
    if p[i] == 0:
        escape.append(p[i])

print(len(planet), len(disk), len(escape))
