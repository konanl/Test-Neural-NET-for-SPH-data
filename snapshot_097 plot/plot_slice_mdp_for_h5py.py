from matplotlib import *
from numpy import *
from matplotlib.pyplot import *
from sys import *
import os
import h5py

# Set a font
rcParams['font.family'] = 'Arial'                  # 字体属性
rcParams['font.size'] = 10.0

files = os.listdir('./')                          # 读取所有文件
files.sort()

for file in files:
    filename = os.path.splitext(file)             # filename-('snapshot_000', '.hdf5')
    if filename[1] == '.hdf5':
        rcParams['legend.handlelength'] = 0.5  # 图例之间的长度,使用字体大小的几分之几表示
        rcParams['legend.frameon'] = False  # 是否在图例外显示外框
        rcParams['legend.numpoints'] = 1  # the number of points in the legend line
        rcParams['legend.scatterpoints'] = 1  # 为散点图图例条目创建的标记点数

        # Adjust axes line width
        rcParams['axes.linewidth'] = 0.5  # 边的宽

        # Adjust ticks
        rcParams['xtick.major.size'] = 4  # 最大刻度大小
        rcParams['xtick.minor.size'] = 2  # 最小刻度大小
        rcParams['ytick.major.size'] = 4
        rcParams['ytick.minor.size'] = 2

        # Adjust Font Size
        rcParams['xtick.labelsize'] = 'x-small'  # 刻度标签字体大小
        rcParams['ytick.labelsize'] = 'x-small'
        rcParams['axes.labelsize'] = 'small'  # x轴和y轴的字体大小

        # Set Up Figure, Single Column MNRAS
        fig = gcf()
        ax = gca()
        fig, ax = subplots(1, 1)
        # ax.set_facecolor('k')                   # 设置背景颜色

        # Plot two plots in one (w,h)
        # fig.set_size_inches(8.27*0.5,8.27*(10./13.)*0.5)    # 设置图形的尺寸，单位为英寸。1英寸等于 2.54 cm
        # fig.set_size_inches(8.27 * 0.39, 8.27 * (10. / 13.) * 0.39)
        fig.set_size_inches(8.27 * 0.39, 8.27 * 0.39)

        """
        Read the file name from the command line.
        """

        dot_size = 1.5
        line_width_thin = 0.001
        line_width_thick = 0.1

        # num = round(int(filename[0][9:]) * 0.05, 2)  # 保留两位小数

        bin = h5py.File('./snapshot_097.hdf5')

        rho0 = 15
        rhoe = 15.20
        ids = bin["PartType0"]["ParticleIDs"][:]
        mass = bin["PartType0"]["Masses"][:] / 9.56072e+25
        pos = bin["PartType0"]["Coordinates"][:]
        pos_c = bin["PartType0"]["Coordinates"][:] / 6.37813e+08
        vel = bin["PartType0"]["Velocities"][:] / 10e+5
        rho = bin["PartType0"]["Density"][:] / 0.368477

        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        Ngas = len(mass)
        pgrp = np.zeros(Ngas)
        pgrp[:] = 1
        # 1 escape; 2 planet; 0 disk
        # pgrp[np.where(rho > rho0)] = 2.
        pgrp[np.where(rho > rhoe)] = 2.
        xx = pos_c[:, 0]
        yy = pos_c[:, 1]
        zz = pos_c[:, 2]
        vx = vel[:, 0]
        vy = vel[:, 1]
        vz = vel[:, 2]

        # subgrp_planet=pgrp[(pgrp>0.5)&(pgrp<1.5)]
        # Nplanet=
        # print(pgrp)
        mm = mass[np.where(pgrp == 2)]
        # print(mm, len(mm))
        Mp = np.sum(mm)
        # print(Mp)
        Mp0 = 0.

        while (np.fabs(Mp - Mp0) > 0.001):
            Mp0 = Mp
            rp = np.cbrt(3 * Mp / (4. * np.pi * rhoe))
            xx_p = xx[np.where(pgrp > 1)]
            yy_p = yy[np.where(pgrp > 1)]
            zz_p = zz[np.where(pgrp > 1)]

            x0 = np.sum(xx_p * mm) / Mp
            y0 = np.sum(yy_p * mm) / Mp
            z0 = np.sum(zz_p * mm) / Mp

            vx_p = vx[np.where(pgrp > 1)]
            vy_p = vy[np.where(pgrp > 1)]
            vz_p = vz[np.where(pgrp > 1)]

            vx0 = np.sum(vx_p * mm) / Mp
            vy0 = np.sum(vy_p * mm) / Mp
            vz0 = np.sum(vz_p * mm) / Mp

            xxr = xx - x0
            yyr = yy - y0
            zzr = zz - z0
            vxr = vx - vx0
            vyr = vy - vy0
            vzr = vz - vz0

            Radius = np.sqrt(xxr * xxr + yyr * yyr + zzr * zzr)

            E = 0.5 * (vxr * vxr + vyr * vyr + vzr * vzr) - Mp / Radius

            pgrp[:] = 1.

            pgrp[np.where(E < 0)] = 2.

            # only for particles with E<0

            majora = -Mp / (2.0 * E)
            j2 = (xxr * vyr - yyr * vxr) * (xxr * vyr - yyr * vxr) + (yyr * vzr - zzr * vyr) * (
                    yyr * vzr - zzr * vyr) + (xxr * vzr - zzr * vxr) * (xxr * vzr - zzr * vxr)
            ecent = np.sqrt(1. - j2 / (Mp * majora))
            # if E>0 we set artificially majora>0 to bypass majora*(1-ecent) argument
            # print(majora,)
            # print(ecent)
            pgrp[np.where((majora * (1 - ecent) > rp) & (E < 0) & (rho < 50.3))] = 0.

            mm = mass[np.where(pgrp > 1)]
            # print(mm)
            Mp = np.sum(mm)

            mdisk = mass[np.where(pgrp < 1)]
            # print(mdisk)
            Md = np.sum(mdisk)

            mesccpe = mass[np.where(pgrp == 1)]
            # print(mesccpe)
            Mesc = np.sum(mesccpe)

            # print(pgrp)

            # 分别取出不同趋势的粒子的id
            pos_mm = pos[np.where(pgrp > 1)]  # planet id
            # print(pos_mm)
            # print('-----')

            pos_disk = pos[np.where(pgrp < 1)]  # disk id
            # print(pos_disk)
            # print('-----')

            pos_mesccpe = pos[np.where(pgrp == 1)]  # escape id
            # print(pos_mesccpe)
            # print('-----')

            # 找到不同id对应的坐标
            x_slice_m = pos_mm[:, 0]
            y_slice_m = pos_mm[:, 1]
            x_slice_d = pos_disk[:, 0]
            y_slice_d = pos_disk[:, 1]
            x_slice_e = pos_mesccpe[:, 0]
            y_slice_e = pos_mesccpe[:, 1]

        # for i in :
        #     if pgrp[i] == 1:
        #         print(pgrp[i])
        # print(pgrp)
            print(len(x_slice_m), len(y_slice_m))
            print(len(x_slice_d), len(y_slice_d))
            print(len(x_slice_e), len(y_slice_e))


        s1 = scatter(x_slice_m, y_slice_m, s=dot_size, c='r', linewidth=line_width_thin)
        s2 = scatter(x_slice_d, y_slice_d, s=dot_size, c='b', linewidth=line_width_thin)
        s3 = scatter(x_slice_e, y_slice_e, s=dot_size, c='g', linewidth=line_width_thin)

        xlim((-4e9, 4e9))
        ylim((-4e9, 4e9))

        # ax.set_axisbelow(True)  # 网格显现在图形下方
        # grid()

        xlabel('X [cm]')
        ylabel('Y [cm]')

        # ax.text(-1.5e9, 1.5e9, str(num) + 'hr', fontsize='8')

        title("Disk Analysiser")
        legend(handles=[s1, s2, s3], labels=["Planet", "Disk", "Escape"], loc='best', fontsize='xx-small', frameon=True)

        savefig('snapshot_097.hdf5.slice.png', dpi=300, bbox_inches='tight')  # facecolor 设置背景颜色
        close()






