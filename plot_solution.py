import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use("Agg")
import numpy as np
import pynbody
from itertools import islice


def get_config():
    parser = argparse.ArgumentParser("The Config Of Figure Data", add_help=False)
    parser.add_argument('--times', type=list, default=['00500', '01000', '02000'], help='The Times')
    parser.add_argument('--fileDirectory', type=str, default='./data', help='File Directory')
    parser.add_argument('--fileKeyName', type=str, default='aspuaug_340k_100k_MANEOS.', help='The Key Name Of File')
    # parser.add_argument('--size', type=list, default=[2, 3], help='The Figure Size Of Sub-figure')
    parser.add_argument('--nrows', type=int, default=2)
    parser.add_argument('--ncols', type=int, default=3)
    parser.add_argument('--saveName', type=str, default='test', help='The Name Of Save Figure')

    return parser.parse_args()


def set_rcParams():
    """Set the style of rcParams."""
    params = {
        "font.family": "serif",
        "font.size": 10.0,
        "legend.handlelength": 0.5,
        "legend.frameon": False,
        "legend.numpoints": 1,
        "legend.scatterpoints": 1,

        "axes.linewidth": 0.1,
        "xtick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.major.size": 4,
        "ytick.minor.size": 2,

        "xtick.labelsize": 'x-small',
        "ytick.labelsize": 'x-small',
        "axes.labelsize": 'small',
    }
    matplotlib.rcParams.update(params)


def find_all_files(files_path, file_type):
    """遍历指定文件夹所有指定类型文件"""
    files = os.listdir(files_path)
    files_names = [] 
    for file in files:
        filename = os.path.splitext(file)
         # file_type: '.press',
        if filename[1] == file_type:
            files_names.append(str(file)) 

    return files_names


def object_files(files_names, key_word):
    """找出目标文件"""
    files = []  # 存储符合条件的目标文件路径名称
    for i in files_names:  # 遍历所有文件
        if key_word in i:  # key_word表示文件命名中存在的关键词 可根据关键词筛选文件
            files.append(i)  # 若文件名称中存在关键词 则将其保存在列表中

    return files


def load_dataset(filesname):
    dataset = []  # 保存读取的数据
    with open(filesname, 'r') as file_obj:  # 打开文件
        for line in islice(file_obj, 1, None):  # 跳过1行数据
            for i, data in enumerate(line.split()):
                if i < 2:
                    data_value = data.strip()  # 删除数字两边的空格
                    dataset.append(str(data_value))  # 将每一行数据保存

    dataset.pop(-1)  # 若最后一行数据不完全，则可删除

    dataset = np.array(dataset)  # 列表转换为数组

    dataset = dataset.astype(float)  # 将字符串转换为浮点数

    return dataset


"""
    The Scope:
        b05-v30-2.25MM-1.125MP-MANEOS 
        target (1,58801) (58802,208441) impactor (208442,237638)(237639,299870)

        b051-v30-2.25MM-1.125MP-MANEOS 
        target (1,58801) (58802,208441) impactor (208442,237638)(237639,299870)   

        b05-v30-2.25MM-0.91MP-MANEOS 
        target (1,58801) (58802,208441) impactor (208442,229862)(229863,285746)

        b048-v30-2.25MM-0.91MP-MANEOS 
        target (1,58801) (58802,208441) impactor (208442,229862)(229863,285746)

        aspuaug_340k_100k_MANEOS
        target (1,101977) (101978,338269) impactor (338270,367910)(367911,437054)    
"""


def plot_press(time, ax, fileDir='./data', fileKeyName="aspuaug_340k_100k_MANEOS."):
    """Plot pressure."""

    # fileDir = "./data/pressure"
    # fileDir_ = os.path.join(fileDir, 'press')
    filename = fileKeyName + time
    file = filename
    filename = os.path.splitext(filename)

    # Automatic read range.
    kayname = filename[0]
    particleScope = None
    if kayname == 'aspuaug_340k_100k_MANEOS':
        particleScope = (338270, 367910)
    elif kayname == 'b05-v30-2.25MM-1.125MP-MANEOS' or kayname == 'b051-v30-2.25MM-1.125IMP-MANEOS':
        particleScope = (208442, 237638)
    elif kayname == 'b05-v30-2.25MM-0.91MP-MANEOS' or kayname == 'b048-v30-2.25MM-0.91MP-MANEOS':
        particleScope = (208442, 229862)
    else:
        assert particleScope is not None, 'The particleScope is None'

    if filename[1][1:].isdigit():
        # Calculate time
        num = round(int(filename[1][1:]) * 0.053148, 2)
        file_path = os.path.join(fileDir, file)
        # file_path_ = os.path.join(fileDir_, file)

        bin = pynbody.tipsy.TipsySnap(file_path)
        pos = bin["pos"]
        # N = np.size(pos[:, 0])

        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        x_imp_core = []
        y_imp_core = []
        z_imp_core = []

        for i in range(particleScope[0], particleScope[1]):
            x_imp_core.append(x[i])
            y_imp_core.append(y[i])
            z_imp_core.append(z[i])

        x = x - np.mean(x_imp_core)
        y = y - np.mean(y_imp_core)
        z = z - np.mean(z_imp_core)

        Radius = np.sqrt(x ** 2.0 + y ** 2.0 + z ** 2.0)

        files_path = os.path.join(fileDir, "press")
        
        key_word = file
        files_names = find_all_files(files_path, file_type='.press')
        object_files_names = object_files(files_names, key_word)

        dataset = []
        for filename in object_files_names:
            fp = os.path.join(files_path, filename)
            data_one = load_dataset(fp)
            dataset.append(data_one)

        Press = dataset[0]

        Radius_imp_core = []
        Press_imp_core = []

        for i in range(particleScope[0], particleScope[1]):
            Radius_imp_core.append(Radius[i])
            Press_imp_core.append(Press[i])

        dot_size = 1.5
        line_width_thin = 0.001
        line_width_thick = 0.1

        ax.set_axisbelow(True)

        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        
        bwith = 0.5
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)

        # ax.grid()
        ax.scatter(Radius_imp_core, Press_imp_core, s=dot_size, linewidth=line_width_thin)

        # ax.xlabel("Radius(RE)")
        # ax.ylabel("Pressure(GPa)")

        # ax.title(str(num) + 'min', fontsize='8')
        # plt.savefig(file + '_press.png', dpi=300, bbox_inches='tight')


def plot_density(time, ax, fileDir='./data', fileKeyName="aspuaug_340k_100k_MANEOS."):
    """Plot density."""

    # ax.cla()

    # fileDir_ = os.path.join(fileDir, 'density')
    filename = fileKeyName + time
    file = filename
    filename = os.path.splitext(filename)

    # Automatic read range.
    kayname = filename[0]
    particleScope = None
    if kayname == 'aspuaug_340k_100k_MANEOS':
        particleScope = (338270, 367910)
    elif kayname == 'b05-v30-2.25MM-1.125MP-MANEOS' or kayname == 'b051-v30-2.25MM-1.125IMP-MANEOS':
        particleScope = (208442, 237638)
    elif kayname == 'b05-v30-2.25MM-0.91MP-MANEOS' or kayname == 'b048-v30-2.25MM-0.91MP-MANEOS':
        particleScope = (208442, 229862)
    else:
        assert particleScope is not None, 'The particleScope is None'

    if filename[1][1:].isdigit():
        num = round(int(filename[1][1:]) * 0.053148, 2)
        file_path = os.path.join(fileDir, file)

        bin = pynbody.tipsy.TipsySnap(file_path)
        pos = bin["pos"]

        # N = np.size(pos[:, 0])

        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        x_imp_core = []
        y_imp_core = []
        z_imp_core = []

        for i in range(particleScope[0], particleScope[1]):
            x_imp_core.append(x[i])
            y_imp_core.append(y[i])
            z_imp_core.append(z[i])

        Radius = np.sqrt(x ** 2.0 + y ** 2.0 + z ** 2.0)

        files_path = os.path.join(fileDir, "density")
        key_word = file
        
        files_names = find_all_files(files_path, file_type=".den")
        object_files_names = object_files(files_names, key_word)
        
        dataset = []
        for filename in object_files_names:
            fp = os.path.join(files_path, filename)
            data_one = load_dataset(fp)
            dataset.append(data_one)
            

        Density = dataset[0]

        Radius_imp_core = []
        Den_imp_core = []

        for i in range(particleScope[0], particleScope[1]):
            Radius_imp_core.append(Radius[i])
            Den_imp_core.append(Density[i])

        dot_size = 1.5
        line_width_thin = 0.001
        line_width_thick = 0.1

        ax.set_axisbelow(True)

        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        
        bwith = 0.5
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)

        # ax.grid()
        ax.scatter(Radius_imp_core, Den_imp_core, s=dot_size, linewidth=line_width_thin)

        # ax.xlabel("Radius(RE)")
        # ax.ylabel("Density($$g/cm^3$$)")

        # ax.title(str(num) + 'min', fontsize='8')
        # plt.savefig(file + '_den.png', dpi=300, bbox_inches='tight')


def plot_temp(time, ax, fileDir='./data', fileKeyName="aspuaug_340k_100k_MANEOS."):
    """Plot temp."""

    # ax.cla()

    # fileDir = os.path.join(fileDir, 'temp')
    filename = fileKeyName + time
    file = filename
    filename = os.path.splitext(filename)

    # Automatic read range.
    kayname = filename[0]
    particleScope = None
    if kayname == 'aspuaug_340k_100k_MANEOS':
        particleScope = (338270, 367910)
    elif kayname == 'b05-v30-2.25MM-1.125MP-MANEOS' or kayname == 'b051-v30-2.25MM-1.125IMP-MANEOS':
        particleScope = (208442, 237638)
    elif kayname == 'b05-v30-2.25MM-0.91MP-MANEOS' or kayname == 'b048-v30-2.25MM-0.91MP-MANEOS':
        particleScope = (208442, 229862)
    else:
        assert particleScope is not None, 'The particleScope is None'

    if filename[1][1:].isdigit():
        num = round(int(filename[1][1:]) * 0.053148, 2)
        file_path = os.path.join(fileDir, file)

        bin = pynbody.tipsy.TipsySnap(file_path)
        pos = bin["pos"]

        # N = np.size(pos[:, 0])

        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        x_imp_core = []
        y_imp_core = []
        z_imp_core = []

        for i in range(particleScope[0], particleScope[1]):
            x_imp_core.append(x[i])
            y_imp_core.append(y[i])
            z_imp_core.append(z[i])

        Radius = np.sqrt(x ** 2.0 + y ** 2.0 + z ** 2.0)

        files_path = os.path.join(fileDir, "temp")
        key_word = file
        files_names = find_all_files(files_path, file_type='.temp')
        object_files_names = object_files(files_names, key_word)

        dataset = []
        for filename in object_files_names:
            fp = os.path.join(files_path, filename)
            data_one = load_dataset(fp)
            dataset.append(data_one)

        Temp = dataset[0]

        Radius_imp_core = []
        Temp_imp_core = []

        for i in range(particleScope[0], particleScope[1]):
            Radius_imp_core.append(Radius[i])
            Temp_imp_core.append(Temp[i])

        dot_size = 1.5
        line_width_thin = 0.001
        line_width_thick = 0.1

        ax.set_axisbelow(True)

        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        
        bwith = 0.5
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)

        # ax.grid()
        ax.scatter(Radius_imp_core, Temp_imp_core, s=dot_size, linewidth=line_width_thin)

        # ax.xlabel("Radius(RE)")
        # ax.ylabel("Temp($$g/cm^3$$)")

        # ax.title(str(num) + 'min', fontsize='8')
        # plt.savefig(file + '_temp.png', dpi=300, bbox_inches='tight')        
 

def plot_process(time, ax, fileDir='./data', fileKeyName="aspuaug_340k_100k_MANEOS."):
    """Plot the color-bar figure according to the density."""

    # ax.cla()
    # ax.axis('off')

    # fileDir = './data'
    filename = fileKeyName + time
    file = filename
    filename = os.path.splitext(filename)

    if filename[1][1:].isdigit():
        num = round(int(filename[1][1:]) * 0.053148, 2)
        file_path = os.path.join(fileDir, file)

        bin = pynbody.tipsy.TipsySnap(file_path)
        pos = bin["pos"]
        vel = bin['vel']
        mass = bin['mass']
        den = bin['rho'] * 0.368477
        temp = bin['temp']
        # energy  = bin['energy']
        # press  = bin['press']
        # soundspeed  = bin['soundspeed']
        # mat  = bin['mat']

        N = np.size(pos[:, 0])

        x = pos[:, 0]
        y = pos[:, 1]
        # z = pos[:, 2]

        den_min = np.min(den)
        den_max = np.max(den)

        # Sort the particles according to density
        idx = den.argsort()

        ax.set_axisbelow(True)

        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        
        bwith = 0.5
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)

        # ax.grid()
        # plt.xlim((-3e0,3e0))
        # plt.ylim((-3e0,3e0))

        dot_size = 1.5
        line_width_thin = 0.001
        line_width_thick = 0.1

        # Make a scatter plot
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", "5%", pad="3%")
        im = ax.scatter(x[idx], y[idx], s=dot_size, c=den[idx], cmap=plt.cm.rainbow, linewidth=line_width_thin)
        # cb = plt.colorbar(im,
        #                   cax=cax
        #                   )
        # # set color_bar edges and its` color
        # cb.outline.set_linewidth(0.35)
        # # cb.outline.set_visible(True)
        # cb.outline.set_edgecolor('black')
        # cb.dividers.set_color('black')

        # add color_bar
        # cax = add_right_cax(ax, pad=0, width=0.02)
        # cbar = plt.colorbar(im, cax=cax)

        # plt.tight_layout()

        # ax.set_xlabel("x [R$_{\oplus}$]")
        # ax.set_ylabel("y [R$_{\oplus}$]")

        # ax.set_title("M-ANEOS")
        # plt.legend(loc='best')
        ax.text(-13, 13, str(num) + 'hr', fontsize='8')
        # ax.text(-0.9e9, 0.9e9, str(num) + 'hr', fontsize='8')
        # plt.savefig(file + '.png', dpi=300, bbox_inches='tight')
        return im


def plot_color_bar(ax, im):
    """Plot Right ColorBar"""
    ax.cla()
    ax.axis('off')
    plt.colorbar(im, location='left')


def plot_multi_figure(times, nrows=2, ncols=3, plot=None,
                      fileDir='./data', fileKeyName="aspuaug_340k_100k_MANEOS.", savename='test'):
    """Plot multi-figure."""
    # Set up the figure
    # figsize = (4*ncols, 4*nrows)
    # fig = plt.figure(figsize=figsize)
    fig = plt.figure()

    # Set fig size
    # fig.set_size_inches(8.27 * 0.5, 8.27 * (10./13.) * 0.5)
    # fig.set_size_inches(8.27 * 0.39 * nrows, 8.27 * (10./13.) * 0.39 * ncols)
    fig.set_size_inches(8.27 * 0.39 * ncols * 1.5, 8.27 * (10. / 13.) * 0.39 * nrows * 2)

    gs = matplotlib.gridspec.GridSpec(nrows, ncols)
    axes = [plt.subplot(gs[i_y, i_x]) for i_y in range(nrows) for i_x in range(ncols)]

    # Plot each snapshot

    for i_ax, ax in enumerate(axes):

        plt.sca(ax)
        ax.set_rasterization_zorder(1)

        # Plot
        plot_model = i_ax % ncols
        plot_index = int(i_ax / ncols)

        # for time in times:
        # ax.cla()

        if (plot_model + 1) % (ncols) == 0:
            print("Start plot color_bar...")
            print("Plot time-{}-color_bar...".format(times[plot_index]))
            im = plot[plot_model-ncols+1](times[plot_index], ax, fileDir, fileKeyName)
            plot[plot_model](ax, im)
        else:
            print("Start plot...")
            print("Plot time-{}...".format(times[plot_index]))
            plot[plot_model](times[plot_index], ax, fileDir, fileKeyName)

        # if (i_ax + 1) % nrows == 0:
        #     im = plot[0](times[plot_index], ax, fileDir, fileKeyName)
        #     plot_color_bar(ax, im)

        # Axes etc.
        ax.set_aspect('equal')
        # ax.set_facecolor('k')
        ax.grid()

        # ax.set_xlim(-13, 13)
        # ax.set_ylim(-13, 13)

         # if i_ax in [0, ncols]:
        if plot_model == 0:
            ax.set_ylabel(r"y Postion $(R_\oplus)$")
        else:
            ax.set_yticklabels([])
        if (nrows - 1) * (ncols-1) <= i_ax:
            ax.set_xlabel(r"x Postion $(R_\oplus)$")
        else:
            ax.set_xticklabels([])

        # Corner time labels
        # x = ax.get_xlim()[0] + 0.04 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        # y = ax.get_ylim()[0] + 0.89 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        # ax.text(x, y, "%.1f h" % (output_list[i_ax] / 60 ** 2), color='w')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    # Save
    save_name = savename + '.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # Set the global figure style / 设置全局风格
    set_rcParams()

    config = get_config()
    config.times = [
        "00200",
        "00300",
        "00400"
    ]

    print(config)

    # plot_multi_figure(times, ncols=3, nrows=2, plot=[plot_process, plot_press, plot_density], savename='test_2.3')
    plot_multi_figure(
        config.times,
        ncols=config.ncols + 1, nrows=config.nrows,
        plot=[plot_process, plot_density, plot_press, plot_temp, plot_color_bar],
        fileDir=config.fileDirectory,
        fileKeyName=config.fileKeyName,
        savename=config.saveName
    )

    # python
    # For Example:
    ###  python plot_solution.py
    ################  --fileDirectory './data'
    ################  --fileKeyName 'aspuaug_340k_100k_MANEOS.'
    ################  --nrows 2 --ncols 3
    ################  --saveName 'test'



