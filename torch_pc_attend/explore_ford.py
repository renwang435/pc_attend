"""
Topic : Visualize the UMichigan Ford Dataset
References
 - Umich Dataset : http://robots.engin.umich.edu/SoftwareData/Ford
 - Data Download : http://robots.engin.umich.edu/uploads/SoftwareData/Ford/dataset-1-subset.tgz
 - Blog          :
 - Notes
    - Tested on Python3.5
"""

## Import Libraries
import os
import h5py  # pip3 install h5py
import vispy  # pip3 install vispy
import vispy.scene
import numpy as np
import scipy.io as sio


## 1. Read a matlab file
def get_data(filename_mat, verbose=0):
    mat_contents = sio.loadmat(filename_mat)
    keys = mat_contents.keys()
    data = mat_contents['SCAN']
    a = data[0][0]

    if verbose:
        for each in a:
            print(each.shape)
    data = a[0].T
    return data


## 2. Read a directory containing matlab files and store as hfd5
def write_datum_h5(dirname_mat, filename_h5, file_count_max=100):
    datum = []
    file_count = 0
    if os.path.exists(dirname_mat):
        with h5py.File(filename_h5, 'w') as hf:
            for file_id, file in enumerate(sorted(os.listdir(dirname_mat))):
                file_tmp = os.path.join(dirname_mat, file)
                print(' - File : ', file_tmp)
                data = get_data(file_tmp)
                hf.create_dataset('data_%.4d' % (file_id), data=data)
                file_count += 1
                if file_count > file_count_max:
                    return 1
        return 1
    else:
        return 0


## 3. Read a h5d5 file
def read_datum_h5(filename_h5):
    datum = []
    with h5py.File(filename_h5, 'r') as hf:
        for node in hf:
            print(' ->', node)
            datum.append(np.array(hf.get(node)))
        return datum


## Plot U.Mich LiDAR data saved as .h5 file
def plot_data(datum, point_size=3):
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    view.bgcolor = '#ffffff'
    view.bgcolor = '#111111'
    view.camera = ['perspective', 'panzoom', 'fly', 'arcball', 'base', 'turntable', None][2]
    if 1:
        view.camera.fov = 60
        view.camera.scale_factor = 0.7
        view.camera.keymap['Right'] = (1, 5)
        view.camera.keymap['Left'] = (-1, 5)
        view.camera.keymap['Up'] = (1, 4)
        view.camera.keymap['Down'] = (-1, 4)

    axis = vispy.scene.visuals.XYZAxis(parent=view.scene)
    scatter = vispy.scene.visuals.Markers(parent=view.scene)
    scatter.set_data(datum[0], size=point_size)

    def update(data):
        scatter.set_data(data, size=point_size)

    @canvas.events.key_press.connect
    def keypress(e):
        print(' - Event : ', e._key.name)
        global t
        print('  - File Index : ', t)
        if e._key.name == '=':
            t = min(t + 1, len(datum) - 1)
            update(datum[t])

        if e._key.name == '-':
            t = max(t - 1, 0)
            update(datum[t])

    canvas.show()


if __name__ == '__main__':
    ## Define the directory containing the .mat files
    dirname_umich_scans = './ford_data/IJRR-Dataset-1-subset/SCANS'  # contains the .mat files
    filename_h5 = './ford_data/data_umich.h5';
    file_count_max = 1000

    ## First convert all .mat files into a .h5 file (data_write=1)
    data_write = 0
    if data_write:
        write_datum_h5(dirname_umich_scans, filename_h5, file_count_max)

    ## Then plot that 3D LiDAR data (data_write=0)
    ## You can use wasd and arrow keys to move around. (f,c)-> increase or decrease your z
    ## Note : Zooming does not currently work, so scrolling wont have an effect
    else:
        t = 0
        datum_3d = read_datum_h5(filename_h5)
        plot_data(datum_3d, point_size=2)
        vispy.app.run()