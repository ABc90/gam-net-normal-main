import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d


root_dir = "/media/junz/volume1/dataset/normal-data/"
data_set = 'PCPNet' # ['PCPNet', 'SceneNN', 'FamousShape']
data_list = 'testset_no_noise'  #['trainingset_whitenoise','testset_PCPNet','test_FamousShape','test_SceneNN']
is_vis = True
pre_dir = '/home/junz/works/SHS-Net-main/log/231229_103322/results_PCPNet/ckpt_800/pred_normal' #231229_103322，240102_193803

with open(os.path.join(root_dir, data_set, 'list', data_list + '.txt')) as f:
    cur_sets = f.readlines()
    cur_sets = [x.strip() for x in cur_sets]
    cur_sets = list(filter(None, cur_sets))

for filename in cur_sets:
    file_pts = os.path.join(os.path.join(root_dir, data_set,filename + '.xyz'))
    file_normal_gt = os.path.join(os.path.join(root_dir, data_set,filename + '.normals'))

    file_normal = os.path.join(os.path.join(pre_dir,filename + '.normals'))
    file_normal_s = os.path.join(os.path.join(pre_dir,filename + '.normals_s'))
    file_ids = os.path.join(os.path.join(root_dir,data_set,filename + '.pidx'))
    pts = np.loadtxt(file_pts, dtype=np.float32)
    normals= np.loadtxt(file_normal, dtype=np.float32)
    normal_gt = np.loadtxt(file_normal_gt, dtype=np.float32)
    normals_s = np.loadtxt(file_normal_s, dtype=np.float32)
    ids = np.loadtxt(file_ids,dtype=int)
    # normals_s[normals_s==-1]=0

    if is_vis is True:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[ids,:])
        pp = np.repeat(normals_s, 3).reshape(-1, 3)
        pcd.normals = o3d.utility.Vector3dVector(pp*normals)
        colors = plt.get_cmap("hot")(normals_s)
        colors[normals_s==1,:3] = [1, 0, 0]
        colors[normals_s==-1,:3] = [0, 1, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])  
     
        # o3d.visualization.draw_geometries(geometries, point_show_normal=True)
        o3d.visualization.draw_geometries([pcd],point_show_normal=True)
        # o3d.visualization.draw_geometries([pcd])


