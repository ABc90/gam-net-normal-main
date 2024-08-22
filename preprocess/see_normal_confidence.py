import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d


root_dir = "/media/junz/volume1/dataset/normal-data/"
data_set = 'PCPNet' # ['PCPNet', 'SceneNN', 'FamousShape']
data_list = 'trainingset_whitenoise'  #['trainingset_whitenoise','testset_PCPNet','test_FamousShape','test_SceneNN']
is_vis = True
# pre_dir = '/home/junz/works/SHS-Net-main/log/231229_103322/results_PCPNet/ckpt_800/pred_normal' #231229_103322，240102_193803
confidence_dir = root_dir + data_set +'/'+ 'modify2/'

def normalize_point_cloud(point_cloud):
    # 中心化点云数据
    center = np.mean(point_cloud, axis=0)
    centered_points = point_cloud - center

    # 计算归一化常数（例如最大范围）
    max_range = np.max(np.linalg.norm(centered_points, axis=1))
    normalized_points = centered_points / max_range
    return normalized_points, max_range


with open(os.path.join(root_dir, data_set, 'list', data_list + '.txt')) as f:
    cur_sets = f.readlines()
    cur_sets = [x.strip() for x in cur_sets]
    cur_sets = list(filter(None, cur_sets))

for filename in cur_sets:
    file_pts = os.path.join(os.path.join(root_dir, data_set,filename + '.xyz'))
    file_normal_gt = os.path.join(os.path.join(root_dir, data_set,filename + '.normals'))

    file_confidence = os.path.join(os.path.join(confidence_dir,filename + '.mdensity'))
    file_ids = os.path.join(os.path.join(root_dir,data_set,filename + '.pidx'))
    pts = np.loadtxt(file_pts, dtype=np.float32)
    pts,_= normalize_point_cloud(pts)
    normal_cf= np.loadtxt(file_confidence, dtype=np.float32)
    normal_gt = np.loadtxt(file_normal_gt, dtype=np.float32)
    ids = np.loadtxt(file_ids,dtype=int)

    ppts = pts[normal_cf>=0.5]
    ppts2 = pts[normal_cf<0.5]

    if is_vis is True:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pp = np.repeat(normal_cf, 3).reshape(-1, 3)
        # pcd.normals = o3d.utility.Vector3dVector(pp*normals)
        colors = plt.get_cmap("hot")(normal_cf)
        # colors[normals_s==1,:3] = [1, 0, 0]
        # colors[normals_s==-1,:3] = [0, 1, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])  
     
        # o3d.visualization.draw_geometries(geometries, point_show_normal=True)
        # o3d.visualization.draw_geometries([pcd])

        pcd2 = o3d.geometry.PointCloud()   
        pcd2.points = o3d.utility.Vector3dVector(ppts+[3,0,0])

        pcd3 = o3d.geometry.PointCloud()   
        pcd3.points = o3d.utility.Vector3dVector(ppts2+[6,0,0])

        o3d.visualization.draw_geometries([pcd,pcd2,pcd3],point_show_normal=True)

