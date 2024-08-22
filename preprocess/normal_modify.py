import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d


root_dir = "/media/junz/volume1/dataset/normal-data/"
data_set = 'PCPNet' # ['PCPNet', 'SceneNN', 'FamousShape']
data_list = 'trainingset_whitenoise'  #['trainingset_whitenoise','testset_PCPNet','test_FamousShape','testset_SceneNN']
save_dir = 'modify_0.05'
sigma_density = 0.1
sigma_bais = 0.05 #0.05(o),0.1,0.02,0.01
is_vis = False
is_save = True


def remove_after_word(file_name, word):
    index = file_name.find(word)
    if index != -1:
        cleaned_file_name = file_name[:index-1]
        return cleaned_file_name
    else:
        return file_name


def normalize_point_cloud(point_cloud):
    # 中心化点云数据
    center = np.mean(point_cloud, axis=0)
    centered_points = point_cloud - center

    # 计算归一化常数（例如最大范围）
    max_range = np.max(np.linalg.norm(centered_points, axis=1))
    normalized_points = centered_points / max_range
    return normalized_points, max_range


def evaluate_point_to_surface_self(point_cloud,sigma):
    kdtree = KDTree(point_cloud, 10)
    lengths = np.zeros((point_cloud.shape[0],1))
    _ , pos_local_id= kdtree.query(point_cloud, k=100+1,p=1)
    for i, neighbors in enumerate(pos_local_id):
        k_nearest_neighbors = neighbors[1:] 
        offsets = np.mean(point_cloud[k_nearest_neighbors], axis=0)-point_cloud[neighbors[0]]
        lengths[i] = np.linalg.norm(offsets)
    lengths = np.exp(-lengths/sigma)
    lengths = (lengths-lengths.min())/(lengths.max()-lengths.min())
    return lengths


def l2_norm(v):
    norm_v = np.sqrt(np.sum(np.square(v), axis=1))
    return norm_v


min_bias = 1
cur_sets = []
o_filename = []
near_filename = []
o_pts = []
near_pts = []
o_normals = []
near_normals = []
confidence = []
noise_density = []
normal_confidence = []

with open(os.path.join(root_dir, data_set, 'list', data_list + '.txt')) as f:
    cur_sets = f.readlines()
    cur_sets = [x.strip() for x in cur_sets]
    cur_sets = list(filter(None, cur_sets))

for filename in cur_sets:
    file_pts = os.path.join(os.path.join(root_dir, data_set,filename + '.xyz'))
    file_normal = os.path.join(os.path.join(root_dir, data_set,filename + '.normals'))
    
    #  可能仅仅可用于pcpnet的数据集形式， 删除特定词 "noise,ddist" 之后的部分
    cleaned_name = remove_after_word(filename, "noise")
    cleaned_name = remove_after_word(cleaned_name, "ddist")
    cleaned_name = remove_after_word(cleaned_name, "density")

    filename2 = cleaned_name
    print(filename)
    print(filename2)

    file_pts2 = os.path.join(os.path.join(root_dir, data_set,filename2 + '.xyz'))
    file_normal2 = os.path.join(os.path.join(root_dir, data_set,filename2 + '.normals'))
    pts1 = np.loadtxt(file_pts, dtype=np.float32)
    pts1_norm, s1 = normalize_point_cloud(pts1)
    pts2 = np.loadtxt(file_pts2, dtype=np.float32)
    pts2_norm, s2  = normalize_point_cloud(pts2)
    normals1 = np.loadtxt(file_normal, dtype=np.float32)
    normals2 = np.loadtxt(file_normal2, dtype=np.float32)
    print("scale of two shape: s1:"+str(s1)+',s2: '+str(s2))

    
    # ==========计算点云的其他属性==========:
    #1. 创建点云对象
    # pcl = o3d.geometry.PointCloud()
    # pcl.points = o3d.utility.Vector3dVector(pts1[::100])

    #2. 计算点云的凸包
    # hull, _ = pcl.compute_convex_hull()
    # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    # hull_ls.paint_uniform_color((1, 0, 0))    
    # if is_vis is True:
    #     o3d.visualization.draw_geometries([pcl, hull_ls])  

    #3. 估计点云的骨架
    # downsampled = point_cloud.voxel_down_sample(voxel_size=0.05)  # 调整体素的大小来降采样点云
    # downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # line_set = o3d.geometry.LineSet.create_from_point_cloud_wlop(downsampled, alpha=0.003)
    # line_set.paint_uniform_color([0.1, 0.1, 0.7])
    # if is_vis is True:
    #     o3d.visualization.draw_geometries([downsampled, line_set])


    # 创建两个 KDTree
    kdtree_1 = KDTree(pts1)
    kdtree_2 = KDTree(pts2)

    # 对于 point_cloud_1 中的每个点，找到 point_cloud_2 中的最近邻点
    nearest_neighbors_indices = kdtree_2.query(pts1)[1]
    
    near_point_2_pts2 = pts2[nearest_neighbors_indices]
    near_normal_2_normals2 = normals2[nearest_neighbors_indices]
    print("diff p:" +str((pts1-near_point_2_pts2).max()))

    bias_2_pts2 =  np.exp(-np.linalg.norm(pts2[nearest_neighbors_indices]-pts1, axis=1)/(s1*sigma_bais))
    

    normals1_l = l2_norm(normals1)
    normals1_norm = np.divide(normals1, np.tile(np.expand_dims(normals1_l, axis=1), [1, 3]))
    near_normal_2_normals2_l = l2_norm(near_normal_2_normals2)
    near_normal_2_normals2_norm = np.divide(near_normal_2_normals2, np.tile(np.expand_dims(near_normal_2_normals2_l, axis=1), [1, 3]))
    nn = np.sum(np.multiply(normals1_norm, near_normal_2_normals2_norm), axis=1)
    nn[nn > 1] = 1
    nn[nn < -1] = -1
    ang = np.rad2deg(np.arccos(np.abs(nn)))/90
    bia_2_normal2 = np.exp(-ang/(sigma_bais))



    # bia_2_normal2 = np.exp(-np.linalg.norm(normals1-near_normal_2_normals2, axis=1)/(s1*sigma_bais))
    # near_normal_2_normals2 = normals2[nearest_neighbors_indices]
    print("diff n:" +str((normals1-near_normal_2_normals2).max()))

    print("max: " +str(bias_2_pts2.max()))
    print("min: " +str(bias_2_pts2.min()))
    print("mean: " +str(bias_2_pts2.mean()))
    print("mid: "+ str(np.median(bias_2_pts2)))
    percentage_less_than_05 = np.sum(bias_2_pts2 < 0.5) / len(bias_2_pts2) * 100
    percentage_less_than_05_2 = np.sum(bia_2_normal2 < 0.5) / len(bia_2_normal2) * 100

    print("surface <0.5 % :" + str(percentage_less_than_05)+'%') 
    print("normal <0.5 % :" + str(percentage_less_than_05_2)+'%') 





    # surface denstiy :
    density_pts = evaluate_point_to_surface_self(pts1_norm,sigma_density)

    # save to lists
    bais_min_t  = bias_2_pts2.min()
    if bais_min_t < min_bias:
        min_bias = bais_min_t

    o_filename.append(filename)
    near_filename.append(filename2)
    o_pts.append(pts1)
    near_pts.append(near_point_2_pts2)
    o_normals.append(normals1)
    near_normals.append(near_normal_2_normals2)
    confidence.append(bias_2_pts2)
    noise_density.append(density_pts[:,0])
    normal_confidence.append(bia_2_normal2)

    # print
    print("point_cloud_1 中每个点对于 point_cloud_2 的最近邻点索引：")
    print(nearest_neighbors_indices)    
    print('\n')
    print("每个点与无噪声点的距离：")
    print(bias_2_pts2)
    print('\n')
    print("当前信赖值的scale")
    print(min_bias)

# 测试可视化
if is_save is True:
    output_folder = os.path.join(os.path.join(root_dir, data_set,save_dir))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

for i in range(len(o_filename)):
    
    if is_save is True:
        filename = o_filename[i]
        save_normal_path = os.path.join(output_folder, filename + '.mnormals')
        save_confidence_path =  os.path.join(output_folder, filename + '.mconf')
        save_density_path =  os.path.join(output_folder, filename + '.mdensity')
        save_normal_confidence_path = os.path.join(output_folder, filename + '.normalconf')

        np.savetxt(save_normal_path, near_normals[i])
        np.savetxt(save_confidence_path,confidence[i])
        np.savetxt(save_density_path,noise_density[i])
        np.savetxt(save_normal_confidence_path,normal_confidence[i])

        print(filename + " is saved! ")
    
    # 创建 Open3D 点云对象
    if is_vis is True:
        pts1, _ = normalize_point_cloud(o_pts[i])
        near_normal_2_normals2 = near_normals[i]
        normals1 = o_normals[i]
        bais = confidence[i]
        density = noise_density[i]
        normal_bais = normal_confidence[i]

        pcd1= o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts1)
        pcd1.normals = o3d.utility.Vector3dVector(near_normal_2_normals2)
        colors1 = plt.get_cmap("hot")(bais)
        pcd1.colors = o3d.utility.Vector3dVector(colors1[:,:3])
        # o3d.visualization.draw_geometries([pcd],point_show_normal=True)
        
        pcd2= o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts1+[3,0,0])
        pcd2.normals = o3d.utility.Vector3dVector(normals1)
        colors2 = plt.get_cmap("hot")(density)
        pcd2.colors = o3d.utility.Vector3dVector(colors2[:,:3])        


        pcd3= o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(pts1+[6,0,0])
        pcd3.normals = o3d.utility.Vector3dVector(near_normal_2_normals2)
        colors3 = plt.get_cmap("hot")(normal_bais)
        pcd3.colors = o3d.utility.Vector3dVector(colors3[:,:3])       

        # 将两个点云对象放入一个列表中
        geometries = [pcd1, pcd2, pcd3]

        # 显示两个点云
        # o3d.visualization.draw_geometries(geometries, point_show_normal=True)
        o3d.visualization.draw_geometries(geometries)