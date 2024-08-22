import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d


root_dir = "/media/junz/volume1/dataset/normal-data/"
data_set = 'PCPNet' # ['PCPNet', 'SceneNN', 'FamousShape']
data_list = 'testset_PCPNet'  #['trainingset_whitenoise','testset_PCPNet','test_FamousShape','test_SceneNN']
save_dir = 'category1_n5_special'
sigma_density = 0.1
sigma_bais = 0.05
category_num = 5 ## ? m*m
is_vis = True
is_save = False


def generate_colors(num_classes):
    # 生成一系列颜色（这里使用了一些预定义的颜色，你可以根据需要自定义颜色）
    colors = []
    for i in range(num_classes):
        color = np.random.rand(3,)  # 生成随机颜色
        colors.append(color)
    return colors


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


def polar_bin_generate_special(bin_num = 5,theta_range=(0-0.001, np.pi/2+0.001),phi_range=(-np.pi-0.001, np.pi+0.001)):

    theta_range =theta_range  # 极角范围
    phi_range = phi_range  # 方位角范围
    # 生成 m 个 bin
    bins = []
 
    phi_bins = {}
    mm = bin_num
    theta_bins = np.linspace(theta_range[0], theta_range[1], bin_num+1)  # 将极角范围等间距地分为 m 个区间
    phi_bins[0] = np.linspace(phi_range[0], phi_range[1], 1+1)  # 将方位角范围等间距地分为 m 个区间
    phi_bins[1] = np.linspace(phi_range[0], phi_range[1], 1*bin_num+1)  # 将方位角范围等间距地分为 m 个区间
    phi_bins[2] = np.linspace(phi_range[0], phi_range[1], 2*bin_num+1)  # 将方位角范围等间距地分为 m 个区间
    phi_bins[3] = np.linspace(phi_range[0], phi_range[1], 3*bin_num+1)  # 将方位角范围等间距地分为 m 个区间
    phi_bins[4] = np.linspace(phi_range[0], phi_range[1], 4*bin_num+1)  # 将方位角范围等间距地分为 m 个区间

    for i in range(mm):
        nn = len(phi_bins[i])-1
        for j in range(nn):
            # 计算每个 bin 的角度范围
            theta_start = theta_bins[i]
            theta_end = theta_bins[i+1]
            phi_start = phi_bins[i][j] 
            phi_end = phi_bins[i][j+1]
        
            bins.append({
                'theta_range': (theta_start, theta_end),
                'phi_range': (phi_start, phi_end)
            })

    return bins


def polar_bin_generate_old(bin_num = 10,theta_range=(0, np.pi),phi_range=(0, 2 * np.pi)): 
    m = bin_num  # bin 数量
    theta_range =theta_range  # 极角范围
    phi_range = phi_range  # 方位角范围
    # 生成 m 个 bin
    bins = []
    theta_bins = np.linspace(theta_range[0], theta_range[1], m+1)  # 将极角范围等间距地分为 m 个区间
    phi_bins = np.linspace(phi_range[0], phi_range[1], m+1)  # 将方位角范围等间距地分为 m 个区间
    for i in range(m):
        for j in range(m):
            # 计算每个 bin 的角度范围
            theta_start = theta_bins[i]
            theta_end = theta_bins[i+1]
            phi_start = phi_bins[j] 
            phi_end = phi_bins[j+1]
        
            bins.append({
                'theta_range': (theta_start, theta_end),
                'phi_range': (phi_start, phi_end)
            })

    return bins


def polar_bin_generate(bin_num = 10,theta_range=(0, np.pi),phi_range=(0, 2 * np.pi)):
    xx = theta_range[1]-theta_range[0]
    yy = phi_range[1] - phi_range[0]
    if xx>yy:
        ll = np.ceil(xx/yy)
        ll = ll.astype(int)
        mm = bin_num*ll  # bin 数量
        nn = bin_num
    else:
        ll = np.ceil(yy/xx)
        ll = ll.astype(int)
        mm = bin_num  # bin 数量
        nn = bin_num*ll
    
    theta_range =theta_range  # 极角范围
    phi_range = phi_range  # 方位角范围
    # 生成 m 个 bin
    bins = []
    if xx>yy:
        theta_bins = np.linspace(theta_range[0], theta_range[1], mm+1)  # 将极角范围等间距地分为 m 个区间
        phi_bins = np.linspace(phi_range[0], phi_range[1], nn+1)  # 将方位角范围等间距地分为 m 个区间
    else:
        theta_bins = np.linspace(theta_range[0], theta_range[1], mm+1)  # 将极角范围等间距地分为 m 个区间
        phi_bins = np.linspace(phi_range[0], phi_range[1], nn+1)  # 将方位角范围等间距地分为 m 个区间
    for i in range(mm):
        for j in range(nn):
            # 计算每个 bin 的角度范围
            theta_start = theta_bins[i]
            theta_end = theta_bins[i+1]
            phi_start = phi_bins[j] 
            phi_end = phi_bins[j+1]
        
            bins.append({
                'theta_range': (theta_start, theta_end),
                'phi_range': (phi_start, phi_end)
            })

    return bins


def cartesian_to_spherical(vectors):
    r_values, theta_values, phi_values = [], [], []
    
    for vector in vectors:
        x, y, z = vector
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        
        r_values.append(r)
        theta_values.append(theta)
        phi_values.append(phi)
    # 将转换后的列表转换为NumPy数组
    r_array = np.array(r_values)
    theta_array = np.array(theta_values)
    phi_array = np.array(phi_values)
    
    return r_array, theta_array, phi_array


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
category_shape = []
category_shape_modify = []

# 生成m个bin # 暂时: bin的构成并不均匀 phi_range 每个bin过大,theta_range有点小
bins = polar_bin_generate_special(bin_num = category_num,theta_range=(0-0.001, np.pi/2+0.001),phi_range=(-np.pi-0.001, np.pi+0.001))
# 输出每个 bin 的角度范围
for i, bin_info in enumerate(bins):
    print(f"Bin {i + 1}:")
    print(f"Theta range: {bin_info['theta_range']}")
    print(f"Phi range: {bin_info['phi_range']}")
    print()


with open(os.path.join(root_dir, data_set, 'list', data_list + '.txt')) as f:
    cur_sets = f.readlines()
    cur_sets = [x.strip() for x in cur_sets]
    cur_sets = list(filter(None, cur_sets))

for filename in cur_sets[1:20]:
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


    #######==================================================================================
    # 创建两个 KDTree
    kdtree_1 = KDTree(pts1)
    kdtree_2 = KDTree(pts2)

    # 对于 point_cloud_1 中的每个点，找到 point_cloud_2 中的最近邻点
    nearest_neighbors_indices = kdtree_2.query(pts1)[1]
    
    near_point_2_pts2 = pts2[nearest_neighbors_indices]
    print("diff p:" +str((pts1-near_point_2_pts2).max()))

    bias_2_pts2 =  np.exp(-np.linalg.norm(pts2[nearest_neighbors_indices]-pts1, axis=1)/(s1*sigma_bais))
    near_normal_2_normals2 = normals2[nearest_neighbors_indices]
    print("diff n:" +str((normals1-near_normal_2_normals2).max()))

    print("max: " +str(bias_2_pts2.max()))
    print("min: " +str(bias_2_pts2.min()))
    print("mean: " +str(bias_2_pts2.mean()))
    print("mid: "+ str(np.median(bias_2_pts2)))
    percentage_less_than_05 = np.sum(bias_2_pts2 < 0.5) / len(bias_2_pts2) * 100
    print("<0.5 % :" + str(percentage_less_than_05)+'%') 

    #######==================================================================================
    # normal to category 
    normals1_norm = normals1 / np.linalg.norm(normals1,axis=1,keepdims=True)
    near_normal_2_normals2_norm = near_normal_2_normals2/np.linalg.norm(near_normal_2_normals2,axis=1,keepdims=True) 
    # 假设您要投影到半球上半部分，即z轴非负方向的“bin”内
    normals1_norm[normals1_norm[:,2]<0,:]*= -1
    near_normal_2_normals2_norm[near_normal_2_normals2_norm[:,2]<0,:]*= -1

    r_values1, theta_values1, phi_values1 = cartesian_to_spherical(normals1_norm)
    r_values2, theta_values2, phi_values2 = cartesian_to_spherical(near_normal_2_normals2_norm)
    # print("r array:", r_values1)
    # print("theta array:", theta_values1)
    # print("phi array:", phi_values1)
    total_n = len(bins)
    C1 = (total_n+1)*np.ones(len(r_values1))
    C2 = (total_n+1)*np.ones(len(r_values2))

    # 判断每个极坐标在哪个 bin 内
    for i in range(len(r_values1)):
        for j, bin_info in enumerate(bins):
            theta_range = bin_info['theta_range']
            phi_range = bin_info['phi_range']
            if (theta_range[0] <= theta_values1[i] < theta_range[1]) and (phi_range[0] <= phi_values1[i] < phi_range[1]):
                # print(f"极坐标 {i + 1} 在 bin {j + 1} 内")
                C1[i]=j
                continue
    aaaa1 = (C1==101)
    
    for i in range(len(r_values2)):
        for j, bin_info in enumerate(bins):
            theta_range = bin_info['theta_range']
            phi_range = bin_info['phi_range']
            if (theta_range[0] <= theta_values2[i] < theta_range[1]) and (phi_range[0] <= phi_values2[i] < phi_range[1]):
                # print(f"极坐标 {i + 1} 在 bin {j + 1} 内")
                C2[i]=j
                continue
    aaaa2 = (C2==101)
    
    if aaaa1.sum()>0 or aaaa2.sum()>0:
        print("warning: " +filename+ " out of range ！！" )
    else:
        print("OK: " +filename+ " category is done ！！" )

    #######==================================================================================
    # surface denstiy :
    density_pts = evaluate_point_to_surface_self(pts1_norm,sigma_density)

    # save to lists
    bais_min_t  = bias_2_pts2.min()
    if bais_min_t < min_bias:
        min_bias = bais_min_t

    #######==================================================================================
    o_filename.append(filename)
    near_filename.append(filename2)
    o_pts.append(pts1)
    near_pts.append(near_point_2_pts2)
    o_normals.append(normals1)
    near_normals.append(near_normal_2_normals2)
    confidence.append(bias_2_pts2)
    noise_density.append(density_pts[:,0])
    category_shape.append(C1)
    category_shape_modify.append(C2)

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
        category_shape_path = os.path.join(output_folder, filename + '.category')
        category_shape_mofidy_path = os.path.join(output_folder, filename + '.mcategory')

        np.savetxt(save_normal_path, near_normals[i])
        np.savetxt(save_confidence_path,confidence[i])
        np.savetxt(save_density_path,noise_density[i])
        np.savetxt(category_shape_path,category_shape[i])
        np.savetxt(category_shape_mofidy_path,category_shape_modify[i])

        print(filename + " is saved! ")
    
    # 创建 Open3D 点云对象
    if is_vis is True:
        # # 绘制直方图
        # plt.hist(category_shape[i], bins=total_n, alpha=0.7, color='blue', edgecolor='black')
        # # 设置图表标题和标签
        # plt.title('Histogram of o normal distribution: '+ filename)
        # plt.xlabel('bins_category')
        # plt.ylabel('Frequency')
        # # 显示网格线
        # plt.grid(True)
        # # 显示直方图
        # plt.show()

        # plt.hist(category_shape_modify[i], bins=total_n, alpha=0.7, color='blue', edgecolor='black')
        # # 设置图表标题和标签
        # plt.title('Histogram of modify normal distribution: '+ filename)
        # plt.xlabel('bins_category')
        # plt.ylabel('Frequency')
        # # 显示网格线
        # plt.grid(True)
        # # 显示直方图
        # plt.show()
        colors_t = generate_colors(total_n)
        colors_t =  np.array(colors_t)

        pts1, _ = normalize_point_cloud(o_pts[i])
        near_normal_2_normals2 = near_normals[i]
        normals1 = o_normals[i]
        bais = confidence[i]
        density = noise_density[i]
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
        # colors3 = plt.get_cmap("tab20")(category_shape[i]/(total_n))
        colors3 = colors_t[category_shape[i].astype(int)]
        pcd3.colors = o3d.utility.Vector3dVector(colors3[:,:3])    


        pcd4= o3d.geometry.PointCloud()
        pcd4.points = o3d.utility.Vector3dVector(pts1+[9,0,0])
        # colors4 = plt.get_cmap("tab20")(category_shape_modify[i]/(total_n))
        colors4 = colors_t[category_shape_modify[i].astype(int)]
        pcd4.colors = o3d.utility.Vector3dVector(colors4[:,:3])    

        # 将两个点云对象放入一个列表中
        # geometries = [pcd1, pcd2, pcd3, pcd4]
        geometries = [pcd3,pcd4]


        # 显示两个点云
        o3d.visualization.draw_geometries(geometries, point_show_normal=True)