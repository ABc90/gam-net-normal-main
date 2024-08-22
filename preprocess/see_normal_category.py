import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

def generate_colors(num_classes):
    # 生成一系列颜色（这里使用了一些预定义的颜色，你可以根据需要自定义颜色）
    colors = []
    for i in range(num_classes):
        color = np.random.rand(3,)  # 生成随机颜色
        colors.append(color)
    return colors


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


def polar_bin_generate_special(bin_num = 5, theta_range=(0-0.001, np.pi/2+0.001),phi_range=(-np.pi-0.001, np.pi+0.001)):
    
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


def normalize_point_cloud(point_cloud):
    # 中心化点云数据
    center = np.mean(point_cloud, axis=0)
    centered_points = point_cloud - center

    # 计算归一化常数（例如最大范围）
    max_range = np.max(np.linalg.norm(centered_points, axis=1))
    normalized_points = centered_points / max_range
    return normalized_points, max_range


def categorical_cross_entropy(predictions, labels):
    epsilon = 1e-15  # 添加一个极小值以防止对数运算中出现的无穷大值
    predictions = np.clip(predictions, epsilon, 1 - epsilon)  # 将预测值限制在一个极小值和1之间，避免对数运算中出现无穷大值
    loss = -np.sum(labels * np.log(predictions)) / len(predictions)
    return loss


# 计算分类精度
def accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)  # 统计预测正确的样本数
    total = len(y_true)  # 总样本数
    accuracy = correct / total  # 计算分类精度
    return accuracy


root_dir = "/media/junz/volume1/dataset/normal-data/"
data_set = 'PCPNet' # ['PCPNet', 'SceneNN', 'FamousShape']
data_list = 'testset_med_noise'  #['trainingset_whitenoise','testset_PCPNet','test_FamousShape','test_SceneNN']
is_vis = True
pre_dir = '/home/junz/works/SHS-Net-main/log/001/results_PCPNet/ckpt_800/pred_normal' #231229_103322，240102_193803
category_num = 5  ## ? m*m
other_dir = 'category1_n5_special'

# 生成m个bin
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

A = []
for filename in cur_sets:
    file_pts = os.path.join(os.path.join(root_dir, data_set,filename + '.xyz'))
    file_normal_gt = os.path.join(os.path.join(root_dir, data_set,filename + '.normals'))
    file_category_gt = os.path.join(os.path.join(root_dir, data_set,other_dir,filename + '.category'))


    file_normal = os.path.join(os.path.join(pre_dir,filename + '.normals'))
    file_normal_s = os.path.join(os.path.join(pre_dir,filename + '.normals_s'))
    file_ids = os.path.join(os.path.join(root_dir,data_set,filename + '.pidx'))
    pts = np.loadtxt(file_pts, dtype=np.float32)
    normals= np.loadtxt(file_normal, dtype=np.float32)
    normal_gt = np.loadtxt(file_normal_gt, dtype=np.float32)
    normals_s = np.loadtxt(file_normal_s, dtype=np.float32)
    ids = np.loadtxt(file_ids,dtype=int)
    labels = np.loadtxt(file_category_gt,dtype=int)
    normal_gt_sparse = normal_gt[ids,:]
    labels_sparse = labels[ids]


    # normal to category 
    normal_norm = normals / np.linalg.norm(normals,axis=1,keepdims=True)
    normal_gt_sparse_norm = normal_gt_sparse / np.linalg.norm(normal_gt_sparse,axis=1,keepdims=True)
    # 假设您要投影到半球上半部分，即z轴非负方向的“bin”内
    normal_norm[normal_norm[:,2]<0,:]*= -1
    normal_gt_sparse_norm[normal_gt_sparse_norm[:,2]<0,:]*= -1

    # normal to 
    r_values1, theta_values1, phi_values1 = cartesian_to_spherical(normal_norm)
    r_values2, theta_values2, phi_values2 = cartesian_to_spherical(normal_gt_sparse_norm)

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
    aaaa1 = (C1==(total_n+1))

    for i in range(len(r_values2)):
        for j, bin_info in enumerate(bins):
            theta_range = bin_info['theta_range']
            phi_range = bin_info['phi_range']
            if (theta_range[0] <= theta_values2[i] < theta_range[1]) and (phi_range[0] <= phi_values2[i] < phi_range[1]):
                # print(f"极坐标 {i + 1} 在 bin {j + 1} 内")
                C2[i]=j
                continue
    aaaa1 = (C2==(total_n+1))

    # normals_s[normals_s==-1]=0
    num_classes = total_n
    onehot_encoded1 = np.eye(total_n)[C1.astype(int)]
    onehot_encoded2 = np.eye(total_n)[C2.astype(int)]
    
    # 调用函数计算分类精度
    accuracy = accuracy_score(C2.astype(int), C1.astype(int))
    print(filename+" 的分类精度: ", accuracy)
    A.append(accuracy)

    if is_vis is True:
        colors_t = generate_colors(total_n)
        colors_t =  np.array(colors_t)
        pts_n, s1 = normalize_point_cloud(pts)

        pcd4= o3d.geometry.PointCloud()
        pcd4.points = o3d.utility.Vector3dVector(pts_n[ids,:])
        # colors4 = plt.get_cmap("tab20")(C1/(total_n))
        colors4 = colors_t[C1.astype(int)]
        pcd4.colors = o3d.utility.Vector3dVector(colors4[:,:3])    

        pcd5= o3d.geometry.PointCloud()
        pcd5.points = o3d.utility.Vector3dVector(pts_n[ids,:]+[2,0,0])
        # colors5= plt.get_cmap("tab20")(C2/(total_n))
        colors5 = colors_t[C2.astype(int)]
        pcd5.colors = o3d.utility.Vector3dVector(colors5[:,:3])    


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_n[ids,:])
        pp = np.repeat(normals_s, 3).reshape(-1, 3)
        pcd.normals = o3d.utility.Vector3dVector(pp*normals)
        colors = plt.get_cmap("hot")(normals_s)
        colors[normals_s==1,:3] = [1, 0, 0]
        colors[normals_s==-1,:3] = [0, 1, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])  

        geometries = [pcd4,pcd5]
        o3d.visualization.draw_geometries(geometries)
        # o3d.visualization.draw_geometries([pcd],point_show_normal=True)
        # o3d.visualization.draw_geometries([pcd])


print("mean accuracy:" + str(np.mean(A)))