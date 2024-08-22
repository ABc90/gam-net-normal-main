import os
import sys
import numpy as np
from tqdm.auto import tqdm
import scipy.spatial as spatial
import torch
from torch.utils.data import Dataset

def cos_angle(v1, v2):
    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)

def load_data(filedir, filename, dtype=np.float32, wo=False):
    filepath = os.path.join(filedir, 'npy', filename + '.npy')
    os.makedirs(os.path.join(filedir, 'npy'), exist_ok=True)
    if os.path.exists(filepath):
        if wo:
            return True
        data = np.load(filepath)
    else:
        data = np.loadtxt(os.path.join(filedir, filename), dtype=dtype)
        np.save(filepath, data)
    return data


class PCATrans(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        # compute PCA of points in the patch, center the patch around the mean
        pts = data['pcl_pat']
        pts_mean = pts.mean(0)
        pts = pts - pts_mean

        trans, _, _ = torch.svd(torch.t(pts))  # (3, 3)
        pts = torch.mm(pts, trans)

        # since the patch was originally centered, the original cp was at (0,0,0)
        cp_new = -pts_mean
        cp_new = torch.matmul(cp_new, trans)

        # re-center on original center point
        data['pcl_pat'] = pts - cp_new
        data['pca_trans'] = trans

        # t_1 = torch.mean(data['pcl_pat'],dim=0)
        # print("mean of ceter patch: " + str(t_1))

        if 'normal_center' in data:
            data['normal_center'] = torch.matmul(data['normal_center'], trans)
            N = data['normal_center']
            data['z-axis-direction'] = np.dot(N,[0,0,1])
    

            # print("normal to z-axis: " +str(aa))
            # trans_T = np.transpose(trans)
            # NN = torch.matmul(data['normal_center'], trans)

        if 'mnormal_center' in data:
            data['mnormal_center'] = torch.matmul(data['mnormal_center'], trans)
        if 'normal_pat' in data:
            data['normal_pat'] = torch.matmul(data['normal_pat'], trans)
        if 'mnormal_pat' in data:
            data['mnormal_pat'] = torch.matmul(data['mnormal_pat'], trans)
        # TODO
        if 'pcl_sample' in data:
            data['pcl_sample'] = torch.matmul(data['pcl_sample'], trans)
            aa = data['pcl_sample']
            # t_2 = torch.mean(aa, dim=0)
            # print("mean of global shape: " + str(t_2))

        if 'sample_near' in data:
            data['sample_near'] = torch.matmul(data['sample_near'], trans)
        if 'normal_sample' in data:
            data['normal_sample'] = torch.matmul(data['normal_sample'], trans)
        return data


class SequentialPointcloudPatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = sum(data_source.datasets.shape_patch_count)

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class RandomPointcloudPatchSampler(torch.utils.data.sampler.Sampler):
    # Randomly get subset data from the whole dataset
    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(data_source.datasets.shape_names):
            self.total_patch_count += min(self.patches_per_shape, data_source.datasets.shape_patch_count[shape_ind])

    def __iter__(self):
        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.datasets.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class PointCloudDataset(Dataset):
    def __init__(self, root, mode=None, data_set='', data_list='', sparse_patches=False):
        super().__init__()
        self.mode = mode
        self.data_set = data_set
        self.sparse_patches = sparse_patches
        self.data_dir = os.path.join(root, data_set)

        # add new information
        self.mnormals = []
        self.d= []
        self.c = []
        self.nc = []

        self.pointclouds = []
        self.shape_names = []
        self.normals = []
        self.pidxs = []
        self.kdtrees = []
        self.shape_patch_count = []   # point number of each shape
        assert self.mode in ['train', 'val', 'test']

        # get all shape names
        if len(data_list) > 0:
            cur_sets = []
            with open(os.path.join(root, data_set, 'list', data_list + '.txt')) as f:
                cur_sets = f.readlines()
            cur_sets = [x.strip() for x in cur_sets]
            cur_sets = list(filter(None, cur_sets))
        else:
            raise ValueError('Data list need to be given.')

        print('Current %s dataset:' % self.mode)
        for s in cur_sets:
            print('   ', s)

        self.get_data(cur_sets)
        self.cur_sets = cur_sets

    def __len__(self):
        return len(self.pointclouds)

    def get_data(self, cur_sets):
        for s in tqdm(cur_sets, desc='Loading data'):
            pcl = load_data(filedir=self.data_dir, filename='%s.xyz' % s, dtype=np.float32)[:, :3]

            if os.path.exists(os.path.join(self.data_dir, s + '.normals')):
                nor = load_data(filedir=self.data_dir, filename=s + '.normals', dtype=np.float32)
            else:
                nor = np.zeros_like(pcl)

            if os.path.exists(os.path.join(self.data_dir, 'modify_0.02/'+ s + '.mnormals')):
                modify_nor = load_data(filedir=self.data_dir+'/modify_0.02/', filename=s + '.mnormals', dtype=np.float32)
            else:
                modify_nor = nor
            
            if os.path.exists(os.path.join(self.data_dir, 'modify_0.02/'+ s + '.mdensity')):
                noise_density = load_data(filedir=self.data_dir+'/modify_0.02/', filename=s + '.mdensity', dtype=np.float32)
            else:
                noise_density = np.ones((pcl.shape[0]))

            if os.path.exists(os.path.join(self.data_dir, 'modify_0.02/'+ s + '.mconf')):
                confidence = load_data(filedir=self.data_dir+'/modify_0.02/', filename=s + '.mconf', dtype=np.float32)
            else:
                confidence = np.ones((pcl.shape[0]))

            if os.path.exists(os.path.join(self.data_dir, 'modify_0.02/'+ s + '.normalconf')):
                normal_confidence = load_data(filedir=self.data_dir+'/modify_0.02/', filename=s + '.normalconf', dtype=np.float32)
            else:
                normal_confidence = np.ones((pcl.shape[0]))

            self.pointclouds.append(pcl)
            self.normals.append(nor)
            self.shape_names.append(s)
            self.c.append(confidence)
            self.nc.append(normal_confidence)
            self.d.append(noise_density)
            self.mnormals.append(modify_nor)


            # KDTree construction may run out of recursions
            sys.setrecursionlimit(int(max(1000, round(pcl.shape[0]/10))))
            kdtree = spatial.cKDTree(pcl, 10)
            self.kdtrees.append(kdtree)

            if self.sparse_patches:
                pidx = load_data(filedir=self.data_dir, filename='%s.pidx' % s, dtype=np.int32)
                self.pidxs.append(pidx)
                self.shape_patch_count.append(len(pidx))
            else:
                self.shape_patch_count.append(pcl.shape[0])

    def __getitem__(self, idx):
        # kdtree uses a reference, not a copy of these points,
        # so modifying the points would make the kdtree give incorrect results!
        data = {
            'pcl': self.pointclouds[idx].copy(),
            'kdtree': self.kdtrees[idx],
            'normal': self.normals[idx],
            'pidx': self.pidxs[idx] if len(self.pidxs) > 0 else None,
            'name': self.shape_names[idx],
            'mnormal': self.mnormals[idx],
            'confidence': self.c[idx],
            'normal_confidence':self.nc[idx],
            'density': self.d[idx],
        }
        return data


class PatchDataset(Dataset):
    def __init__(self, datasets, patch_size=1, with_trans=True, sample_size=1, seed=None):
        super().__init__()
        self.datasets = datasets
        self.patch_size = patch_size
        self.trans = None
        if with_trans:
            self.trans = PCATrans()

        self.sample_size = sample_size
        self.rng_global_sample = np.random.RandomState(seed)

    def __len__(self):
        return sum(self.datasets.shape_patch_count)

    def shape_index(self, index):
        """
            Translate global (dataset-wide) point index to shape index & local (shape-wide) point index
        """
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.datasets.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset  # index in shape with ID shape_ind
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count
        return shape_ind, shape_patch_ind

    def make_patch(self, pcl, kdtree=None, nor=None, conf=None, nconf=None, density=None, mnor = None, query_idx=None, patch_size=1):


        """
        Args:
            pcl: (N, 3)
            kdtree:
            nor: (N, 3)
            query_idx: (P,)
            patch_size: K
        Returns:
            pcl_pat, nor_pat: (P, K, 3)
        """
        seed_pnts = pcl[query_idx, :]
        dists, pat_idx = kdtree.query(seed_pnts, k=patch_size)  # sorted by distance (nearest first)
        dist_max = max(dists)

        pcl_pat = pcl[pat_idx, :]        # (K, 3)
        pcl_pat = pcl_pat - seed_pnts    # center
        pcl_pat = pcl_pat / dist_max     # normlize

        nor_pat = None
        if nor is not None:
            nor_pat = nor[pat_idx, :]
        if mnor is not None:
            mnor_pat = mnor[pat_idx, :]
        if conf is not None:
            conf_pat = conf[pat_idx] 
        if nconf is not None:
            nconf_pat = nconf[pat_idx] 
        if density is not None:
            denstiy_pat = density[pat_idx]
        return pcl_pat, nor_pat, mnor_pat, conf_pat,nconf_pat, denstiy_pat, dist_max, seed_pnts 

    def make_patch_pair(self, pcl, kdtree=None, pcl_2=None, kdtree_2=None, nor=None, query_idx=None, patch_size=1, ratio=1.2):
        """
        Args:
            pcl: (N, 3)
            kdtree:
            pcl_2: (N, 3)
            kdtree_2:
            nor: (N, 3)
            query_idx: (P,)
            patch_size: K
        Returns:
            pcl_pat, nor_pat: (P, K, 3)
        """
        seed_pnts = pcl[query_idx, :]
        dists, pat_idx = kdtree.query(seed_pnts, k=patch_size)  # sorted by distance (nearest first)
        dist_max = max(dists)

        pcl_pat = pcl[pat_idx, :]            # (K, 3)
        pcl_pat = pcl_pat - seed_pnts        # center
        pcl_pat = pcl_pat / dist_max         # normlize

        dists_2, pat_idx_2 = kdtree_2.query(seed_pnts, k=patch_size*ratio)
        pcl_pat_2 = pcl_2[pat_idx_2, :]      # (K, 3)
        pcl_pat_2 = pcl_pat_2 - seed_pnts    # center
        pcl_pat_2 = pcl_pat_2 / dist_max     # normlize

        nor_pat = None
        if nor is not None:
            nor_pat = nor[pat_idx, :]
        
        return pcl_pat, pcl_pat_2, nor_pat, dist_max, seed_pnts

    def get_subsample(self, pts, query_idx, sample_size, pts_1=None, rng=None, mdist = None, seed_pnts = None, fixed=False, uniform=False):
        """
            pts: (N, 3)
            query_idx: (1,)
            Warning: the query point may not be included in the output point cloud !
        """
        N_pts = pts.shape[0]
        query_point = pts[query_idx, :]

        ### if there are too much points, it is not helpful for orienting normal.
        # N_max = sample_size * 50   # TODO
        # if N_pts > N_max:
        #     point_idx = np.random.choice(N_pts, N_max, replace=False)
        #     # if query_idx not in point_idx:
        #     #     point_idx[0] = query_idx
        #     #     query_idx = 0
        #     pts = pts[point_idx, :]
        #     if pts_1 is not None:
        #         pts_1 = pts_1[point_idx, :]
        #     N_pts = N_max

        pts = pts - query_point
        dist = np.linalg.norm(pts, axis=1)
        dist_max = np.max(dist)
        pts = pts / dist_max
        # pts = pts/mdist

        if pts_1 is not None:
            pts_1 = pts_1 - query_point
            pts_1 = pts_1 / dist_max

        if N_pts >= sample_size:
            if fixed:
                rng.seed(42)

            if uniform:
                sub_ids = rng.randint(low=0, high=N_pts, size=sample_size)
            else:
                dist_normalized = dist / dist_max
                prob = 1.0 - 1.5 * dist_normalized
                prob_clipped = np.clip(prob, 0.05, 1.0)

                ids = rng.choice(N_pts, size=int(sample_size / 1.5), replace=False)
                prob_clipped[ids] = 1.0
                prob = prob_clipped / np.sum(prob_clipped)
                sub_ids = rng.choice(N_pts, size=sample_size, replace=False, p=prob)

            # Let the query point be included
            if query_idx not in sub_ids:
                sub_ids[0] = query_idx
            pts_sub = pts[sub_ids, :]
            # id_new = np.argsort(dist[sub_ids])
            # pts_sub = pts_sub[id_new, :]
        else:
            pts_shuffled = pts[:, :3]
            rng.shuffle(pts_shuffled)
            zeros_padding = np.zeros((sample_size - N_pts, 3), dtype=np.float32)
            pts_sub = np.concatenate((pts_shuffled, zeros_padding), axis=0)

        # pts_sub[0, :] = 0    # TODO
        if pts_1 is not None:
            return pts_sub, pts_1[sub_ids, :]
        return pts_sub, sub_ids

    def __getitem__(self, idx):
        """
            Returns a patch centered at the point with the given global index
            and the ground truth normal of the patch center
        """
        ### find shape that contains the point with given global index
        shape_idx, patch_idx = self.shape_index(idx)
        shape_data = self.datasets[shape_idx]

        ### get the center point
        if shape_data['pidx'] is None:
            query_idx = patch_idx
        else:
            query_idx = shape_data['pidx'][patch_idx]

        pcl_pat, normal_pat, mnor_pat, conf_pat,nconf_pat, denstiy_pat, dist_max, seed_pnts = self.make_patch(pcl=shape_data['pcl'],
                                                                              kdtree=shape_data['kdtree'],
                                                                              nor=shape_data['normal'],
                                                                              conf =  shape_data['confidence'],
                                                                              nconf = shape_data['normal_confidence'],
                                                                              density = shape_data['density'],
                                                                              mnor = shape_data['mnormal'],
                                                                              query_idx=query_idx,
                                                                              patch_size=self.patch_size,
                                                                            )
        data = {'name': shape_data['name'],
                'pcl_pat': torch.from_numpy(pcl_pat).float(),
                'normal_pat': torch.from_numpy(normal_pat).float(),
                'normal_center': torch.from_numpy(shape_data['normal'][query_idx, :]).float(),
                'mnormal_pat': torch.from_numpy(mnor_pat).float(),
                'mnormal_center': torch.from_numpy(shape_data['mnormal'][query_idx, :]).float(),
                'conf_pat': torch.from_numpy(np.expand_dims(conf_pat,axis=1)).float(),
                'conf_center': torch.from_numpy(np.expand_dims(shape_data['confidence'][query_idx], axis=0)).float(),
                'nconf_pat': torch.from_numpy(np.expand_dims(nconf_pat,axis=1)).float(),
                'nconf_center': torch.from_numpy(np.expand_dims(shape_data['normal_confidence'][query_idx], axis=0)).float(),
                'density_pat': torch.from_numpy(np.expand_dims(denstiy_pat, axis=1)).float(),
                'density_center': torch.from_numpy(np.expand_dims(shape_data['density'][query_idx], axis=0)).float(),
            }

        if self.sample_size > 0:
            pcl_sample, sample_ids = self.get_subsample(pts=shape_data['pcl'],
                                            query_idx=query_idx,
                                            sample_size=self.sample_size,
                                            rng=self.rng_global_sample,
                                            mdist = dist_max,
                                            seed_pnts = seed_pnts,
                                            uniform=False,
                                        )
            data['pcl_sample'] = torch.from_numpy(pcl_sample).float()
            data['normal_sample'] = torch.from_numpy(shape_data['normal'][sample_ids, :]).float()

        if self.trans is not None:
            data = self.trans(data)
        else:
            data['pca_trans'] = torch.eye(3)
        return data



