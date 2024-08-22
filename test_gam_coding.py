import os, sys
import shutil
import time
import argparse
import torch
import numpy as np
import seaborn as sns 
# from net.network_modify import Network
# from net.network_qstn_noglobal import Network
# from net.network_no_gloabl_qstn_z_neighborsin import Network
# from net.network_global_qstn_zloss_neighborsin import Network
# from net.network_no_gloabl_qstn_z_neighborsin_gam import Network
# from net.network_no_gloabl_qstn_z_neighborsin_gam_coding import Network
# from net.network_no_global import Network
from net.network_no_gloabl_qstn_z_neighborsin_gam import Network
# from net.network_no_gloabl_qstn_z_neighborsin_coding_beta import Network

from misc import get_logger, seed_all
from dataset_modify_loadGF import PointCloudDataset, PatchDataset, SequentialPointcloudPatchSampler, load_data

import open3d as o3d
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default='/media/junz/volume1/dataset/normal-data/')
    parser.add_argument('--data_set', type=str, default='PCPNet') #PCPNet,FamousShape,SceneNN
    parser.add_argument('--log_root', type=str, default='/media/junz/volume1/junz/SHSNET/log')
    # parser.add_argument('--log_root', type=str, default='./log')
    #global_qstn_zloss_neighborsin_qc,
    parser.add_argument('--ckpt_dirs', type=str, default='base1best_GAM') #'240102_193803',231225_100014,240103_182530,240103_185256,240113_172122
    parser.add_argument('--ckpt_iters', type=str, default='800')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=700)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--testset_list', type=str, default='')
    parser.add_argument('--eval_list', type=str, nargs='*', help='list of .txt files containing sets of point cloud names for evaluation')
    parser.add_argument('--patch_size', type=int, default=700)
    parser.add_argument('--sample_size', type=int, default=1200)
    parser.add_argument('--encode_knn', type=int, default=16)
    parser.add_argument('--sparse_patches', type=eval, default=True, choices=[True, False],
                        help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--save_pn', type=eval, default=True, choices=[True, False])
    parser.add_argument('--is_fusion', type=eval, default=False, choices=[True, False])
    parser.add_argument('--fea_adaptive', type=eval, default=False, choices=[True, False])
    args = parser.parse_args()
    return args


def get_data_loaders(args):
    test_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='test',
            data_set=args.data_set,
            data_list=args.testset_list,
            sparse_patches=args.sparse_patches,
        )
    test_set = PatchDataset(
            datasets=test_dset,
            patch_size=args.patch_size,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    test_dataloader = torch.utils.data.DataLoader(
            test_set,
            sampler=SequentialPointcloudPatchSampler(test_set),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    return test_dset, test_dataloader


# Arguments
args = parse_arguments()


data_set = args.data_set

if data_set == 'PCPNet':
    testset_list = 'testset_PCPNet'
    # testset_list = 'test_t'
    eval_list = ['testset_no_noise', 'testset_low_noise', 'testset_med_noise', 'testset_high_noise','testset_vardensity_striped','testset_vardensity_gradient']
    # testset_list = 'testset_high_noise'
    # eval_list = ['testset_high_noise']

elif data_set == 'FamousShape':
    testset_list = 'testset_FamousShape'
    eval_list = ['testset_noise_clean', 'testset_noise_low', 'testset_noise_med', 'testset_noise_high','testset_density_stripe','testset_density_gradient']
    # testset_list = 'testset_noise_clean'
    # eval_list = ['testset_noise_clean']

elif data_set == 'SceneNN':
    testset_list = 'testset_SceneNN'
    eval_list = ['testset_SceneNN_clean', 'testset_SceneNN_noise']

elif data_set == 'Semantic3D':
    testset_list = 'testset_Semantic3D'
    eval_list = testset_list

elif data_set == 'KITTI_sub':
    testset_list = ['testset_KITTI0608']
    eval_list = testset_list

args.eval_list = eval_list
args.testset_list = testset_list



arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
print('Arguments:\n %s\n' % arg_str)

seed_all(args.seed)
PID = os.getpid()

assert args.gpu >= 0, "ERROR GPU ID!"
_device = torch.device('cuda:%d' % args.gpu)

### Datasets and loaders
test_dset, test_dataloader = get_data_loaders(args)


def normal_RMSE(normal_gts, normal_preds, eval_file='log.txt'):
    """
        Compute normal root-mean-square error (RMSE)
    """
    def l2_norm(v):
        norm_v = np.sqrt(np.sum(np.square(v), axis=1))
        return norm_v

    log_file = open(eval_file, 'w')
    def log_string(out_str):
        log_file.write(out_str+'\n')
        log_file.flush()
        # print(out_str)

    rms = []
    rms_o = []
    pgp30 = []
    pgp25 = []
    pgp20 = []
    pgp15 = []
    pgp10 = []
    pgp5 = []
    pgp_alpha = []
    P_d = []

    for i in range(len(normal_gts)):
        normal_gt = normal_gts[i]
        normal_pred = normal_preds[i]

        normal_gt_norm = l2_norm(normal_gt)
        normal_results_norm = l2_norm(normal_pred)
        normal_pred = np.divide(normal_pred, np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))
        normal_gt = np.divide(normal_gt, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))

        ### Unoriented RMSE
        ####################################################################
        nn = np.sum(np.multiply(normal_gt, normal_pred), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1

        ang = np.rad2deg(np.arccos(np.abs(nn)))

        ### portion of good points
        rms.append(np.sqrt(np.mean(np.square(ang))))
        pgp30_shape = sum([j < 30.0 for j in ang]) / float(len(ang))
        pgp25_shape = sum([j < 25.0 for j in ang]) / float(len(ang))
        pgp20_shape = sum([j < 20.0 for j in ang]) / float(len(ang))
        pgp15_shape = sum([j < 15.0 for j in ang]) / float(len(ang))
        pgp10_shape = sum([j < 10.0 for j in ang]) / float(len(ang))
        pgp5_shape  = sum([j < 5.0 for j in ang])  / float(len(ang))
        pgp30.append(pgp30_shape)
        pgp25.append(pgp25_shape)
        pgp20.append(pgp20_shape)
        pgp15.append(pgp15_shape)
        pgp10.append(pgp10_shape)
        pgp5.append(pgp5_shape)

        pgp_alpha_shape = []
        for alpha in range(30):
            pgp_alpha_shape.append(sum([j < alpha for j in ang]) / float(len(ang)))
        pgp_alpha.append(pgp_alpha_shape)

        ### Oriented RMSE
        ####################################################################
        ang_o = np.rad2deg(np.arccos(nn))   # angle error in degree
        ids = ang_o > 90.0
        p = sum(ids) / normal_pred.shape[0]

        ### if more than half of points have wrong orientation, then flip all normals
        if p > 0.5:
            nn = np.sum(np.multiply(normal_gt, -1 * normal_pred), axis=1)
            nn[nn > 1] = 1
            nn[nn < -1] = -1
            ang_o = np.rad2deg(np.arccos(nn))    # angle error in degree
            ids = ang_o > 90.0
            p = sum(ids) / normal_pred.shape[0]

        rms_o.append(np.sqrt(np.mean(np.square(ang_o))))
        print(np.sqrt(np.mean(np.square(ang_o))))
        P_d.append(p)

    avg_rms   = np.mean(rms)
    avg_rms_o = np.mean(rms_o)
    avg_pgp30 = np.mean(pgp30)
    avg_pgp25 = np.mean(pgp25)
    avg_pgp20 = np.mean(pgp20)
    avg_pgp15 = np.mean(pgp15)
    avg_pgp10 = np.mean(pgp10)
    avg_pgp5  = np.mean(pgp5)
    avg_pgp_alpha = np.mean(np.array(pgp_alpha), axis=0)
    avg_P_d  = np.mean(P_d)


    log_string('RMS per shape: ' + str(rms))
    log_string('RMS not oriented (shape average): ' + str(avg_rms))
    log_string('RMS oriented (shape average): ' + str(avg_rms_o))
    log_string('wrong direcrion:' +str(avg_P_d))

    log_string('PGP30 per shape: ' + str(pgp30))
    log_string('PGP25 per shape: ' + str(pgp25))
    log_string('PGP20 per shape: ' + str(pgp20))
    log_string('PGP15 per shape: ' + str(pgp15))
    log_string('PGP10 per shape: ' + str(pgp10))
    log_string('PGP5 per shape: ' + str(pgp5))
    log_string('PGP30 average: ' + str(avg_pgp30))
    log_string('PGP25 average: ' + str(avg_pgp25))
    log_string('PGP20 average: ' + str(avg_pgp20))
    log_string('PGP15 average: ' + str(avg_pgp15))
    log_string('PGP10 average: ' + str(avg_pgp10))
    log_string('PGP5 average: ' + str(avg_pgp5))
    log_string('PGP alpha average: ' + str(avg_pgp_alpha))
    log_file.close()

    return avg_rms, avg_rms_o,avg_P_d


def normal_RMSE_center(normal_gts, normal_preds,q_confs, eval_file='log.txt'):
    """
        Compute normal root-mean-square error (RMSE)
    """
    def l2_norm(v):
        norm_v = np.sqrt(np.sum(np.square(v), axis=1))
        return norm_v

    log_file = open(eval_file, 'w')
    def log_string(out_str):
        log_file.write(out_str+'\n')
        log_file.flush()
        # print(out_str)

    rms = []
    rms_o = []
    pgp30 = []
    pgp25 = []
    pgp20 = []
    pgp15 = []
    pgp10 = []
    pgp5 = []
    pgp_alpha = []
    P_d = []



    for i in range(len(normal_gts)):
        normal_gt = normal_gts[i]
        normal_pred = normal_preds[i]
        qq = q_confs[i]
        qq[qq>=0.6] = 1
        qq[qq<0.6] = 0
        # qq = 1-qq ## 选择 远离的点
        normal_gt_norm = l2_norm(normal_gt)
        normal_results_norm = l2_norm(normal_pred)
        normal_pred = np.divide(normal_pred, np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))
        normal_gt = np.divide(normal_gt, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))

        ### Unoriented RMSE
        ####################################################################
        nn = np.sum(np.multiply(normal_gt, normal_pred), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1
        
        ang = np.rad2deg(np.arccos(np.abs(nn)))
        ang_surf = ang[qq==1]
        ### portion of good points
        rms.append(np.sqrt(np.mean(np.square(ang_surf))))
        pgp30_shape = sum([j < 30.0 for j in ang]) / float(len(ang))
        pgp25_shape = sum([j < 25.0 for j in ang]) / float(len(ang))
        pgp20_shape = sum([j < 20.0 for j in ang]) / float(len(ang))
        pgp15_shape = sum([j < 15.0 for j in ang]) / float(len(ang))
        pgp10_shape = sum([j < 10.0 for j in ang]) / float(len(ang))
        pgp5_shape  = sum([j < 5.0 for j in ang])  / float(len(ang))
        pgp30.append(pgp30_shape)
        pgp25.append(pgp25_shape)
        pgp20.append(pgp20_shape)
        pgp15.append(pgp15_shape)
        pgp10.append(pgp10_shape)
        pgp5.append(pgp5_shape)

        pgp_alpha_shape = []
        for alpha in range(30):
            pgp_alpha_shape.append(sum([j < alpha for j in ang]) / float(len(ang)))
        pgp_alpha.append(pgp_alpha_shape)

        ### Oriented RMSE
        ####################################################################
        ang_o = np.rad2deg(np.arccos(nn))   # angle error in degree
        ids = ang_o > 90.0
        p = sum(ids) / normal_pred.shape[0]

        ### if more than half of points have wrong orientation, then flip all normals
        if p > 0.5:
            nn = np.sum(np.multiply(normal_gt, -1 * normal_pred), axis=1)
            nn[nn > 1] = 1
            nn[nn < -1] = -1
            ang_o = np.rad2deg(np.arccos(nn))    # angle error in degree
            ids = ang_o > 90.0
            p = sum(ids) / normal_pred.shape[0]

        rms_o.append(np.sqrt(np.mean(np.square(ang_o))))
        print(np.sqrt(np.mean(np.square(ang_o))))
        P_d.append(p)


    avg_rms   = np.mean(rms)
    avg_rms_o = np.mean(rms_o)
    avg_pgp30 = np.mean(pgp30)
    avg_pgp25 = np.mean(pgp25)
    avg_pgp20 = np.mean(pgp20)
    avg_pgp15 = np.mean(pgp15)
    avg_pgp10 = np.mean(pgp10)
    avg_pgp5  = np.mean(pgp5)
    avg_pgp_alpha = np.mean(np.array(pgp_alpha), axis=0)
    avg_P_d  = np.mean(P_d)


    log_string('RMS per shape: ' + str(rms))
    log_string('RMS not oriented (shape average): ' + str(avg_rms))
    log_string('RMS oriented (shape average): ' + str(avg_rms_o))
    log_string('wrong direcrion:' +str(avg_P_d))

    log_string('PGP30 per shape: ' + str(pgp30))
    log_string('PGP25 per shape: ' + str(pgp25))
    log_string('PGP20 per shape: ' + str(pgp20))
    log_string('PGP15 per shape: ' + str(pgp15))
    log_string('PGP10 per shape: ' + str(pgp10))
    log_string('PGP5 per shape: ' + str(pgp5))
    log_string('PGP30 average: ' + str(avg_pgp30))
    log_string('PGP25 average: ' + str(avg_pgp25))
    log_string('PGP20 average: ' + str(avg_pgp20))
    log_string('PGP15 average: ' + str(avg_pgp15))
    log_string('PGP10 average: ' + str(avg_pgp10))
    log_string('PGP5 average: ' + str(avg_pgp5))
    log_string('PGP alpha average: ' + str(avg_pgp_alpha))
    log_file.close()

    return avg_rms, avg_rms_o,avg_P_d



def test(ckpt_dir, ckpt_iter):
    ### Input/Output
    ckpt_path = os.path.join(args.log_root, ckpt_dir, 'ckpts/ckpt_%s.pt' % ckpt_iter)
    output_dir = os.path.join(args.log_root, ckpt_dir, 'results_vis_%s/ckpt_%s' % (args.data_set, ckpt_iter))
    if args.tag is not None and len(args.tag) != 0:
        output_dir += '_' + args.tag
    if not os.path.exists(ckpt_path):
        print('ERROR path: %s' % ckpt_path)
        return False, False

    file_save_dir = os.path.join(output_dir, 'pred_normal')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(file_save_dir, exist_ok=True)
    # pc_dir = os.path.join(output_dir, 'sample_%s' % args.data_set)
    # os.makedirs(pc_dir, exist_ok=True)

    logger = get_logger('test(%d)(%s-%s)' % (PID, ckpt_dir, ckpt_iter), output_dir)
    logger.info('Command: {}'.format(' '.join(sys.argv)))

    ### Model
    logger.info('Loading model: %s' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=_device)
    model = Network(num_pat=args.patch_size,
                    num_sam=args.sample_size,
                    encode_knn=args.encode_knn,
                    is_fusion=args.is_fusion,
                    fea_adaptive=args.fea_adaptive,
                ).to(_device)

    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # num_params = sum([np.prod(p.size()) for p in model_parameters])
    # logger.info('Num_params: %d' % num_params)
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Number of trainable parameters: %d' % trainable_num)

    model.load_state_dict(ckpt)
    model.eval()

    shape_ind = 0
    shape_patch_offset = 0
    shape_num = len(test_dset.shape_names)
    shape_patch_count = test_dset.shape_patch_count[shape_ind]

    num_batch = len(test_dataloader)
    normal_prop = torch.zeros([shape_patch_count, 3])
    normal_s = torch.zeros([shape_patch_count, 1])


    total_time = 0
    for batchind, batch in enumerate(test_dataloader, 0):
        pcl_pat = batch['pcl_pat'].to(_device)
        data_trans = batch['pca_trans'].to(_device)
        pcl_sample = batch['pcl_sample'].to(_device) if 'pcl_sample' in batch else None
        # noise_density_pat = batch['noise_density_pat'].to(_device)
        direction = batch['z-axis-direction'].to(_device)  
        direction2 = batch['z-axis-direction-gf'].to(_device)  
        gf_normal = batch['gf_normal_center'].to(_device)
        # direction*direction2

        # normal_pat = batch['normal_pat'].to(_device)


        # #### vis points
        # pcl_pat_t = batch['pcl_pat'].numpy()
        # pcl_sample_t = batch['pcl_sample'].numpy()

        # pcd1= o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(pcl_pat_t[1,...])
        
        # pcd2= o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(pcl_sample_t[1,...])

        # geometries = [pcd1,pcd2]
        # # # 显示两个点云
        # o3d.visualization.draw_geometries(geometries)


        start_time = time.time()
        with torch.no_grad():
            n_est, n_est_s, w, nn,aa_out,gg_out, pos_nn, pcl_pat_out, x_e= model(pcl_pat, pcl_sample=pcl_sample, d = direction2, gf=gf_normal, mode_test=True)
        end_time = time.time()
        elapsed_time = 1000 * (end_time - start_time)
        total_time += elapsed_time

        ## see the attention map 
        if 0: 
            pcl_pat_c = pcl_pat_out.cpu().numpy().transpose(0,2,1)
            x_e_c = x_e.cpu().numpy()
            
            save_path_p = os.path.join('/media/junz/volume1/junz/SHSNET/', 'temp_'+str(batchind) +'_points.npy')
            np.save(save_path_p, pcl_pat_c)
            save_path_x_e_c= os.path.join('/media/junz/volume1/junz/SHSNET/', 'temp_'+str(batchind) +'_x_e_c.npy')
            np.save(save_path_x_e_c, x_e_c)

            for i in range(2):
                aa_out_c = aa_out[i].cpu().numpy()
                gg_out_c = gg_out[i].cpu().numpy()
                pos_nn_c = pos_nn[i].cpu().numpy()
                save_path_aa_out_c = os.path.join('/media/junz/volume1/junz/SHSNET/', 'temp_'+str(batchind) +'_aa_out_c_'+ str(i)+'.npy')
                np.save(save_path_aa_out_c, aa_out_c)
                save_path_gg_out_c = os.path.join('/media/junz/volume1/junz/SHSNET/', 'temp_'+str(batchind) +'_gg_out_c_'+ str(i)+'.npy')
                np.save(save_path_gg_out_c, gg_out_c)
                save_path_pos_nn_c = os.path.join('/media/junz/volume1/junz/SHSNET/', 'temp_'+str(batchind) +'_pos_nn_c_'+ str(i)+'.npy')
                np.save(save_path_pos_nn_c, pos_nn_c)

        if 0:
            b_size = pcl_pat_c.shape[0]
            for j in range(30,b_size):
                for i in range(2):
                    aa_out_c = aa_out[i].cpu().numpy()
                    gg_out_c = gg_out[i].cpu().numpy()
                    pos_nn_c = pos_nn[i].cpu().numpy().transpose(0,2,3,1)
                    pcd1= o3d.geometry.PointCloud()
                    pcd1.points = o3d.utility.Vector3dVector(pcl_pat_c[j,...])
                    pcd1.paint_uniform_color([1, 0.5, 1])
                    pcd2= o3d.geometry.PointCloud()
                    pcd2.points = o3d.utility.Vector3dVector(pos_nn_c[j,0,...])
                    aaa = aa_out_c[j,0,0,...]
                    aaa = (aaa-np.min(aaa))/(np.max(aaa)-np.min(aaa))
                    colors_aa = plt.get_cmap("PuBu")(aaa)
                    pcd2.colors = o3d.utility.Vector3dVector(colors_aa[:,:3])
                    pcd3 =  o3d.geometry.PointCloud()
                    pcd3.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
                    pcd3.paint_uniform_color([0, 1, 0])
                    geometries = [pcd1, pcd2, pcd3]
                    o3d.visualization.draw_geometries(geometries, point_show_normal=False, window_name = str(j))


                    pcd4= o3d.geometry.PointCloud()
                    pcd4.points = o3d.utility.Vector3dVector(pcl_pat_c[j,...])
                    pcd4.paint_uniform_color([1, 0.5, 1])
                    pcd5= o3d.geometry.PointCloud()
                    pcd5.points = o3d.utility.Vector3dVector(pos_nn_c[j,0,...])
                    aaaa = gg_out_c[j,0,0,...]
                    # aaaa = (aaaa-np.min(aaaa))/(np.max(aaaa)-np.min(aaaa))
                    colors_aaa = plt.get_cmap("coolwarm")(aaaa)
                    pcd5.colors = o3d.utility.Vector3dVector(colors_aaa[:,:3])
                    pcd6 =  o3d.geometry.PointCloud()
                    pcd6.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
                    pcd6.paint_uniform_color([0, 1, 0])
                    geometries2 = [pcd4, pcd5, pcd6]
                    o3d.visualization.draw_geometries(geometries2, point_show_normal=False)

                plt.figure()
                # plt.imshow(x_e_c[j], cmap='RdBu')
                sns.heatmap(x_e_c[j], annot=False, cmap='coolwarm')
                # plt.colorbar()  # 显示颜色条以表示不同的数值范围
                plt.show()
                
                plt.figure()
                # plt.imshow(x_e_c[j], cmap='RdBu')
                sns.heatmap(pcl_pat_c[j], annot=False, cmap='coolwarm')
                # plt.colorbar()  # 显示颜色条以表示不同的数值范围
                plt.show()
                


                # fig,axes = plt.subplots(1,2,figsize=(8,4))
                # axes[0].imshow(x)
                # plt.title("imshow")
                # axes[1].matshow(x)
                # plt.title("matshow")
                # plt.tight_layout()
                # plt.show()
        
        ### vis 
        if 0:
            pcl_pat_t = batch['pcl_pat'].numpy()
            normal_pat_t =  batch['normal_center'].numpy()
            # n_est_t = (n_est*n_est_s.squeeze(-1)[:,None]).cpu().numpy()
            n_est_t = n_est.cpu().numpy()

            nn_t =  nn.cpu().numpy()
            w_t = w.cpu().numpy()
            pcd= o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcl_pat_t[1,:43,...])
            pcd.normals = o3d.utility.Vector3dVector(nn_t[1,...])
            colors = plt.get_cmap("hot")(w_t[1,0,...])
            pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])

            pcd_o =  o3d.geometry.PointCloud()
            pcd_o.points = o3d.utility.Vector3dVector(np.zeros((1,3)))
            pcd_o.normals = o3d.utility.Vector3dVector(n_est_t[1:2,...])

            pcd_gt =  o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(np.zeros((1,3))+[0,0,0.002])
            pcd_gt.normals = o3d.utility.Vector3dVector(normal_pat_t[1:2,...])


            geometries = [pcd,pcd_o,pcd_gt]
            o3d.visualization.draw_geometries(geometries, point_show_normal=True)




        if batchind % 5 == 0:
            batchSize = pcl_pat.size()[0]
            logger.info('[%d/%d] %s: time per patch: %.3f ms' % (
                        batchind, num_batch-1, test_dset.shape_names[shape_ind], elapsed_time / batchSize))

            # weights = weights.transpose(2, 1)                                 # (B, N, 1)
            # pcl = torch.cat([pcl_pat[:,:model.num_out,:], weights], dim=-1)   # (B, N, 4)
            # normal = pcl_pat[:,0:1,:] + n_est.unsqueeze(1) / 2          # (B, 1, 3)
            # normal = torch.cat([pcl_pat[:,0:1,:], normal], dim=1)       # (B, 2, 3)
            # # pcl = torch.cat([pcl, normal], dim=1)
            # pcl = pcl[0].cpu().detach().numpy()
            # np.savetxt(pc_dir + '/%d_pc.txt' % batchind, pcl, fmt='%.6f')
            # normal = normal[0].cpu().detach().numpy()
            # np.savetxt(pc_dir + '/%d_nor.poly' % batchind, normal, fmt='%.6f')

        if data_trans is not None:
            ### transform predictions with inverse pca rotation (back to world space)
            n_est[:, :] = torch.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)

        ### Save estimated normals to file
        batch_offset = 0
        while batch_offset < n_est.shape[0] and shape_ind + 1 <= shape_num:
            shape_patches_remaining = shape_patch_count - shape_patch_offset
            batch_patches_remaining = n_est.shape[0] - batch_offset

            ### append estimated patch properties batch to properties for the current shape on the CPU
            normal_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining), :] = \
                n_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining), :]
            
            normal_s[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining), :] = \
                n_est_s[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining),:]

            batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
            shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

            if shape_patches_remaining <= batch_patches_remaining:
                normals_to_write = normal_prop.cpu().numpy()
                normal_s_to_write = normal_s.cpu().numpy()
                # eps=1e-6
                # normals_to_write[np.logical_and(normals_to_write < eps, normals_to_write > -eps)] = 0.0

                save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '_normal.npy') # for faster reading speed
                save_path_s = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '_normal_s.npy') # for faster reading speed

                np.save(save_path, normals_to_write)
                np.save(save_path_s, normal_s_to_write)

                if args.save_pn:
                    save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '.normals')
                    save_path_s = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '.normals_s')

                    np.savetxt(save_path, normals_to_write, fmt='%.6f')
                    np.savetxt(save_path_s, normal_s_to_write, fmt='%.6f')

                logger.info('Save normal: {}'.format(save_path))
                logger.info('Total Time: %.2f sec, Shape Num: %d / %d \n' % (total_time/1000, shape_ind+1, shape_num))

                sys.stdout.flush()
                shape_patch_offset = 0
                shape_ind += 1
                if shape_ind < shape_num:
                    shape_patch_count = test_dset.shape_patch_count[shape_ind]
                    normal_prop = torch.zeros([shape_patch_count, 3])
                    normal_s = torch.zeros([shape_patch_count, 1])


    logger.info('Total Time: %.2f sec, Shape Num: %d' % (total_time/1000, shape_num))
    return output_dir, file_save_dir


def eval(normal_gt_path, normal_pred_path, output_dir):
    print('\n  Evaluation ...')
    eval_summary_dir = os.path.join(output_dir, 'summary')
    os.makedirs(eval_summary_dir, exist_ok=True)

    all_avg_rms = []
    all_avg_rms_o = []
    all_avg_P_d = []
    for cur_list in args.eval_list:
        print("\n***************** " + cur_list + " *****************")
        print("Result path: " + normal_pred_path)

        ### get all shape names in the list
        shape_names = []
        normal_gt_filenames = os.path.join(normal_gt_path, 'list', cur_list + '.txt')
        with open(normal_gt_filenames) as f:
            shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))

        ### load all shape data of the list
        normal_gts = []
        q_confs = []
        normal_preds = []
        for shape in shape_names:
            print(shape)
            normal_pred = np.load(os.path.join(normal_pred_path, shape + '_normal.npy'))                  # (n, 3)
            points_idx = load_data(filedir=normal_gt_path, filename=shape + '.pidx', dtype=np.int32)      # (n,)
            normal_gt = load_data(filedir=normal_gt_path, filename=shape + '.normals', dtype=np.float32)  # (N, 3)
            # normal_gt = load_data(filedir=normal_gt_path+'/modify', filename=shape + '.mnormals', dtype=np.float32)  # (N, 3)
            q_conf=  load_data(filedir=normal_gt_path+'/modify', filename=shape + '.mconf', dtype=np.float32)  # (N, 1) 

            points_idx = points_idx.astype(int)
            normal_gt = normal_gt[points_idx.astype(int), :]
            q_conf = q_conf[points_idx.astype(int)]
            if normal_pred.shape[0] > normal_gt.shape[0]:
                normal_pred = normal_pred[points_idx, :]
            normal_gts.append(normal_gt)
            q_confs.append(q_conf)
            normal_preds.append(normal_pred)
            

        ## compute RMSE per-list
        avg_rms, avg_rms_o,avg_P_d= normal_RMSE(normal_gts=normal_gts,
                            normal_preds=normal_preds,
                            eval_file=os.path.join(eval_summary_dir, cur_list + '_evaluation_results.txt'))
        
        # avg_rms, avg_rms_o,avg_P_d= normal_RMSE_center(normal_gts=normal_gts,
        #                     normal_preds=normal_preds,
        #                     q_confs = q_confs,
        #                     eval_file=os.path.join(eval_summary_dir, cur_list + '_evaluation_results.txt'))        
        
        all_avg_rms.append(avg_rms)
        all_avg_rms_o.append(avg_rms_o)
        all_avg_P_d.append(avg_P_d)

        print('### RMSE: %f' % avg_rms)
        print('### RMSE_Ori: %f' % avg_rms_o)
        print('### Direction_W: %f' % avg_P_d)


    s = '\n {} \n All RMS not oriented (shape average): {} | Mean: {}\n'.format(
                normal_pred_path, str(all_avg_rms), np.mean(all_avg_rms))
    print(s)

    s = '\n {} \n All RMS oriented (shape average): {} | Mean: {}\n'.format(
                normal_pred_path, str(all_avg_rms_o), np.mean(all_avg_rms_o))
    print(s)

    s = '\n {} \n All Direction Wrong (shape average): {} | Mean: {}\n'.format(
                normal_pred_path, str(all_avg_P_d), np.mean(all_avg_P_d))
    print(s)

    ### delete the normal files
    if not args.save_pn:
        shutil.rmtree(normal_pred_path)
    return all_avg_rms, all_avg_rms_o, all_avg_P_d



if __name__ == '__main__':
    ckpt_dirs = args.ckpt_dirs.split(',')
    ckpt_iters = args.ckpt_iters.split(',')
    

    for ckpt_dir in ckpt_dirs:
        eval_dict = ''
        sum_file = 'eval_' + args.data_set + ('_'+args.tag if len(args.tag) != 0 else '')
        log_file_sum = open(os.path.join(args.log_root, ckpt_dir, sum_file+'.txt'), 'a')
        log_file_sum.write('\n====== %s ======\n' % args.eval_list)

        for ckpt_iter in ckpt_iters:
            output_dir, file_save_dir = test(ckpt_dir=ckpt_dir, ckpt_iter=ckpt_iter)
            if not output_dir or args.data_set == 'Semantic3D' or args.data_set == 'KITTI_sub':
                continue
            all_avg_rms, all_avg_rms_o , all_avg_P_d= eval(normal_gt_path=os.path.join(args.dataset_root, args.data_set),
                                                normal_pred_path=file_save_dir,
                                                output_dir=output_dir)

            s = '%s: %s | Mean: %f \t|| %s | Mean: %f\t|| %s | Mean: %f \n' % (ckpt_iter, str(all_avg_rms), np.mean(all_avg_rms),
                                                                    str(all_avg_rms_o), np.mean(all_avg_rms_o),
                                                                    str(all_avg_P_d),np.mean(all_avg_P_d))
            log_file_sum.write(s)
            log_file_sum.flush()
            eval_dict += s

        log_file_sum.close()
        s = '\n All RMS not oriented and oriented (shape average): \n{}\n'.format(eval_dict)
        print(s)