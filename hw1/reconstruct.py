import pickle
import numpy as np
import open3d as o3d
import copy
from tqdm import tqdm
from pathlib import Path
import argparse
from reconstruct_utils import CameraUtils, PoseUtils, VisualizeUtils, PCDUtils

np.random.seed(13)

def depth_image_to_point_cloud(rgb, depth, INTRINSIC:np.ndarray, DEPTH_SCALE=1000.0):
    height, width = rgb.shape[0:2]
    x, y = np.arange(width), np.arange(height)
    X, Y = np.meshgrid(x, y)
    depth = depth * DEPTH_SCALE
    image_hom = np.stack((X,Y,np.ones_like(X)), axis=2) * depth[:, :, np.newaxis]
    coor = (np.linalg.inv(INTRINSIC) @ image_hom.reshape((-1, 3)).T).T
    rgb = rgb.reshape((-1, 3)) / 255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coor)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def execute_global_registration(src_down, tgt_down, src_fpfh,
                                tgt_fpfh, voxel_size):
    
    # * Global Registration Method: RANSAC
    distance_threshold = voxel_size * 20
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100, confidence=0.9)
    trans_estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.7), 
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.pi * 20 /180)
    ]
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, False,
        distance_threshold,
        trans_estimation,
        6, checkers, criteria)
    return result.transformation

def local_icp_algorithm(src_down, tgt_down, trans_init, threshold):
    # TODO: Use Open3D ICP function to implement
    reg_icp = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return reg_icp.transformation

def my_local_icp_algorithm(src_down, tgt_down, trans_init, max_iteration=30):
    # TODO: Write your own ICP function
    # * Preprocess pcd, filter out ceil & floor
    tgt_filter, low_thresh, high_thresh = PCDUtils.filter_ceil_floor(tgt_down)
    src_filter, _, _ = PCDUtils.filter_ceil_floor(copy.deepcopy(src_down).transform(trans_init), 
                                                             numeric_threshold=(low_thresh, high_thresh))
    ORIGIN_src_filter = copy.deepcopy(src_filter)
    update_trans = trans_init
    dist_min, _  = PCDUtils.nearest_neighbor(src_filter, tgt_down, k=PCDUtils.NUM_CHECK_PTS)
    prev_costs = [ np.mean(dist_min) ] 
    failed_costs = []
    trans_record = [trans_init]
    cost_change = PCDUtils.VOXEL_SIZE*1.2
    
    # Reference : https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf
    # x = [alpha, beta, gamma, tx, ty, tz]
    for i in range(max_iteration):    
        src2tgt_indx, inverse = PCDUtils.find_correspondence(src_filter, tgt_filter, 
                                                    mode='nearest')
        # Get target normal vector
        tgt_normal = PCDUtils.get_pcd_attr(tgt_filter, attr='normals', indx=src2tgt_indx[:,1]) * inverse[:, np.newaxis]
        trans = PCDUtils.point2plane_trans(src=src_filter,
                                           tgt=tgt_filter,
                                           tgt_normal=tgt_normal,
                                           src_indx=src2tgt_indx[:,0],
                                           tgt_indx=src2tgt_indx[:,1]
                                           )
        # Check result performance
        src_tmp = copy.deepcopy(src_filter).transform(trans)
        dist_min,_ = PCDUtils.nearest_neighbor(src_tmp, tgt_down, k=PCDUtils.NUM_CHECK_PTS)
        curr_cost = np.mean(dist_min)
        # Define converge
        if curr_cost > np.mean(prev_costs)*3:
            arg = np.argmin(np.array(prev_costs))
            update_trans = trans_record[arg]
            src_filter, _, _ = PCDUtils.filter_ceil_floor(copy.deepcopy(src_down).transform(update_trans), 
                                                             numeric_threshold=(low_thresh, high_thresh))
            # print("Curr_cost explodes. set trans to the one with minimum previous costs: ", prev_costs[arg])
            continue
        if curr_cost < prev_costs[-1]:
                src_filter = src_tmp
                update_trans = trans @ update_trans  
                prev_costs.append(curr_cost)
                trans_record.append(update_trans)
        elif np.abs(curr_cost - prev_costs[-1]) < cost_change and np.min(prev_costs) < cost_change:
            break
        else:
            failed_costs.append(curr_cost)
        if np.all( np.array(failed_costs[-5:]) == curr_cost) and curr_cost > PCDUtils.VOXEL_SIZE*3:
            update_trans = trans_init
            src_filter = copy.deepcopy(ORIGIN_src_filter)
    prev_costs = np.array(prev_costs)
    sort_arg = np.argsort(np.array(prev_costs))
    trans= trans_record[sort_arg[0]]
    return update_trans

def reconstruct(args):
    # TODO: Return results
    """
    For example:
        ...
        args.version == 'open3d':
            trans = local_icp_algorithm()
        args.version == 'my_icp':
            trans = my_local_icp_algorithm()
        ...
    """
    INTRINSIC = CameraUtils.get_intrinsic(height=512, width=512, FOV_x=np.pi/2, FOV_y=np.pi/2)
    data_dir = Path(args.data_root)
    img_pths = sorted( (data_dir / Path("rgb")).iterdir() , key=lambda a: int(a.stem))
    depth_pths = sorted( (data_dir / Path('depth')).iterdir(), key=lambda a: int(a.stem))
    trans2tgt_list = list()
    result_pcd = o3d.geometry.PointCloud()
    updated_trans = np.identity(4)
    pred_cam_pos = [np.array([0,0,0])]
    
    PoseRecord = PoseUtils(args.gt_path)
    for cnt, (rgbs, depths) in enumerate(
                        tqdm( zip( CameraUtils.load_image(img_pths), CameraUtils.load_depth(depth_pths)), 
                             total=len(img_pths))
    ):
        # * source:  next_frame, target: current_frame
        pc_current = depth_image_to_point_cloud(rgbs[0], depths[0], INTRINSIC=INTRINSIC)
        pc_next = depth_image_to_point_cloud(rgbs[1], depths[1], INTRINSIC=INTRINSIC)
        src, tgt = pc_next, pc_current
        src_down, src_critical, src_fpfh = PCDUtils.preprocess_point_cloud(src)
        
        tgt_down, tgt_critical, tgt_fpfh = PCDUtils.preprocess_point_cloud(tgt)
        if args.version == "gt":
            trans = PoseRecord.get_extrinsic_between(cnt+1, cnt)
        else:
            if args.global_reg == 1:
                trans_init = execute_global_registration(src_down=src_critical, 
                                                        tgt_down=tgt_critical, 
                                                        src_fpfh=src_fpfh, 
                                                        tgt_fpfh=tgt_fpfh, 
                                                        voxel_size=PCDUtils.VOXEL_SIZE)
            else:     
                trans_init = np.identity(4)
            if args.version == "open3d":
                trans = local_icp_algorithm(
                                    src_down=src_critical, 
                                    tgt_down=tgt_critical, 
                                    trans_init=trans_init, 
                                    threshold=PCDUtils.DST_MAX)    
            elif args.version == "my_icp":
                trans = my_local_icp_algorithm(
                                    src_down=src_critical, 
                                    tgt_down=tgt_critical,
                                    trans_init=trans_init, 
                                    )    
        
        updated_trans =  updated_trans @ trans 
        trans2tgt_list.append(updated_trans)
        src_remove_ceil, _, _ = PCDUtils.filter_ceil_floor(src_down, opt_low_ratio=0.18, opt_up_ratio=1.0)
        result_pcd = tgt_down if cnt == 0 else result_pcd + src_remove_ceil.transform(updated_trans)
        pred_cam_pos.append(updated_trans[:3,3])
        # print(pred_cam_pos)
        
    o3d.io.write_point_cloud(f"{args.version}_{args.floor}_{args.global_reg}.pcd", result_pcd)
    with open(f"{args.version}_trans_{args.floor}_{args.global_reg}.pkl", 'wb') as file:
        pickle.dump(trans2tgt_list, file)
    with open(f"{args.version}_cam_pos_{args.floor}_{args.global_reg}.pkl", 'wb') as file:
        pickle.dump(pred_cam_pos, file)
    return result_pcd, np.array(pred_cam_pos)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp or gt')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    parser.add_argument('--global_reg', type=int, default=1)
    args = parser.parse_args()
    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"
    args.gt_path = args.data_root + "GT_pose.npy"
    
    # TODO: Output result point cloud and estimated camera pose
    '''
    Hint: Follow the steps on the spec
    '''
    PCDUtils.update_attrs()
    import time
    t = time.time()
    result_pcd, pred_cam_pos = reconstruct(args)
    print("Reconstruction time: ", time.time() -t)
    
    # TODO: Calculate and print L2 distance
    '''
    Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    '''
    with open(f"{args.version}_cam_pos_{args.floor}_{args.global_reg}.pkl", 'rb') as file:
        pred_cam_pos = pickle.load(file)
        pred_cam_pos = np.array(pred_cam_pos)
    DEPTH_SCALE = 1000
    gt = np.load(args.gt_path)[:,:3] * DEPTH_SCALE
    gt[:,1] = gt[:, 1] *(-1)  # Align y axis & z axis as same coordinate
    gt[:,2] = gt[:, 2] *(-1)
    gt = gt - gt[0,:]# Treat the first place as origin( 0, 0, 0)
    
    m_l2 = np.mean( np.linalg.norm(gt - pred_cam_pos, axis =1) )
    print("Mean L2 distance: ", m_l2 / DEPTH_SCALE)

    # TODO: Visualize result
    '''
    Hint: Sould visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    '''
    gt_line_set = o3d.geometry.LineSet()
    gt_line_set.points = o3d.utility.Vector3dVector(gt)
    gt_lines = [[i-1, i] for i in range(1,gt.shape[0])]
    gt_line_set.lines = o3d.utility.Vector2iVector(gt_lines)
    gt_line_set.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(1, gt.shape[0])])
    
    cam_line_set = o3d.geometry.LineSet()
    cam_line_set.points = o3d.utility.Vector3dVector(pred_cam_pos)
    cam_lines = [[i-1, i] for i in range(1,pred_cam_pos.shape[0])]
    cam_line_set.lines = o3d.utility.Vector2iVector(cam_lines)
    cam_line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for i in range(1, pred_cam_pos.shape[0])])
    
    result_pcd = o3d.io.read_point_cloud(f"{args.version}_{args.floor}_{args.global_reg}.pcd")
    o3d.visualization.draw_geometries([result_pcd, gt_line_set, cam_line_set])
