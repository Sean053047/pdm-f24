import open3d as o3d
import numpy as np 
from sklearn.neighbors import NearestNeighbors

def get_pcd_from_xyz_color(xyz:np.ndarray, color:np.ndarray, COLOR_SCALE=1/255):
    color = color * COLOR_SCALE
    assert len(xyz) == len(color), f"Array shape doesn't matched. {xyz.shape} to {color.shape}"
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(color)
    return pc

def filter_ceiling_floor( pc, f_indx=None, ratios=None, numeric_threshold=None, return_indx=False, col=1):
    points = np.array(pc.points)
    if numeric_threshold is None:
        assert numeric_threshold is not None or ratios is not None, "Wrong Input arguments."
        min_y, max_y = np.min(points[:,1]), np.max(points[:,1])
        low_ratio, up_ratio = ratios
        low_threshold = low_ratio * max_y + (1-low_ratio) * min_y
        high_threshold = up_ratio * max_y + (1-up_ratio) * min_y
    else:
        low_threshold, high_threshold = numeric_threshold
    
    outof_floor_ceiling = np.bitwise_or( points[:,col] < low_threshold , points[:,col] >high_threshold)
    if f_indx is not None:
        pc_indx_mask = np.zeros_like(outof_floor_ceiling)
        pc_indx_mask[f_indx] = 1
        invalid = np.bitwise_and(outof_floor_ceiling, pc_indx_mask)
    else:
        invalid = outof_floor_ceiling
    valid_indx = np.where(~invalid)[0]
    result_pcd = pc.select_by_index(valid_indx)
    if not return_indx:
        return result_pcd, low_threshold, high_threshold
    else:
        return result_pcd, low_threshold, high_threshold, valid_indx

def filter_color( pc, remove_colors:list):
    if type(remove_colors) is not list:
        remove_colors = [remove_colors]
    remove_colors = np.array(remove_colors).reshape((-1, 3)) 
    pc_colors = np.array(pc.colors) *255
    matched = np.any(
        np.all(pc_colors[:,None, :] == remove_colors[None, :, :], axis=2),
        axis=1,
    )
    extract = np.where(~matched)[0]
    result = pc.select_by_index(extract)
    return result, np.where(matched)[0]

def nearest_neighbor(src:np.ndarray, tgt:np.ndarray, k=1):
        neigh = NearestNeighbors(n_neighbors=k, algorithm ='kd_tree')
        neigh.fit(tgt)
        dist_min, tgt_indx = neigh.kneighbors(src, return_distance=True)
        if k != 1:
            dist_min = np.mean(dist_min, axis=1)
            # print(dist_min)
        return dist_min.ravel(), tgt_indx.ravel()

def find_distance(  src:np.ndarray, 
                    tgt:np.ndarray, 
                    src_indx=None,
                    tgt_indx=None,
                    dim=2):
        ''' src -> tgt '''
        src_xyz = src[src_indx, :].reshape((-1, 1, dim))
        tgt_xyz = tgt[tgt_indx, :].reshape(( 1,-1, dim))
        return np.linalg.norm(src_xyz - tgt_xyz, ord=2, axis=2)
