import copy
import random
from pathlib import Path
from typing_extensions import Union

import cv2 
import open3d as o3d
import numpy as np 
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
np.random.seed(37)
random.seed(37)

def inverse_softmax(costs):
    neg_costs = -np.array(costs)
    e_x = np.exp(neg_costs - np.max(neg_costs))
    return e_x / e_x.sum()

class VisualizeUtils:
    @staticmethod
    def draw_registration_result(src, tgt, transformation, geometry_type="pcd", title=None, voxel_size=None):
        src_temp = copy.deepcopy(src)
        tgt_temp = copy.deepcopy(tgt)
        src_temp.transform(transformation)
        if geometry_type == "pcd":
            final_result = src_temp + tgt_temp            
        elif geometry_type == "voxel":
            assert voxel_size is not None, "voxel_size should not be None for voxel type."
            final_result = src_temp + tgt_temp
            final_result = o3d.geometry.VoxelGrid.create_from_point_cloud(final_result, voxel_size)
        else:
            raise f"Wrong geometry_type: {geometry_type}"
        o3d.visualization.draw_geometries([final_result],window_name=title if title is not None else "window")
    
    @staticmethod
    def draw_unified_result(result, geometry_type='voxel', voxel_size =None):
        points = np.array(result.points)
        colors = np.array(result.colors)
        ceil_y = np.percentile(points[:,1], 30)
        valid_range = points[:, 1] >= ceil_y
        
        final_result = o3d.geometry.PointCloud()
        final_result.points = o3d.utility.Vector3dVector(points[valid_range, :])
        final_result.colors = o3d.utility.Vector3dVector(colors[valid_range, :])
        if geometry_type == 'voxel':
            assert voxel_size is not None, f"Wrong value of voxel_size: {voxel_size}"
            final_result = o3d.geometry.VoxelGrid.create_from_point_cloud(final_result, voxel_size)
        
        o3d.visualization.draw_geometries([final_result], window_name='final')
    
    @staticmethod
    def draw_pcd(pcd:o3d.visualization.draw_geometries, specified_indx=None, vector=None,  bias=None, color=None, voxel_size=None):
        
        pcds = [pcd] if type(pcd) is not list else pcd
        num_pcd = len(pcds)
        specified_indxes = [specified_indx for _ in range(num_pcd)] if type(specified_indx) is not list else specified_indx
        vectors = [vector for _ in range(num_pcd)] if type(vector) is not list else vector
        biases = [bias for _ in range(num_pcd)] if type(bias) is not list else bias
        colors = [color for _ in range(num_pcd)] if type(color) is not list else color

        final_draw = o3d.geometry.PointCloud()       
        for pcd, specified_indx, vector, bias, color in zip(pcds, specified_indxes, vectors, biases, colors):
            pcd_xyz = np.array(pcd.points)
            pcd_color = np.array(pcd.colors)
            
            if color is not None:
                if specified_indx is not None:
                    pcd_color[specified_indx,:] = color
                else:
                    pcd_color[:, :] = color
                
            if vector is not None and bias is not None:
                color = color if color is not None else [1, 0, 0]
                ratios = np.arange(-PCDUtils.VOXEL_SIZE*6, PCDUtils.VOXEL_SIZE*6, PCDUtils.VOXEL_SIZE/4).reshape((-1, 1))
                vector = vector.reshape((1, -1))
                vector_pts = ratios @ vector + bias
                vector_colors = np.repeat( np.array(color).reshape(-1,3), vector_pts.shape[0], axis=0)
                pcd_xyz = np.vstack((pcd_xyz, vector_pts))
                pcd_color = np.vstack((pcd_color, vector_colors))
                
            draw_pcd = o3d.geometry.PointCloud()
            draw_pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
            draw_pcd.colors = o3d.utility.Vector3dVector(pcd_color)
            final_draw = final_draw + draw_pcd
            
        if voxel_size is not None:
            final_draw = o3d.geometry.VoxelGrid.create_from_point_cloud(final_draw, voxel_size)
        o3d.visualization.draw_geometries([final_draw])
        
class CameraUtils:
    @staticmethod    
    def load_image(pths:Union[list, str], mode=2, color='rgb'):
        def im_read(pth:str):
            m = cv2.imread(pth)
            return cv2.cvtColor(m, cv2.COLOR_BGR2RGB) if color =='rgb' else m
        
        current_img = im_read(pths[0])
        if mode == 1:
            return current_img
        if mode == 2: 
            for pth in pths[1:]:
                next_img = im_read(pth)
                yield (current_img, next_img)            
                current_img = next_img
    
    @staticmethod
    def load_depth(pths:Union[list, str], mode=2):
        if type(pths) is str: 
            pths = list(pths)
        suffix = Path(pths[0]).suffix
        
        def depth_read(pth:str):
            if suffix == '.npy': return np.load(pth)
            elif suffix == '.png': return cv2.imread(pth, cv2.IMREAD_GRAYSCALE)
            else: raise f"Wrong depth data suffix: {suffix}"
        current_depth = depth_read(pths[0])
        if mode == 1:
            return current_depth
        if mode == 2:
            for pth in pths[1:]:
                next_depth = depth_read(pth)
                yield (current_depth, next_depth)            
                current_depth = next_depth
    
    @staticmethod
    def get_intrinsic(height, width, FOV_x, FOV_y):
        cx = (width-1) / 2
        cy = (height-1) / 2
        fx = width / (2 * np.tan( FOV_x /2))
        fy = height / (2 * np.tan( FOV_y /2))
        intrinsic = np.array([  [     fx,    0,    cx],
                                [    0.0,   fy,   cy],
                                [    0.0,  0.0,          1]], dtype=np.float64)
        return intrinsic

class PoseUtils:
    def __init__(self, gt_path:str):
        self.GT_POSES = np.load(gt_path)
        
    def __len__(self):
        return self.GT_POSES.shape[0]
    
    def get_extrinsic_between(self, src_indx, target_indx):
        # x, y, z, rw, rx, ry, rz
        # From indx1 to indx2
        x1, y1, z1, rw1, rx1, ry1, rz1 = self.GT_POSES[src_indx, :]
        x2, y2, z2, rw2, rx2, ry2, rz2 = self.GT_POSES[target_indx, :]
        
        quat1 = Rotation.from_quat([rx1, ry1, rz1, rw1])
        quat2 = Rotation.from_quat([rx2, ry2, rz2, rw2])
        quat_rel = quat2 * quat1.inv()
        rotation = quat_rel.as_matrix()
        translation = np.array([(x2-x1),(y2-y1), (z2 - z1)]).reshape((-1, 1))  * 100
        extrinsic = np.vstack(
                            (  np.hstack(
                                (rotation, translation)
                                ), np.array([0,0,0,1]))
                        )
        return extrinsic
    
    @classmethod
    def get_extrinsic(cls, alpha, beta, gamma, tx, ty, tz):
        R = cls.get_rotation(alpha, beta, gamma)
        T = np.array([tx, ty, tz]).reshape((-1, 1))
        extrinsic = np.vstack(
            ( np.hstack((R,T)), np.array([0,0,0,1]).reshape((1,-1)))
        )
        return extrinsic
    
    @ staticmethod
    def get_rotation(alpha, beta, gamma):
        ''' 
        alpha: rotation around x axis
        beta: rotation around y axis 
        gamma: rotation around z axis
        all theta is expresses as radians.
        '''
        # Rx = np.array([ [1, 0, 0], 
        #                 [0, np.cos(alpha), -np.sin(alpha)], 
        #                 [0, np.sin(alpha),  np.cos(alpha)]]  )
        # Ry = np.array([ [ np.cos(beta), 0, np.sin(beta)],
        #                 [            0, 1,            0],
        #                 [-np.sin(beta), 0, np.cos(beta)]])
        
        # Rz = np.array([ [ np.cos(gamma), -np.sin(gamma), 0], 
        #                 [ np.sin(gamma),  np.cos(gamma), 0],
        #                 [             0,              0, 1]])
        # R = Rz @ Ry @ Rx
        return Rotation.from_rotvec([alpha, beta, gamma]).as_matrix()

class PCDUtils:   
    PLANE_PTS :int
    CORR_PTS : int
    NUM_CHECK_PTS :int

    VOXEL_SIZE :float
    NORMAL_COS_THRESH :float
    DST_MAX :float

    PLANE_RANGE_RATIO:float
    FEATURE_RATIO:float
    up_ratio:float
    low_ratio:float
    @staticmethod
    def get_pcd_attr(pcd:Union[o3d.geometry.PointCloud, np.ndarray], attr='points', indx=None):
        assert type(pcd) is o3d.geometry.PointCloud or type(pcd) is np.ndarray, "Wrong type for pcd."
        result = np.array( getattr(pcd , attr) )  if type(pcd) is o3d.geometry.PointCloud else pcd
        if indx is not None:
            result = result[indx, :]
        return result
    
    @classmethod
    def update_attrs(cls, attrs=None):
        if attrs is None:
            attrs = {
                "PLANE_PTS": 4,
                "CORR_PTS": 5,
                "NUM_CHECK_PTS": 8,
                "VOXEL_SIZE": 50,
                "NORMAL_COS_THRESH": np.cos(20 / 180 * np.pi),
                "DST_MAX": 500,
                "PLANE_RANGE_RATIO":2,
                "FEATURE_RATIO": 8,
                "up_ratio":0.95,
                "low_ratio":0.05,
            }
        for k, v in attrs.items():
            setattr(cls, k, v)
        return attrs
    
    @classmethod    
    def preprocess_point_cloud(cls, pcd:o3d.geometry.PointCloud):
        # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
        pcd_down = pcd.voxel_down_sample(cls.VOXEL_SIZE)
        pcd_critical, indx = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.8)
        
        radius_normal = cls.VOXEL_SIZE * cls.PLANE_RANGE_RATIO
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=20))
        pcd_critical.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=20))
        radius_feature = cls.VOXEL_SIZE * cls.FEATURE_RATIO
        
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_critical,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=40)
        )
        return pcd_down, pcd_critical, pcd_fpfh

    @classmethod
    def filter_ceil_floor(cls, pcd, numeric_threshold=None, return_indx=False, opt_up_ratio=None, opt_low_ratio=None):
        points = cls.get_pcd_attr(pcd, attr='points')
        if numeric_threshold is None:
            min_y, max_y = np.min(points[:,1]), np.max(points[:,1])
            low_ratio = cls.low_ratio if opt_low_ratio is None else opt_low_ratio
            up_ratio = cls.up_ratio if opt_up_ratio is None else opt_up_ratio

            low_threshold = low_ratio * max_y + (1-low_ratio) * min_y
            high_threshold = up_ratio * max_y + (1-up_ratio) * min_y
        else:
            low_threshold, high_threshold = numeric_threshold
        valid_indx = np.where(np.bitwise_and( points[:,1] >= low_threshold , points[:,1] <=high_threshold))[0]
        result_pcd = pcd.select_by_index(valid_indx)
        if not return_indx:
            return result_pcd, low_threshold, high_threshold
        else:
            return result_pcd, low_threshold, high_threshold, valid_indx
        
    @classmethod
    def find_correspondence(cls, src, tgt, 
                            mode='nearest'):
        src_xyz = cls.get_pcd_attr(src, attr='points')
        tgt_xyz = cls.get_pcd_attr(tgt, attr='points')
        
        result_src2tgt = np.empty((0, 2))
        result_inverse = np.empty((0, ))
        
        num_src = len(src_xyz)
        available_indx = set(range(num_src))
        
        while result_src2tgt.shape[0] < cls.CORR_PTS and len(available_indx) > cls.CORR_PTS:
            
            cos_vals = cls.find_close_cos_val(src=src, tgt=src, tgt_indx=random.sample(available_indx, 1)).flatten()
            array_args = np.arange(len(cos_vals))
            close_pts = array_args[cos_vals < np.cos(np.pi * 30 /180)]
            src_indx =np.random.choice(close_pts.flatten(), size=cls.CORR_PTS)
            # src_indx = random.sample(available_indx, k=cls.CORR_PTS)
            available_indx = available_indx - set(src_indx)
                
            if mode =="nearest":
                dist_min, tgt_indx = cls.nearest_neighbor( src_xyz[src_indx], tgt_xyz)
                valid = dist_min < cls.DST_MAX
                src2tgt_indx = np.vstack( (src_indx, tgt_indx)).T
                
            # Check angle between two corresponding vector is closed enough.
            src_normal = cls.get_pcd_attr(src, attr='normals', indx=src2tgt_indx[:,0])
            tgt_normal = cls.get_pcd_attr(tgt, attr='normals', indx=src2tgt_indx[:,1])
            cos_val = cls.get_cos_val(src_normal, tgt_normal)
            inverse_normal = np.sign(cos_val)
            valid = np.bitwise_and(valid, np.abs(cos_val) > cls.NORMAL_COS_THRESH)
            result_src2tgt = np.vstack( ( result_src2tgt, src2tgt_indx[valid].reshape((-1, 2))))
            result_inverse = np.hstack( ( result_inverse, inverse_normal[valid]))
            # VisualizeUtils.draw_pcd(
            #     [src, tgt],
            #     [src2tgt_indx[0,0], src2tgt_indx[0,1]],
            #     [src_normal, tgt_normal],
            #     [src_xyz[src2tgt_indx[0,0]], tgt_xyz[src2tgt_indx[0,1]]],
            #     [(1,0,0), (0,1,0)]
            # )
        result_src2tgt = result_src2tgt[:cls.CORR_PTS, :]
        result_inverse = result_inverse[:cls.CORR_PTS]
        return result_src2tgt.astype(dtype=np.int32), result_inverse

    @classmethod
    def nearest_neighbor(cls, src:np.ndarray, tgt:np.ndarray, k=1):
        src = cls.get_pcd_attr(src, attr='points')
        tgt = cls.get_pcd_attr(tgt, attr='points')
        neigh = NearestNeighbors(n_neighbors=k, algorithm ='kd_tree')
        neigh.fit(tgt)
        dist_min, tgt_indx = neigh.kneighbors(src, return_distance=True)
        if k != 1:
            dist_min = np.mean(dist_min, axis=1)
            # print(dist_min)
        return dist_min.ravel(), tgt_indx.ravel()
                
    @classmethod
    def find_distance(  cls,
                        src:Union[o3d.geometry.PointCloud, np.ndarray], 
                        tgt:Union[o3d.geometry.PointCloud, np.ndarray], 
                        src_indx=None,
                        tgt_indx=None):
        ''' src -> tgt '''
        src_xyz = cls.get_pcd_attr(src, attr='points', indx =src_indx).reshape((-1, 1, 3))
        tgt_xyz = cls.get_pcd_attr(tgt, attr='points', indx =tgt_indx).reshape(( 1,-1, 3))
        return np.linalg.norm(src_xyz - tgt_xyz, ord=2, axis=2)
    
    @classmethod
    def find_plane(cls, pcd:Union[o3d.geometry.PointCloud, np.ndarray], pcd_indx):
        pcd_xyz = cls.get_pcd_attr(pcd, attr='points')
        dist = cls.find_distance(pcd_xyz, pcd_xyz, src_indx=pcd_indx)
        
        sorted_dist = np.sort(dist, axis=1)
        sorted_dist_arg = np.argsort(dist, axis=1)
        # valid_pts_indx = sorted_dist_arg[sorted_dist < ]
        PLANE_SEARCH_RANGE = cls.PLANE_RANGE_RATIO * cls.VOXEL_SIZE
        result_planes = np.empty((0, 4))
        valid_pts_indxes = np.where(sorted_dist < PLANE_SEARCH_RANGE, sorted_dist_arg, np.nan)
        
        for row in range(valid_pts_indxes.shape[0]):
            pts_indxes = valid_pts_indxes[row,:]
            pts_indxes = pts_indxes[np.bitwise_not(np.isnan(pts_indxes))].astype(np.int64)
            pts = pcd_xyz[pts_indxes, :]
            
            center = np.mean(pts, axis=0)
            centered_pts = pts - center
            _, _, vh = np.linalg.svd(centered_pts)
            normal_vector = vh[-1,:] / np.linalg.norm(vh[-1,:])
            d = -np.dot(center, normal_vector)
            plane_par = np.array([*normal_vector, d])
            result_planes = np.vstack((result_planes, np.array(plane_par)))
            # VisualizeUtils.draw_pcd(pcd, pts_indxes, normal_vector, pts[0,:])
        
        return result_planes
    
    @classmethod
    def find_close_cos_val(cls, src, tgt, src_indx=None, tgt_indx=None):
        tgt_normals = cls.get_pcd_attr(tgt, attr="normals", indx=tgt_indx)
        src_normals = cls.get_pcd_attr(src, attr="normals", indx=src_indx)
        cos_vals = np.empty((tgt_normals.shape[0], src_normals.shape[0]))
        for i in range(tgt_normals.shape[0]):
            cos_val = np.dot(src_normals, tgt_normals[i,:]) / (np.linalg.norm(src_normals, axis=1) * np.linalg.norm(tgt_normals[i,:]))
            cos_vals[i, :] =  cos_val
        return cos_vals
        
    @staticmethod
    def get_cos_val(n1, n2):
        n_dot = np.diag( n1 @ n2.T)
        scale = np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1)
        cos_val = n_dot /scale
        return cos_val
        
    @classmethod
    def point2plane_trans(cls, src, tgt, tgt_normal, src_indx=None, tgt_indx=None ):
        '''Mathematic: Using SVD
        U, S, VT = np.linalg.svd(A)
        Sigma = np.zeros((U.shape[1], VT.shape[0]))
        np.fill_diagonal(Sigma, S)
        Sigma[Sigma > 1e-10] = 1/ Sigma[Sigma > 1e-10]
        Sigma_plus = Sigma.T
        pseudo_A = VT.T @ Sigma_plus @ U.T
        '''
        src_xyz = cls.get_pcd_attr(src, attr='points', indx=src_indx)
        tgt_xyz = cls.get_pcd_attr(tgt, attr='points', indx=tgt_indx)
        A = np.hstack( (np.cross( src_xyz, tgt_normal), tgt_normal))
        b = np.diag(np.dot(tgt_xyz, tgt_normal.T)) - np.diag( np.dot(src_xyz, tgt_normal.T))
        pseudo_A = np.linalg.pinv(A)
        x = pseudo_A @ b
        
        return PoseUtils.get_extrinsic(*x)


