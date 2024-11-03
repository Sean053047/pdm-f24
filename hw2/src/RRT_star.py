import numpy as np 
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from pcd_utils import nearest_neighbor, find_distance
from render_utils import show_image, draw_sample
np.random.seed(18)
class Node(object):
    def __init__(self, 
                 id,
                 position=None,
                ):
        self.id = id
        self.position = position.astype(np.int32).reshape(2) # (y,x)
        self.parent = None
        self.childs:list[Node] = list()
        self.cost = 0.0
        
    def add_child(self, other):
        assert other not in self.childs, "add_child error."
        self.childs.append(other)
    
    def remove_child(self, id):
        indx = [ i for i, ch in enumerate(self.childs) if ch.id == id][0]
        self.childs.pop(indx)
        
    def set_parent(self, other):
        self.parent = other        
        self.update_cost()
        
    def update_cost(self):
        if self.parent is not None:
            self.cost = np.linalg.norm(self.position - self.parent.position) + self.parent.cost 
        for child in self.childs:    
            child.update_cost()
        
    def __repr__(self):
        return f"Node{self.id}, {self.position}"

class RRT_star:
    def __init__(self, 
                 C_space, 
                 step:float,
                 max_search_dist:float,
                 bias:float = 1.5
                 ) -> None:
        self.C_space = C_space
        self.STEP = step
        self.MAX_DIST = max_search_dist
        self.bias = bias
        
        self.likelihood = np.zeros_like(C_space, dtype=np.float64)
        self.nodes_book = dict() # Record id for each node
        self.gkernel = self.__get_gaussian_kernel(31)
        self.__mesh = np.stack(
            np.meshgrid(
                np.arange(C_space.shape[1]), 
                np.arange(C_space.shape[0])
            )
            , axis=2)
        self.__mesh = self.__mesh[:, :, [1,0]]
        self.id_occupy = np.concatenate((
                                self.__mesh.copy(), 
                                np.ones_like(C_space, dtype=np.int32)[:,:,None]*-2), 
                                axis=2)
        self.prob_sigma = 15
        
    def __get_gaussian_kernel(self, var):
        size = var//2
        X,Y = np.meshgrid(np.arange(-size, size+1, 1), np.arange(-size, size+1, 1))
        gaussian_kernel = np.exp(-(X**2 + Y **2) / (2*var))
        return gaussian_kernel
    
    def random_sample(self, sample_mode,):
        tmp_likelihood = self.likelihood.copy()
        if sample_mode == "diverse":
            end_points_yx = np.array([ v.position for v in self.nodes_book.values() if len(v.childs) == 0])
            tmp_likelihood[end_points_yx[:,0], end_points_yx[:,1]]= 0 
            L = gaussian_filter(tmp_likelihood, sigma=self.prob_sigma)
            sample_prob = ( 1 - L / np.max(L)) if np.sum(L) != 0  else np.ones_like(L)
            cv2.circle(sample_prob, self.nodes_book[-1].position[[1,0]], int(self.STEP), self.bias, thickness=-1)
        elif sample_mode == "centralized":
            if np.sum(tmp_likelihood) != 0 :
                tmp_likelihood = np.where(
                    tmp_likelihood != 0, 
                    np.sum(tmp_likelihood) - tmp_likelihood,
                    tmp_likelihood
                )
            sample_prob = gaussian_filter(tmp_likelihood, sigma=self.prob_sigma)
        elif sample_mode == "random":
            sample_prob = np.ones_like(tmp_likelihood) 
        elif sample_mode == "goal_bias":
            sample_prob = np.ones_like(tmp_likelihood)
            cv2.circle(sample_prob, self.nodes_book[-1].position[[1,0]], int(self.STEP), self.bias, thickness=-1)
            
        sample_prob[self.C_space != 0] = 0.0
        sample_prob = sample_prob / np.sum(sample_prob)
        mesh = self.__mesh.copy()
        P = np.concatenate((mesh, sample_prob[:,:,None]), axis=2).reshape(-1, 3)
        indx = np.random.choice(len(P), p= P[:,-1])
        
        return P[indx, :2].astype(np.int32)
    
    def nearest(self, position, NODE=False) -> np.ndarray:
        row, col = np.where(self.id_occupy[:, :, 2] >=0) 
        src = np.stack((row, col), axis=1).astype(np.int32)
        _, indx = nearest_neighbor(position.reshape((-1, 2)),src)
        
        result = src[indx]
        if NODE:
            id = self.id_occupy[result.flatten()[0], result.flatten()[1], 2]
            return self.nodes_book[id]
        else:
            return result
    
    def steer(self, src_yx:np.ndarray, tgt_yx:np.ndarray):
        '''Using lowest cost as standard. Not just the nearest in Euclidean distance.'''
        diff = tgt_yx - src_yx
        dist = np.linalg.norm(diff)
        if  dist <= self.STEP:
            return tgt_yx
        else:
            step_yx = np.round(diff / dist * self.STEP + src_yx)
            return step_yx.astype(np.int32)
        
    def check_obstacle_free(self, src, tgt, IGNORE_NONE=False) -> bool:
        '''src: new_node, tgt: exist node'''
        assert  (IGNORE_NONE and (src is None or tgt is None))  or \
                (tgt is not None and src is not None), "Error for check_obstacle_free."
        if IGNORE_NONE and (src is None or tgt is None):
            # Ignore None
            return True
        
        tgt_yx = (tgt.position if type(tgt) is Node else tgt).reshape(2)
        src_yx = (src.position if type(src) is Node else src).reshape(2)
        line_check = np.zeros_like(self.C_space)
        cv2.line(line_check, tgt_yx[[1,0]], src_yx[[1,0]], 255, thickness=3)
        collide =np.sum(np.bitwise_and(line_check>0, self.C_space>0))
        
        return True if collide == 0 else False
    
    def check_obstacle_free_for_1pt(self, pt)->bool:
        pt_yx = (pt.position if type(pt) is Node else pt).reshape(2)
        return True if self.C_space[pt_yx[0], pt_yx[1]] == 0 else False
    
    def check_occupied(self, yx) -> bool:
        yx = yx.reshape(2)
        return True if self.id_occupy[yx[0], yx[1], 2] != (-2) else False
    
    def check_reach_end(self) -> bool:
        goal_node = self.nodes_book[-1]
        goal_yx = goal_node.position.reshape(-1,2)
        # TODO: In more general way...
        record_nodes = [ node for node in self.nodes_book.values() if node.id != -1]
        
        tgt = np.array([n.position for n in record_nodes])
        dist, indx = nearest_neighbor(goal_yx, tgt)
        tgt_node = record_nodes[indx[0]]
            
        if dist <= self.STEP and self.check_obstacle_free(goal_yx, tgt_node):
            self.__connect_node(tgt_node, goal_node)
            return True
        else:
            return False
                
    def update_refine_likelihood(self) :
        route = np.array( [n.position for n in  self.get_route_node()], dtype=np.int32)  
        mask = np.zeros_like(self.likelihood)
        mask[route[:,0], route[:,1]] = 1
        mask = gaussian_filter(mask, sigma=self.prob_sigma) > 0
        self.likelihood[~mask] = 0
        
    def update_rrt_record(self, nodes):
        if type(nodes) is not list:
            nodes = [nodes]
        for n in nodes:
            self.nodes_book[n.id] = n
            y, x = n.position
            self.likelihood[y,x] += 1
            self.id_occupy[y,x, 2] = n.id
        
    def rewire(self, node:Node):
        record_nodes = list(self.nodes_book.values())
        tgt = np.array([n.position for n in record_nodes])
        src2tgt_dist = find_distance(node.position, tgt)
        valid_nodes = [record_nodes[i] for i in 
                        np.where(src2tgt_dist.flatten() < self.MAX_DIST)[0]]
        
        for i, vn in enumerate(valid_nodes):
            if vn.parent is None: continue
            new_dist = np.linalg.norm(vn.position - node.position) + node.cost
            if vn.cost > new_dist and self.check_obstacle_free(vn, node):
                vn.parent.remove_child(vn.id)
                self.__connect_node(node, vn)
                
    def __connect_node(self, parent:Node, child:Node):
        child.set_parent(parent)
        parent.add_child(child)

    def get_route_node(self, REFINE=False) ->list:
        route_nodes = list()
        node = self.nodes_book[-1]
        while node.parent is not None:
            if REFINE:
                post_anchor_node = node.parent.parent if node.parent.id != 0 else None # Constrained for post anchor
                prior_anchor_node = route_nodes[0] if node.id != -1 else None # Constrained for prior anchor
                if post_anchor_node is not None and prior_anchor_node is None:
                    new_tmp_node = self.nodes_book[-1] # * Set as goal point
                elif post_anchor_node is not None and prior_anchor_node is not None:
                    new_tmp_node = Node(id=-3, position=(node.position + node.parent.position)/2)
                elif post_anchor_node is None and prior_anchor_node is not None:
                    new_tmp_node = self.nodes_book[0] # * Set as init point
                else:
                    raise "Need to check get_route_node."
                if self.check_obstacle_free(new_tmp_node, prior_anchor_node, IGNORE_NONE=True) and \
                    self.check_obstacle_free(new_tmp_node, post_anchor_node, IGNORE_NONE=True):    
                        # print(f"Prior: {prior_anchor_node}, Node:{node} , Parent:{node.parent} , Post:{post_anchor_node}")
                        if post_anchor_node is None: node = new_tmp_node; break # Boundary case: When new_tmp_node = init node
                        route_nodes.insert(0, new_tmp_node)
                        node = post_anchor_node if post_anchor_node is not None else new_tmp_node
                        continue
            route_nodes.insert(0, node)
            node = node.parent 
        
        assert node is not None, f"node: {node}; Error"
        route_nodes.insert(0, node)
        if REFINE:
            return np.array([nn.position for nn in route_nodes])
        else: 
            return route_nodes
    
    def rrt_algorithm(self, num_iter, rrt_mode, sample_mode):
        '''MODE =   1. "rrt"
                    2. "rrt_star" 
                    3. "rrt_star_refinement" 
        '''
        if rrt_mode == "rrt_star_refinement":
            self.update_refine_likelihood()
            
        tmp_id = 0 if len(self.nodes_book) == 0 else max(self.nodes_book.keys()) +1    
        num_iter = tmp_id + num_iter
        assert num_iter > tmp_id, "Error in rrt_algorithm"
        
        while tmp_id < num_iter:
            if rrt_mode != "rrt_star_refinement" and self.check_reach_end():
                break
            rand_yx     = self.random_sample(sample_mode=sample_mode)
            if self.check_occupied(rand_yx) : continue
            nearest_yx = self.nearest(rand_yx)
            step_yx     = self.steer(nearest_yx, rand_yx)
            nearest_node = self.nearest(step_yx, NODE=True)
            if self.check_obstacle_free(src=step_yx, tgt=nearest_node):
                node_new = Node(id=tmp_id, position=step_yx)
                self.__connect_node(nearest_node, node_new)
                if rrt_mode == "rrt_star" or rrt_mode=="rrt_star_refinement":
                    self.rewire(node_new)
                self.update_rrt_record(node_new)        
                tmp_id = 1 + tmp_id
                draw_sample(self.map_img, self.nodes_book.values())
    
    
    def main(self, init_pt:np.ndarray, goal_pt:np.ndarray, 
             rrt_mode:str, sample_mode:str,
             num_iter=60,  REFINE=False, map_img=None):
        # Rewrite Got some problem.
        goal_node = Node(id = -1, position=goal_pt)
        init_node = Node(id = 0, position=init_pt)
        assert self.check_obstacle_free_for_1pt(init_node) and self.check_obstacle_free_for_1pt(goal_node), \
            "Wrong selection for initial point or goal point."
        self.update_rrt_record([init_node, goal_node])
        self.map_img = map_img
        # Refine result
        self.rrt_algorithm( num_iter = self.C_space.shape[0] * self.C_space.shape[1],
                            rrt_mode=rrt_mode, sample_mode=sample_mode)
        if REFINE:
            self.rrt_algorithm( num_iter=num_iter,
                                    rrt_mode="rrt_star_refinement", sample_mode="centralized")
        