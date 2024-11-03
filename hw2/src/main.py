import json
import pickle
import time
import argparse
from pathlib import Path

import cv2
import numpy as np 
import pandas as pd
import habitat_sim

from pcd_utils import (
    filter_ceiling_floor, 
    filter_color, 
    get_pcd_from_xyz_color, 
    nearest_neighbor
)
from img_utils import (
    fill_black_hole, 
    flood_fill_4_corners
)
from render_utils import (
    draw_circles, 
    draw_sample,
    show_image
)
from RRT_star import RRT_star, Node
from navigation import (
    make_cfg, 
    navigateAndSee, 
    get_agent_state, 
    get_ry_from_vec,
)


def _load_semantic_setting():
    df = pd.read_excel("color_coding_semantic_segmentation_classes.xlsx"  )
    SEMANTIC_COLOR = dict()
    for name, color_code in zip(df['Name'], df['Color_Code (R,G,B)']):
        color_code = [ float(c.strip())for c in color_code.strip('()').split(',')]
        assert len(color_code) == 3 , 'Wrong process method for color str.'
        SEMANTIC_COLOR[name] = np.array(color_code) 
    return SEMANTIC_COLOR

start_point = np.array([[0,0]])
SEMANTIC_COLOR = _load_semantic_setting()
C2S = {
        tuple(v.tolist()):k for k,v in SEMANTIC_COLOR.items()
    }

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        tmp_img = np.copy(map_img)
        cv2.circle(tmp_img, (x, y), 5, (0, 0, 255), 1)
        cv2.imshow('Map', tmp_img)
        start_point[0, :] =  [y,x]
        
def get_semantic_map_img(IMG_WIDTH = 600):
    global output_folder
    # Load pcd information
    pc = get_pcd_from_xyz_color(
        xyz=np.load("semantic_3d_pointcloud/point.npy") * 10000./255, # z-x,
        color=np.load("semantic_3d_pointcloud/color0255.npy")
    )
    # Filter point clouds.
    pc, _ = filter_color(pc, [SEMANTIC_COLOR['ceiling'], SEMANTIC_COLOR['floor']]) # remove ceiling & floor
    _ , wall_indx = filter_color(pc, [SEMANTIC_COLOR['wall'], SEMANTIC_COLOR['door']])
    pc, _, _ = filter_ceiling_floor(pc,f_indx=wall_indx, ratios=[0.0, 0.45])
    pc, _  = pc.remove_statistical_outlier(15, .8) # Remove outliers.
    
    # Convert 3D position to 2D image coordinate.
    pc_zx = np.array(pc.points)[:, [2,0]]
    yx_diff = np.max(pc_zx, axis=0) - np.min(pc_zx, axis=0)
    
    IMG_HEIGHT = int(IMG_WIDTH * yx_diff[0] / yx_diff[1])
    scale =  np.array([(IMG_HEIGHT-1) / yx_diff[0], (IMG_WIDTH -1) / yx_diff[1]])
    pc_img_yx = np.round( (pc_zx - np.min(pc_zx, axis=0)) * scale.reshape((-1, 2)) ) 
    
    pc_color = np.array(pc.colors) *255
    map_img = np.ones((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8) * 255
    LOW_PRIORITY = ['wall', 'wall-plug', 'wall-cabinet']
    used_xy = set()
    for r in range(pc_img_yx.shape[0]):
        row, col =  pc_img_yx[r, :].astype(np.int32).tolist()
        if (row,col) in used_xy:  continue
        else:   used_xy.add((row,col))
        specified_colors = pc_color [np.all(pc_img_yx== pc_img_yx[r], axis=1)]
        unique_colors = np.unique(specified_colors, axis=0)
        
        if len(unique_colors) == 1: 
            final_color = unique_colors[0,:]
        else:
            max_cnt = dict()
            final_name = ''
            for i in range(len(unique_colors)):
                name = C2S[ tuple(map(round, unique_colors[i].tolist()))]
                max_cnt[name] = np.sum( np.all(specified_colors == unique_colors[i], axis=1))
            sorted_items = sorted( max_cnt.items(), key= lambda x: x[1], reverse=True)
            for k,_ in sorted_items:
                if k not in LOW_PRIORITY:
                    final_name = k
                    break
            else:
                final_name = sorted_items[0][0]
            final_color = SEMANTIC_COLOR[final_name]
        map_img[row, col, :] = final_color
    
    min_pc_zx = np.min(pc_zx, axis=0)
    im2pc_info = np.concatenate((1/scale, min_pc_zx))
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    map_img = cv2.cvtColor(map_img.astype(np.uint8), cv2.COLOR_RGB2BGR)    
    return map_img, im2pc_info
    
def decide_stop_point(map_img,  target):
    map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
    row, col, _ = map_img.shape
    valid_r, valid_c = np.where(np.all(map_img == SEMANTIC_COLOR[target], axis=2))
    if len(valid_r) == 0 or len(valid_c) == 0:
        print(f"There isn't any points of {target} in the map. End of program.")
        exit()
    tgt_center = np.array([np.mean(valid_r), np.mean(valid_c)], dtype=np.int32).reshape(-1, 2)

    barrier = np.where(
        np.all(map_img==[255,255,255], axis=2), 
        np.zeros((row, col), dtype=np.uint8),
        np.ones((row, col), dtype=np.uint8)*255
    )
    # Preprocess to get the better barrier image.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, kernel, iterations=6)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_DILATE, kernel, iterations=1)
    barrier = fill_black_hole(barrier, 0.2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_DILATE, kernel, iterations=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    tmp_barrier = cv2.morphologyEx(barrier, cv2.MORPH_DILATE,kernel, iterations=6)
    tmp_barrier = flood_fill_4_corners(tmp_barrier)
    src = np.array(np.where(tmp_barrier == 0), dtype=np.int32).T
    _, indx = nearest_neighbor(tgt_center, src)
    
    stop_point = src[indx, :]
    return stop_point, tgt_center, barrier,

def path_planning(  c_space, 
                    start_pt, end_pt, 
                    rrt_mode, sample_mode,
                    REFINE=True,
                    STEP=50, MAX_DIST=120, 
                    map_img=None):
    global output_folder
    rrt = RRT_star( C_space=c_space, 
                    step=STEP,
                    max_search_dist=MAX_DIST,
                    )
    print("Path planning ...")
    st = time.time()
    rrt.main(   init_pt=np.array(start_pt).reshape((-1, 2)),
                goal_pt=np.array(end_pt).reshape((-1, 2)),
                rrt_mode=rrt_mode, sample_mode=sample_mode,
                REFINE=REFINE, map_img=map_img,
                )
    print(f"Finish! Cost time: {time.time() - st}")
    route = rrt.get_route_node(REFINE=REFINE)
    
    if not REFINE:
        route = np.array([ n.position for n in route])
    sample_img = draw_sample(map_img, rrt.nodes_book.values(), window="sample")
    cv2.imwrite(Path(output_folder)/Path(f"{rrt_mode}_{sample_mode}_samples.png"), sample_img)    
    return route

def navigate(route, im2pc_info, target, tgt_center):
    global test_scene, SEMANTIC_INFO, output_folder, args
    # Initialize agent
    sim_settings = {
        "scene": test_scene,  # Scene path
        "default_agent": 0,  # Index of the default agent
        "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "sensor_pitch": 0,   # sensor pitch (x rotation in rads)
        "action_move": 0.1, # meter
        "action_turn": 2,    # degree
    }
    
    tt = 1
    draw_circles(map_img, route, window="route", tt=tt)
    video_path = Path(output_folder) / Path(f"to_{target}_{args.rrt_mode}_{args.sample_mode}_moving{sim_settings['action_move']}_turning{sim_settings['action_turn']}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(video_path, fourcc, 30, (512, 512))
    def navigateAndRecord(*args, **kwargs):
        video.write(navigateAndSee(*args, **kwargs))
    
    loc_pc = np.concatenate((route.reshape(-1,2), tgt_center), axis=0) * im2pc_info[:2].reshape(-1,2) + im2pc_info[2:].reshape(-1,2) 
    vec_pc = np.concatenate( 
                ( np.array([[-1,0]]), np.diff(loc_pc[0:2, :], axis=0)),
                axis=0)
    ry = get_ry_from_vec(vec_pc[0], vec_pc[1])
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(sim_settings["default_agent"])
    # Set initial location & orientation.
    agent.set_state(get_agent_state(loc=[loc_pc[0,1], 0.0, loc_pc[0,0]],
                                    ex_rot=[ 0.0, ry, 0.0])
                    )
    # All location use (z, x) to represent a point.
    print("Navigating ...")
    loss = list()
    st =time.time()
    i = 1
    while i < len(loc_pc) -1:
        loc1 = agent.get_state().sensor_states['color_sensor'].position[[2, 0]]
        goal_loc = loc_pc[i, :]
        dist = np.linalg.norm(goal_loc - loc1)
        num_forward = int(np.round(dist / sim_settings['action_move']))
        for _ in range(num_forward):
            navigateAndRecord(sim, agent, 'move_forward', target, SEMANTIC_INFO, tt= tt)
        loc2 = agent.get_state().sensor_states['color_sensor'].position[[2, 0]]        
        v1 = loc2 - loc1
        next_loc = loc_pc[i+1, :] 
        v2 = next_loc - loc2
        ry = get_ry_from_vec(v1, v2, degrees=True) # In (z, x)
        print(f"{i}-Rotate:", ry)
        num_rotate = int(np.round(ry / sim_settings['action_turn']))
        for _ in range(abs(num_rotate)):
            action = 'turn_left' if ry > 0 else 'turn_right'
            navigateAndRecord(sim, agent, action, target, SEMANTIC_INFO,  tt=tt)                
        loc3 = agent.get_state().sensor_states['color_sensor'].position[[2,0]]
        
        loss.append(np.linalg.norm(loc3-goal_loc)**2)
        if np.linalg.norm(loc3- goal_loc) < 0.5:
            i = i + 1
    video.release()
    print(f"Finish! Cost time: {time.time() -st} MLSE: {np.mean(loss)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Robot navigation")
    parser.add_argument("--step", type=float, default=50.0)
    parser.add_argument("--max_dist", type=float, default=120.0)
    parser.add_argument("--construct_map", '-cm', action='store_true')
    parser.add_argument("--rrt_mode", type=str, help="rrt or rrt_star")
    parser.add_argument("--sample_mode", type=str, help="diverse, centralized, random, goal_bias")
    parser.add_argument("--refine", action="store_true", help="Refine the result or not.")
    parser.add_argument("--output_folder", type=str, default="./data")
    parser.add_argument("--test_scene", type=str, default="apartment_0/habitat/mesh_semantic.ply")
    parser.add_argument("--test_scene_json", type=str, default="apartment_0/habitat/info_semantic.json")
    args = parser.parse_args()
    # * 0. Set parameters.
    test_scene = args.test_scene
    test_scene_semantic  = args.test_scene_json
    STEP, MAX_DIST= args.step, args.max_dist
    output_folder = args.output_folder
    with open(test_scene_semantic, 'r') as file:
        SEMANTIC_INFO = json.load(file)
        
    # * 1. Create Semantic
    if not ((Path(output_folder)/ Path('map.png')).exists() and \
            (Path(output_folder)/ Path('im2pc_info.npy')).exists() )or \
            args.construct_map:
        print("Construct map ... ")
        st_time = time.time()
        map_img, im2pc_info = get_semantic_map_img()
        cv2.imwrite(Path(output_folder)/ Path("map.png"), map_img)
        np.save(Path(output_folder)/Path("im2pc_info.npy"), np.array(im2pc_info))
        print(f"Finish! Cost time: {time.time()-st_time}")
    else:
        map_img = cv2.imread(Path(output_folder) / Path("map.png"))
        im2pc_info = np.load(Path(output_folder) / Path("im2pc_info.npy"))
    # * 2. Trigger event to select the starting point & decide target.
    target = input("Input target: ")
    assert target in SEMANTIC_COLOR, f"Target {target} isn't in SEMANTIC_COLOR."
    cv2.imshow("Map", map_img)
    cv2.setMouseCallback("Map", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    assert len(start_point), "Select start points failed." 
    stop_point, tgt_center, barrier = decide_stop_point(map_img, target)
    # * 3. Path planning
    route = path_planning(  barrier, start_point, stop_point, 
                            rrt_mode=args.rrt_mode, sample_mode=args.sample_mode, 
                            REFINE=args.refine,
                            STEP=STEP, MAX_DIST=MAX_DIST, 
                            map_img=map_img)
    
    # cv2.imwrite(Path(output_folder) / Path("route.png"), draw_circles(map_img, route))
    # np.save(Path(output_folder) /Path("tgt_center.npy"), tgt_center)
    # np.save(Path(output_folder) /Path("route.npy"), route)
    # target = "refrigerator"
    # im2pc_info = np.load(Path(output_folder) / Path("im2pc_info.npy"))
    # route = np.load(Path(output_folder) / Path("route.npy"))
    # tgt_center = np.load(Path(output_folder) / Path("tgt_center.npy"))
    # * 4. Navigation
    navigate(route, im2pc_info, target, tgt_center)