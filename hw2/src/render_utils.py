import cv2 
import numpy as np 

def visualize_start_end_point(img:np.ndarray, start_pt:np.ndarray, end_pt:np.ndarray):
    tmp_img = img.copy()
    tmp_img = cv2.circle(tmp_img, start_pt[0, [1,0]], 8, color=[0,255,0], thickness=3)
    tmp_img = cv2.circle(tmp_img, end_pt[0, [1,0]], 5, color=[0,0,255], thickness=-1)
    cv2.imshow("Start & End points", tmp_img)
    cv2.waitKey()

def show_image(img:np.ndarray, window="img", normalize=False, tt=2):
    if normalize:
        img = img.astype(np.float64)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) *255
        img = img.astype(np.uint8)
    cv2.imshow(window, img.astype(np.uint8))
    cv2.waitKey(tt)
    

def draw_sample(img:np.ndarray, nodes:list, window="sample", indx=None, NODE_ONLY=False, tt=2):
    tmp_img = img.copy().astype(np.uint8)
    st_node = None
    for i, n in enumerate(nodes):
        if n.id == 0:
            st_node = n
            color = (0,255,0)
        elif n.id == -1:
            color = (0,0,255)
        elif indx is not None and i in indx:
            color = (0,255,255)
        else:
            color = (0,0,0)
        
        cur_yx = n.position
        cv2.circle(tmp_img, cur_yx[[1,0]], 6, color=color, thickness=3)
        if n.parent is None: continue
        if not NODE_ONLY:
            parent_yx = n.parent.position
            cv2.circle(tmp_img, parent_yx[[1,0]], 6, color=(0,0,0), thickness=3)
            cv2.line(tmp_img, cur_yx[[1,0]], parent_yx[[1,0]], color=(0,0,0), thickness=3)
    cv2.circle(tmp_img, st_node.position[[1,0]] , 6, (0, 255,0), thickness=3)
    show_image(tmp_img, window=window, tt=tt)
    return tmp_img

def draw_circles(img:np.ndarray, pts:np.ndarray, window="circles", indx=None, NODE_ONLY=False, tt=2):
    tmp_img = img.copy().astype(np.uint8)
    for i, yx in enumerate(pts):
        if indx is not None and i in indx:
            color = (0,255,255)
        elif i ==0:
            color = (0, 255, 0)
        elif i == len(pts):
            color = (0,0,255)
        else:
            color = (0,0,0)
        cv2.circle(tmp_img, yx[[1,0]], 6, color=color, thickness=3)
    show_image(tmp_img, window=window, tt=tt)
    return tmp_img