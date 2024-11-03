import cv2
import numpy as np

def fill_black_hole(img, ratio) -> np.ndarray:
    tmp_img = img.copy()
    row, col = img.shape[0], img.shape[1]
    cnt, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for i in range(hier.shape[1]):
        if hier[0,i, -1] == -1: continue
        area = cv2.contourArea(cnt[i])
        if area < row*col*ratio:
            cv2.drawContours(img, cnt, i, 255, thickness=cv2.FILLED )
    return img

def flood_fill_4_corners(img):
    row, col = img.shape
    tmp = img.copy()
    mask = np.zeros((row +2, col +2), dtype=np.uint8)
    cv2.floodFill(tmp, mask ,(0,0), 255)
    cv2.floodFill(tmp, mask ,(col-1, 0), 255)
    cv2.floodFill(tmp, mask ,(0, row-1), 255)
    cv2.floodFill(tmp, mask ,(col-1, row-1), 255)
    return tmp