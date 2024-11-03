import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path,  FOV_x = np.pi/2, FOV_y = np.pi/2):
        """
            :param points: Selected pixels on top view(BEV) image
        """
        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape
        # Get Intrinsic matrix
        self.FOV_x = FOV_x
        self.FOV_y = FOV_y
        self.cx = (self.width-1) / 2
        self.cy = (self.height-1) / 2
        self.fx = self.width / (2 * np.tan( FOV_x /2))
        self.fy = self.height / (2 * np.tan( FOV_y /2))
        self.intrinsic = np.array([[self.fx,       0,    self.width/2],
                                   [    0.0, self.fy,   self.height/2],
                                   [    0.0,     0.0,               1]], dtype=np.float64)

        
    
    def bev_to_front(self, points, bev_pose, front_pose):
        """
            Project the bev view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """

        N = len(points)
        # Change to 2D homogeneous coordinate.
        bev_pixel_hom = np.hstack( (np.array(points), np.ones((N, 1)))).T
        bev_cam_coor = np.linalg.inv(self.intrinsic) @  (bev_pixel_hom * bev_pose['pos'][1])
        
        xy_inverse = np.array([[-1, 0, 0],
                                   [ 0,-1, 0],
                                   [ 0, 0, 1]])
        bev_world_coor = xy_inverse  @ bev_cam_coor
        # Change to 3D homogeneous coordinate.
        bev_world_hom = np.vstack((bev_world_coor, np.ones((1 , N)) ) )
        
        
        # Rotation matrix (pitch, yaw, roll)
        bev_pos, bev_ori = np.array(bev_pose['pos']), np.array(bev_pose['ori'])
        front_pos, front_ori = np.array(front_pose['pos']), np.array(front_pose['ori'])
        
        # TODO: Know how euler angle 3D rotation work 
        bev2front_pos = (bev_pos - front_pos).reshape((-1, 1))
        bev2front_ori =  -(bev_ori -front_ori)
        
        # Euler Angle definition. REMEMBER: the results are based on original coordinate which is the one before rotation.
        X_pitch = np.array([[1, 0, 0], 
                            [0, np.cos(bev2front_ori[0]), -np.sin(bev2front_ori[0])], 
                            [0, np.sin(bev2front_ori[0]), np.cos(bev2front_ori[0])]]  )
        Y_yaw = np.array([ [ np.cos(bev2front_ori[1]), 0, np.sin(bev2front_ori[1])],
                           [                        0, 1,                      0],
                           [-np.sin(bev2front_ori[1]), 0, np.cos(bev2front_ori[1])]])
        Z_roll = np.array([[ np.cos(bev2front_ori[2]), -np.sin(bev2front_ori[2]), 0], 
                           [ np.sin(bev2front_ori[2]),  np.cos(bev2front_ori[2]), 0],
                           [                      0,                       0, 1]])
        
        R = Z_roll @ Y_yaw @ X_pitch
        extrinsic = np.vstack(
                            (  np.hstack(
                                (R, bev2front_pos)
                                ), np.array([0,0,0,1]))
                        )
        # print(X_pitch)
        # print(Y_yaw)
        # print(Z_roll)
        # print(extrinsic)
        # print(extrinsic @ bev_world_hom)
        # exit()
        front_pixels = self.intrinsic @ np.hstack((xy_inverse, np.zeros((3,1)))) @ (extrinsic @ bev_world_hom)
        front_pixels /= front_pixels[-1, :]
        
        new_pixels = (front_pixels.T)[:,:2]
        new_pixels = new_pixels.astype(dtype=np.int32)
        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image

    
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--front',  type=str, default="bev_data/front1.png")
    parser.add_argument('--bev',  type=str, default="bev_data/bev1.png")
    args = parser.parse_args()
    top_rgb = args.bev
    front_rgb = args.front
    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    front_pose = {
        "pos": (0, 1, 0),
        "ori": (0, 0, 0)
    }
    bev_pose = {
        "pos": (        0, 2.5,  0),
        "ori": ( -np.pi/2,   0,  0)
    }
    projection = Projection(front_rgb)
    new_pixels = projection.bev_to_front(points, bev_pose=bev_pose, front_pose=front_pose)
    projection.show_image(new_pixels)
