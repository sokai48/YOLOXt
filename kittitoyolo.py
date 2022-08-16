from re import I
from matplotlib import projections
import numpy as np
import os
import cv2
import kitti_util as utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_point_cloud(ax, points, axes=[0, 1, 2], point_size=0.1, xlim3d=None, ylim3d=None, zlim3d=None):
    """
    Convenient method for drawing various point cloud projections as a part of frame statistics.
    """
    axes_limits = [
        [-20, 80],
        [-20, 20],
        [-3,3]

    ]

    axes_str = ['X','Y','Z']
    ax.grid(False)


    ax.scatter(*np.transpose(points[:, axes]), s=point_size, c=points[:, 3], cmap='gray')
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))                
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])

    # User specified limits
    if xlim3d!=None:
        ax.set_xlim3d(xlim3d)
    if ylim3d!=None:
        ax.set_ylim3d(ylim3d)
    if zlim3d!=None:
        ax.set_zlim3d(zlim3d)
        


def draw_box (ax, vertices, axes=[0, 1, 2], color='black') :

    axes_str = ['X','Y','Z']


    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2,3], [3,0],
        [4, 5], [5, 6], [6,7], [7,4],
        [0, 4], [1, 5], [2,6], [3,7]
    ]

    for connection in connections :
        ax.plot(*vertices[:,connection], c=color, lw=0.5)


def compute_box_3d(h, w, l, x, y, z, yaw):


    R = np.array([[np.cos(yaw), 0, np.sin(yaw)],[0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_cam2 += np.vstack([x,y,z])
    return corners_3d_cam2



def distance_point_to_segment(P,A,B) :

    '''
    計算最短距離從P點到AB
    return Q 代表AB上距離最短的點
    '''

    AP = P-A
    BP = P-B
    AB = B-A
    if np.dot(AB, AP)>= 0 and np.dot(-AB,BP) >=0 :  #代表是銳角三角形
        return np.abs(np.cross(AP,AB))/np.linalg.norm(AB), np.dot(AP,AB)/np.dot(AB,AB)*AB + A

    #如果不是以上 就代表P的投影點 在AB之外
    d_PA = np.linalg.norm(AP)
    d_PB = np.linalg.norm(BP)
    if d_PA < d_PB :
        return d_PA, A
    return d_PB, B


def min_distance_cuboids(cub1, cub2) :

    minD = 100000

    for i in range(4) :
        for j in range(4):
            d, Q = distance_point_to_segment(cub1[i, :2], cub2[j, :2], cub2[j+1, :2])
            if d < minD :
                minD = d
                minP = cub1[i, :2]
                minQ = Q

    for i in range(4) : 
        for j in range(4):
            d, Q = distance_point_to_segment(cub2[i, :2], cub1[j, :2], cub1[j+1, :2])
            if d < minD :
                minD = d
                minP = cub2[i, :2]
                minQ = Q
        
    return minP, minQ, minD



class kitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split="training", args=None):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split = split
        # print(root_dir, split)
        # self.split_dir = os.path.join(root_dir, split)
        self.split_dir = os.path.join(root_dir)

        if split == "training":
            self.num_samples = 7481
        elif split == "testing":
            self.num_samples = 0
        else:
            print("Unknown split: %s" % (split))
            exit(-1)

        lidar_dir = "velodyne"
        depth_dir = "depth"
        pred_dir = "pred"
        if args is not None:
            lidar_dir = args.lidar
            depth_dir = args.depthdir
            pred_dir = args.preddir

        self.image_dir = os.path.join(self.split_dir, "image_2")
        self.label_dir = os.path.join(self.split_dir, "label_2")
        self.calib_dir = os.path.join(self.split_dir, "calib")

        self.depthpc_dir = os.path.join(self.split_dir, "depth_pc")
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.depth_dir = os.path.join(self.split_dir, depth_dir)
        self.pred_dir = os.path.join(self.split_dir, pred_dir)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.image_dir, "%06d.png" % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        assert idx < self.num_samples
        lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % (idx))
        print(lidar_filename)
        return utils.load_velo_scan(lidar_filename, dtype, n_vec)
    def get_label_objects(self, idx):
        # assert idx < self.num_samples and self.split == "training"
        # print("%06d.txt" % (idx))
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        # assert 1==0, f"{utils.read_label(label_filename)}"
        return utils.read_label(label_filename)

    def convert_to_yolo(self, size, box) : 

        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh


        return (x,y,w,h)

    def get_calibration(self, idx):
        assert idx < self.num_samples
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % (idx))
        return utils.Calibration(calib_filename)

if __name__ == "__main__":
    # import mayavi.mlab as mlab
    # from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    #抓出原始圖片大小
    img = cv2.imread("/home/lab602.10977014_0n1/.pipeline2/10977014/YOLOXt/datasets/004076.png")
    oh,ow = img.shape[0], img.shape[1]


    #讀入原始label
    root_dir = "/home/lab602.10977014_0n1/.pipeline2/10977014/YOLOXt/YOLO2COCO/dataset/kitti/stamp/original/kitti_train/"
    output_dir = "/home/lab602.10977014_0n1/.pipeline2/10977014/YOLOXt/YOLO2COCO/dataset/kitti/stamp/original/kitti_train/newlabels/"
    img_dir = "/home/lab602.10977014_0n1/.pipeline2/10977014/YOLOXt/YOLO2COCO/dataset/kitti/stamp/original/kitti_train/image_2/"
    dataset = kitti_object(root_dir)
    # calib = utils.Calibration('/home/lab602.10977014_0n1/.pipeline2/10977014/YOLOXt/YOLO2COCO/dataset/kitti/stamp/original/test/calib', from_vide=True)
    # print(len(dataset))

    # for data_idx in range(len(dataset)):
    for f in os.listdir(root_dir+"label_2") :
        data_idx = int(f.split(".", 1)[0])

        # print(len(dataset))
        # print(data_idx)   
        objects = dataset.get_label_objects(data_idx)

        calibs = dataset.get_calibration(data_idx)
        # points = dataset.get_lidar(data_idx)
        # img2 = dataset.get_image(data_idx)
        EGOCAR = np.array([[2.15, 0.9, -1.73], [2.15, -0.9, -1.73],[-1.95, -0.9, -1.73],[-1.95, 0.9, -1.73],
                            [2.15, 0.9, -0.23], [2.15, -0.9, -0.23], [-1.95, -0.9, -0.23], [-1.95, 0.9, -0.23]])

        list = []

        #轉換lable_3d_to_2d

        path = output_dir + ("%06d.txt" % (data_idx))
        f = open(path,'w')

        for obj in objects:

            # if obj.type == "Car" or obj.type == "Van" or obj.type =="Pedestrian" or obj.type =="Person_sitting" or obj.type =="Cyclist" :
            if obj.type == "Car" or obj.type == "Van" or obj.type =="Truck" or obj.type =="Tram" :
                if obj.type == "Car"  : type = 0
                elif obj.type == "Van" : type = 1
                elif obj.type == "Truck" : type = 2
                elif obj.type == "Tram" : type = 3
                # elif obj.type == "Pedestrian" or obj.type =="Person_sitting" : type = 4
                # elif obj.type == "Cyclist" : type = 2

                # print(type)
                box = (obj.xmin,obj.xmax,obj.ymin,obj.ymax)
                x,y,w,h = dataset.convert_to_yolo( (ow,oh), box )

                # self.h = data[8]  # box height
                # self.w = data[9]  # box width
                # self.l = data[10]  # box length (in meters)
                # self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
                # self.ry 

                box_3D = np.array([obj.h, obj.w, obj.l, obj.t[0],obj.t[1],obj.t[2], obj.ry])

                corners_3d_cams =compute_box_3d(*box_3D)
                corners_3d_velo = calibs.project_rect_to_velo(corners_3d_cams.T) # 8x3  -> 為了畫出來3*8
                print("--------------{}---------------------".format(data_idx))
                print("--------------{}------------------".format(obj.type))
                print("which car : {}".format(box))
                print("----------------------------------")
                print(corners_3d_velo.shape)
                print(EGOCAR.shape)

                
                minPQD = min_distance_cuboids(EGOCAR, corners_3d_velo)

                print("New Z : {}".format(minPQD))

                newz = round(minPQD[2], 2)
                print("OLD Z : {}".format(obj.t[2]))



                '''
                
                #畫3D圖必備的
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(111, projection ='3d')
                
                ax.view_init(40,150)
                # draw_point_cloud(ax, points )
                # draw_box(ax, (corners_3d_velo))

                # draw_box(ax, (corners_3d_cams))
                fig,ax = plt.subplots(figsize=(20,10))
                draw_point_cloud(ax, points, axes=[0, 1] )
                draw_box(ax, (corners_3d_velo), axes=[0, 1] )
                # plt.show()
                plt.savefig("kittidraw/{}.png".format("%05d_pointvelo2D" % data_idx+str(box[0])))

                box3d_pts_2d, _ = utils.compute_box_3d(obj, calibs.P)

                img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))

                cv2.imwrite("kittidraw/{}.jpg".format("%05d_image" % data_idx+str(box[0])),img2)

                '''




                # z = obj.t[2]

                z = newz #替換成最鄰近距離

                column = [str(type),' ',str(x),' ',str(y),' ',str(w),' ',str(h),' ',str(z),'\n']


                list.append(column)
                # print(column)
                f.writelines(column)
                # print(list)

        # with open ('N_a.txt','w') as q:
        #     for i in a:
        #         for e in range(len(a[0])):
        #             t=t+str(i[e])+' '
        #         q.write(t.strip(' '))
        #         q.write('\n')
        #         t=''

        f.close
        #輸出成需要的txt




    #清掉空的label.txt
    for f in os.listdir(output_dir) :
        data_idx = int(f.split(".", 1)[0])
        path = output_dir + ("%06d.txt" % (data_idx))
        img_path = img_dir + ("%06d.png" % (data_idx))
        if os.path.getsize(path) == 0 :
            print("空的 : " + path)
            os.remove(path) 
            if os.path.exists(img_path) :
                print(img_path)
                os.remove(img_path)

