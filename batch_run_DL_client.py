# Based on the height of camera obtained from exoskeleton we remove ground from binary to see only resultant images.
# The terrain detection is based on Deep learning model trained using tensorflow

from __future__ import print_function
import pyrealsense2 as rs
import numpy as np
import cv2
import math
import argparse
from matplotlib import pyplot as plt
import torch
import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Dropout, UpSampling2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from datetime import datetime
from scipy.signal import argrelextrema

import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import csv
class Queue:
    def __init__(self):
        self.queue = []
        self.dist_queue = []
        self.height_queue = []

    def enqueue(self, x, d, h):
        self.dist_queue.insert(0, d)
        self.height_queue.insert(0, h)
        return self.queue.insert(0, x)

    def dequeue(self):
        self.dist_queue.pop()
        self.height_queue.pop()
        return self.queue.pop()

    def queue_size(self):
        return len(self.queue)

    def isEmpty(self):
        return len(self.queue) == 0

    def front(self):
        return self.queue[-1]

    def rear(self):
        return self.queue[0]

    def max_elem(self):
        return max(self.queue, key=self.queue.count)

    def avg_terrain_characteristics(self):
        s = max(self.queue, key=self.queue.count)
        indices = [i for i, x in enumerate(self.queue) if x == s]
        # print("Index ")
        dist_lst = [self.dist_queue[i] for i in indices]
        height_lst = [self.height_queue[i] for i in indices]
        # print("Lst ", dist_lst)
        # self.dist_queue[indices]
        return s, sum(dist_lst) / len(dist_lst), sum(height_lst) / len(height_lst)

    def print_elem(self):
        for i in range(0,len(self.queue)):
            print(self.queue[i])

# *******************************************************************************
# Deep learning model requirement and functions
def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def calc_net_size():
    sess = K.get_session()
    graph = sess.graph
    stats_graph(graph)

def step_decay_lr_rate_scheduler(epoch, lr):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 15 #30
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch)/epochs_drop))
    print("1) Current learning rate is ", lr)
    return lr

def lr_rate_scheduler(epoch, lr):
    if epoch <= 15:
        lr = 0.001  # 0.01
        print("1) Current learning rate is ", lr)
        return lr
    else:
        lr = 0.0001  # 0.001
        print("2) Current learning rate is ", lr)
        return lr

def compute_surface_histogram(d_im):
    d_im[d_im == 0] = 1
    zy, zx = np.gradient(d_im)
    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise
    # zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
    # zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

    direction = np.rad2deg(np.arctan2(zx, zy))
    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    # normal *= 255
    return normal, direction


def compute_normal_weightage(surface_nor_w, points_to_consider, reduced_edge_pts):
    # surface_nor_w -  weightage with surface normal (only consider max edge weightage)
    # surface_nor_w1 - additive or cumulative weightage
    # surface_nor_w seems to work better.
    surface_nor_w1 = np.zeros((100, 100, 1))
    for i in range(0, len(reduced_edge_pts)):
        r = points_to_consider[i, 1].astype(int)
        c = points_to_consider[i, 2].astype(int)
        # print(r, c, reduced_edge_pts[i], surface_nor_w[r, c, 0])
        surface_nor_w1[points_to_consider[i, 1].astype(int), points_to_consider[i, 2].astype(int), 0] = \
            surface_nor_w1[points_to_consider[i, 1].astype(int), points_to_consider[i, 2].astype(int), 0] + \
            reduced_edge_pts[i]

        if surface_nor_w[r, c, 0] < reduced_edge_pts[i]:
            surface_nor_w[r, c, 0] = reduced_edge_pts[i]

    # print(np.max(surface_nor_w1), np.min(surface_nor_w1), np.linalg.norm(surface_nor_w1))
    # print(np.max(surface_nor_w), np.min(surface_nor_w), np.linalg.norm(surface_nor_w))

    surface_nor_w1 = (surface_nor_w1 - np.min(surface_nor_w1)) / (np.max(surface_nor_w1) - np.min(surface_nor_w1))
    surface_nor_w = (surface_nor_w - np.min(surface_nor_w)) / (np.max(surface_nor_w) - np.min(surface_nor_w))

    # print(np.max(surface_nor_w1), np.min(surface_nor_w1))
    # print(np.max(surface_nor_w), np.min(surface_nor_w))
    #
    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(surface_nor_w)
    # axarr[1].imshow(surface_nor_w1)
    # # plt.imshow(surface_nor_w)
    # plt.show()
    return surface_nor_w, surface_nor_w1

# *******************************************************************************
def harris_corner(image):
    """
    To detect corners in an image - Corners can be used to differentiate actual obstacles, noise and ground
    """
    # convert the input image into
    # grayscale color space
    operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # For edge detection
    blurred = cv2.GaussianBlur(operatedImage, (5, 5), 0)
    wide = cv2.Canny(blurred, 10, 200)
    # modify the data type
    # setting to 32-bit floating point
    operatedImage = np.float32(operatedImage)
    print(operatedImage.shape)
    # apply the cv2.cornerHarris method
    # to detect the corners with appropriate
    # values as input parameters
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

    # Results are marked through the dilated corners
    dest = cv2.dilate(dest, None)
    # plt.imshow(dest)
    # plt.show()
    # Reverting back to the original image,
    # with optimal threshold value
    image[dest > 0.5 * dest.max()] = [0, 0, 255]
    # image[dest > 0.0001 * dest.max()] = [0, 0, 255]

    # y_corner_index, x_corner_index = np.where(dest > 0.01 * dest.max())
    # Create the basic mask
    a_mask = np.zeros(shape=image.shape[0:2])  # original
    print("Max dest", dest.max(), dest.shape, a_mask.shape) # dest.shape (720,1280)

    # Contours from edge detected image
    # contours, hierarchy = cv2.findContours(wide, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # a_mask[dest > 0.0001 * dest.max()] = 255
    a_mask[dest > 0.5 * dest.max()] = 255
    a_mask[47:53, :] = 0

    dest = (dest - np.min(dest)) / (np.max(dest) - np.min(dest))

    # r, c = np.where(a_mask == 255)
    # print(r, c)


    # f, axarr = plt.subplots(1, 3)
    # axarr[0].imshow(wide)
    # axarr[1].imshow(a_mask)
    # axarr[2].imshow(dest)
    # # plt.imshow(surface_nor_w)
    # plt.show()


    # plt.imshow(wide)
    # plt.show()
    # plt.imshow(a_mask)
    # plt.show()
    # plt.imshow(image)
    # plt.show()
    # plt.imshow(dest, cmap="jet")
    # plt.show()
    return dest
# *******************************************************************************

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Environment or terrain classification into 7 classes")
    parser.add_argument('-v', '--video', help="Video file save path", default=None)
    args = parser.parse_args()

    video = args.video
    if video:
        frame_width = 640 * 2
        frame_height = 480
        # out_vid = cv2.VideoWriter(video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
        out_vid = cv2.VideoWriter(video, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30, (frame_width, frame_height))

    terrain_classes = ['Level Ground', 'Asc_St', 'Des_St', 'Up Slope', 'Down Slope', 'Obstacle',
                       'Gap']

    detect_terrain = 0
    detect_queue = Queue()
    image_shape = (100, 100, 1)
    correct_terrain = 5


    frame_counter = 0
    correct_detect = 4
    hist_detect = np.zeros((7, 1))

    #Terrain characteristic estimation
    ground_row = 50
    # input image dimensions
    img_rows, img_cols = image_shape[0], image_shape[1]
    # *******************************************************************************
    # Deep learning model requirement and functions
    num_classes = 7
    input_shape = (img_rows, img_cols, 1)

    # deep CNN
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # ################################################
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    # ################################################
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=tf.keras.regularizers.L2(0.001), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(128, kernel_regularizer=tf.keras.regularizers.L2(0.001), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(64, kernel_regularizer=tf.keras.regularizers.L2(0.001), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))

    # model_path = 'checkpoint_newnetwork/best_model.h5'
    # model_path = 'checkpoint_red_data/best_model.h5'
    # model_path = 'checkpoint_add_data/best_model.h5'
    # model_path = 'Extra_data/Results_06062023/tf_models/checkpoint_250epochs/best_model.h5'
    # model_path = 'Extra_data/Results_18072023_iter1/ckpt_18072023/checkpoint_250epochs/best_model.h5'
    model_path = 'best_model.h5'
    checkpoint = ModelCheckpoint(model_path,
                                 verbose=1, monitor='val_acc',
                                 save_best_only=True, mode='auto')

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    # load the best model
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    calc_net_size()
    # *******************************************************************************
    # Setting up realsense camera, its pipeline and open3d pipeline for pointcloud
    # align = rs.align(rs.stream.depth)

    config = rs.config()
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 90)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)


    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)


    # get camera intrinsics
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    # print("Intrinsics ", intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    # Get camera intrinsics param
    width = intr.width
    height = intr.height
    # Create empty point cloud in grid format
    u = np.arange(width)
    v = np.arange(height)
    u_mat, v_mat = np.meshgrid(u, v)
    # [h,w,3]
    pcd_grid = np.zeros((height, width, 3))
    # kb = KBHit()
    save_img = False
    save_counter = 0
    save_path = os.path.join(os.getcwd(), 'Extra_data/6_center')

    first = True
    alpha = 0.02
    camera_height = 0.6 # 0.68  # 0.92  # To be provided by Exoskeleton
    # *******************************************************************************
    # For showing prediction probabilities in an auto-update way
    plt.ion()
    # *******************************************************************************
    while True:
        print("Depth scale: ", depth_scale, 1/depth_scale)
        detect_terrain = 0
        dt0 = datetime.now()
        #Get frames from the camera
        frames = pipeline.wait_for_frames()
        aligned_frames = frames

        depth_frame = aligned_frames.get_depth_frame()

        depth_frame = rs.decimation_filter(1).process(depth_frame)
        depth_frame = rs.disparity_transform(True).process(depth_frame)
        depth_frame = rs.temporal_filter().process(depth_frame)
        depth_frame = rs.disparity_transform(False).process(depth_frame)
        
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
       
        iter = 0
        # gather IMU data from camera
        for value in frames:
            if iter > 1:
                if iter == 2:
                    accel = value.as_motion_frame().get_motion_data()
                if iter == 3:
                    gyro = value.as_motion_frame().get_motion_data()
            iter = iter + 1

        ts = depth_frame.get_timestamp()
        # calculation for the first frame
        if (first):
            first = False
            last_ts_gyro = ts

            # accelerometer calculation
            accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
            accel_angle_x = math.degrees(math.atan2(accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)))
            accel_angle_y = math.degrees(math.pi)

            continue

        # calculation for the second frame onwards
        # gyrometer calculations
        dt_gyro = (ts - last_ts_gyro) / 1000
        last_ts_gyro = ts

        gyro_angle_x = gyro.x * dt_gyro
        gyro_angle_y = gyro.y * dt_gyro
        gyro_angle_z = gyro.z * dt_gyro

        dangleX = gyro_angle_x * 57.2958
        dangleY = gyro_angle_y * 57.2958
        dangleZ = gyro_angle_z * 57.2958

        totalgyroangleX = accel_angle_x + dangleX
        totalgyroangleY = accel_angle_y + dangleY
        totalgyroangleZ = accel_angle_z + dangleZ

        # accelerometer calculation
        accel_angle_z = math.degrees(math.atan2(accel.y, accel.z))
        accel_angle_x = math.degrees(math.atan2(accel.x, math.sqrt(accel.y * accel.y + accel.z * accel.z)))
        accel_angle_y = math.degrees(math.pi)

        # combining gyrometer and accelerometer angles
        combinedangleX = totalgyroangleX * alpha + accel_angle_x * (1 - alpha)
        combinedangleZ = totalgyroangleZ * alpha + accel_angle_z * (1 - alpha)
        combinedangleY = totalgyroangleY

        print("Gyro", gyro)
        print("Accel", accel)
        # REF_GROUND = [gyro.x, gyro.y, gyro.z]
        # Camera's inclination angle
        print("Angle -  X: " + str(round(combinedangleX, 2)) + "   Y: " + str(
            round(combinedangleY, 2)) + "   Z: " + str(round(combinedangleZ, 2)))
        
        depth_image = np.asanyarray(depth_frame.get_data())

        
        cv2.imshow('depth image', depth_image)
       
        print("SHAPE", depth_image.shape)
        dt1 = datetime.now()
        # From depth image, populate the point cloud grid created earlier
        pcd_grid[:, :, 0] = (u_mat + 0.5 - intr.ppx) / intr.fx * depth_image # Note: Need to add 0.5!
        pcd_grid[:, :, 1] = (v_mat + 0.5 - intr.ppy) / intr.fy * depth_image # Note: Need to add 0.5!
        pcd_grid[:, :, 2] = depth_image
        pcd_grid = pcd_grid * depth_scale

        
        # normal_edge = np.sqrt(np.square(gx) + np.square(gy))
        normal_edge, direction_edge = compute_surface_histogram(depth_image)
        gx = cv2.Sobel(normal_edge[:, :, 0], cv2.CV_64F, 1, 0, ksize=1)
        gy = cv2.Sobel(normal_edge[:, :, 1], cv2.CV_64F, 0, 1, ksize=1)
        input_edge = np.sqrt(np.square(gx) + np.square(gy))
        print("Max: ", np.max(input_edge), np.min(input_edge))
        # wide = cv2.Canny(input_edge, 10, 200)

        process_time = datetime.now() - dt1
        print("Time taken: ", process_time.total_seconds(), normal_edge.shape, direction_edge.shape)
        sd_pcd = pcd_grid.reshape(-1, 3)
        edge_sd_pcd = input_edge.reshape(-1)

        index = ~np.all(sd_pcd == 0, axis=1)
        sd_pcd = sd_pcd[index]
        edge_mag = edge_sd_pcd[index]

        num_samples = int(sd_pcd.shape[0]/30)
        # downsample the point cloud
        classify_pts = sd_pcd[np.linspace(0, sd_pcd.shape[0]-1, num_samples).astype(int), :]
        edge_classify_pts = edge_sd_pcd[np.linspace(0, sd_pcd.shape[0] - 1, num_samples).astype(int)]
        process_time = datetime.now() - dt1
        print("Time taken: ", process_time.total_seconds())
        print(classify_pts.shape, edge_classify_pts.shape)


        cv2.imshow('Normal x Stream', gx)  # normal_edge[:, :, 0]
        cv2.imshow('Normal y Stream', gy)  # normal_edge[:, :, 1]
        cv2.imshow('Normal z Stream', normal_edge[:, :, 2])
        cv2.imshow('Normal edge Stream', input_edge)  # normal_edge[:, :, 2])
        # plt.imshow(input_edge)
        # plt.show()

        # Process the downsampled cloud to create binary image and pass it to the terrain recognition model
        if classify_pts.shape[0] != 0:
            print("Frame Begin")
            # frame_counter = frame_counter + 1
            # ******************************************************************
            # Inclination angle of camera to rotate the pointcloud
            x_compensation_angle = round(combinedangleX, 2)
            compensation_angle = round(combinedangleZ, 2) + 90
            # Conversion of coordinate system to determine distance to obstacle
            # conv_mat = np.array([

            conv_mat = np.array([
                [np.cos(np.deg2rad(abs(x_compensation_angle))),
                 -np.sin(np.deg2rad(abs(x_compensation_angle))) * np.cos(np.deg2rad(abs(compensation_angle))),
                 np.sin(np.deg2rad(abs(x_compensation_angle))) * np.sin(np.deg2rad(abs(compensation_angle)))],
                [-np.sin(np.deg2rad(abs(x_compensation_angle))),
                 np.cos(np.deg2rad(abs(x_compensation_angle))) * np.cos(np.deg2rad(abs(compensation_angle))),
                 np.cos(np.deg2rad(abs(x_compensation_angle))) * np.sin(np.deg2rad(abs(compensation_angle)))],
                [0, -np.sin(np.deg2rad(abs(compensation_angle))), np.cos(np.deg2rad(abs(compensation_angle)))]
            ])
            points_to_classify = np.matmul(conv_mat, classify_pts.transpose())  # pts.transpose(), conv_mat
            points_to_classify = points_to_classify.transpose()
            
            print(points_to_classify)
            print("Width Limits: ", np.min(points_to_classify[:, 0]), np.max(points_to_classify[:, 0]))
            side_constraint = (points_to_classify[:, 0] >= -0.1) & (points_to_classify[:, 0] <= 0.1)
            print(points_to_classify[side_constraint])
            reduced_dim_points = points_to_classify[side_constraint]
            # ******************************************************************
            # Corresponding Edge
            reduced_edge_pts = edge_classify_pts[side_constraint]
            # # To constraint the pointcloud in the forward direction.
            # forward_constraint = (reduced_dim_points[:, 2] >= 0.2)
            # reduced_dim_points = reduced_dim_points[forward_constraint]

            if reduced_dim_points.shape[0] != 0:
                frame_counter+=1
                # Subtract the minimum y and z limits to make the range start from 0
                # reduced_dim_points[:, 1] = reduced_dim_points[:, 1] - np.min(reduced_dim_points[:, 1])
                min_distance = np.min(reduced_dim_points[:, 2])
                print("Height Check ", np.min(reduced_dim_points[:, 1]), np.max(reduced_dim_points[:, 1]))
                print("Distance Check ", np.min(reduced_dim_points[:, 2]), np.max(reduced_dim_points[:, 2]))
                reduced_dim_points[:, 1] = reduced_dim_points[:, 1] - camera_height
                # print("Height Check 1a ", np.min(reduced_dim_points[:, 1]), np.max(reduced_dim_points[:, 1]))

                min_h = abs(np.min(reduced_dim_points[:, 1]))
                max_h = abs(np.max(reduced_dim_points[:, 1]))
                if max_h > min_h:
                    ground_center = np.min(reduced_dim_points[:, 1])
                else:
                    ground_center = np.max(reduced_dim_points[:, 1])
                # ground_center = np.max(reduced_dim_points[:, 1])
                print("H check", max_h, min_h, ground_center)
                reduced_dim_points[:, 1] = reduced_dim_points[:, 1] - ground_center
                print("Height Check 2 ", np.min(reduced_dim_points[:, 1]), np.max(reduced_dim_points[:, 1]))
                reduced_dim_points[:, 2] = reduced_dim_points[:, 2] - min_distance
                # Only points which are 1m away are considered.
                distance_constraint = (reduced_dim_points[:, 2] < 1) # & (reduced_dim_points[:, 2] > 0.2)
                # points_to_consider = reduced_dim_points[distance_constraint]
                dist_points = reduced_dim_points[distance_constraint]
                # ******************************************************************
                # Corresponding Edge
                reduced_edge_pts = reduced_edge_pts[distance_constraint]

                height_constraint = (dist_points[:, 1] >= -0.5) & (dist_points[:, 1] <= 0.5)  # dist_points[:, 1] < 1
                points_to_consider = dist_points[height_constraint]
                points_to_show = dist_points[height_constraint]
                compute_characteristics = points_to_consider
                # ******************************************************************
                # Corresponding Edge
                reduced_edge_pts = reduced_edge_pts[height_constraint]
                # print(np.max(reduced_edge_pts), np.min(reduced_edge_pts), reduced_edge_pts.shape, points_to_consider.shape)

                if points_to_consider.shape[0] != 0:
                    
                
                    binary_im_arr = np.zeros((100, 100, 1), dtype=np.int64)
                    surface_nor_w = np.zeros((100, 100, 1))
                    mask = np.zeros(shape=(100, 100, 3))
                    corner_mask = np.zeros(shape=(100, 100, 3))
                    div_factor = 0.01
                    ########################################################################
                    print("Height Check 3 ", np.min(points_to_consider[:, 1]), np.max(points_to_consider[:, 1]))
                    points_to_consider[:, 1] = points_to_consider[:, 1] + 0.5
                    print("Height Check 4 ", np.min(points_to_consider[:, 1]), np.max(points_to_consider[:, 1]))
                    points_to_consider[:, 1] = (points_to_consider[:, 1] / div_factor)
                    points_to_consider[:, 2] = points_to_consider[:, 2] / div_factor
                    print("row check ", np.unique(points_to_consider[:, 1].astype(int)))
                    surface_nor_w, surface_nor_w1 = compute_normal_weightage(surface_nor_w, points_to_consider, reduced_edge_pts)
                    binary_im_arr[points_to_consider[:, 1].astype(int), points_to_consider[:, 2].astype(int), :] = 255
                    mask[points_to_consider[:, 1].astype(int), points_to_consider[:, 2].astype(int)] = (255, 255, 255)

                    # Noise removal from binary image
                    dst = cv2.fastNlMeansDenoising(binary_im_arr[:, :, 0].astype('uint8'), None, 45, 7, 21)
                    # print(dst.shape)
                    dst[np.where(dst > 130)] = 255
                    dst[np.where(dst <= 130)] = 0
         
                    ###############################################################################
                    ter_char_img = binary_im_arr.copy()
                    gr_char_img = binary_im_arr.copy() # Ground image for estimating characteristics when terrain is below ground
                    corner_dest = harris_corner(mask.astype('uint8'))

                    terrain_corners = surface_nor_w.reshape(mask.shape[0:2]) + surface_nor_w1.reshape(mask.shape[0:2]) + corner_dest
                    terrain_corners = (terrain_corners - np.min(terrain_corners)) / (np.max(terrain_corners) - np.min(terrain_corners))
                    a_mask = np.zeros(shape=mask.shape[0:2])  # original
                    a_mask[terrain_corners > 0.5 * terrain_corners.max()] = 255
                    # c_r, c_c = np.where(a_mask == 255)
                    # print("Corner detected: ", c_r, c_c)
                    # print(a_mask.shape)

                    corner_mask[:, :, 0] = a_mask
                    corner_mask[:, :, 1] = a_mask
                    corner_mask[:, :, 2] = a_mask

                  
                    # Pass the binary image to deep learning to determine the terrain
                    binary_im_arr = (binary_im_arr / 255).astype('float32')
                    prediction = model.predict(np.reshape(binary_im_arr, (1, 100, 100, 1)).astype('float32'))
                    print("Predicted", prediction, np.argmax(prediction))
                    detect_terrain = np.argmax(prediction)
                    plt.ylim([0, 1])
                    plt.plot(np.arange(num_classes), prediction.reshape(7,))

                    dst = (dst/255).astype('float32')
                    noiseless_prediction = model.predict(np.reshape(dst, (1, 100, 100, 1)).astype('float32'))
                    print("Noiseless Predicted", noiseless_prediction, np.argmax(noiseless_prediction))
                    ###############################################################################
                    print("Detected Terrain ", detect_terrain, terrain_classes[detect_terrain])
                    ter_min_dist = 0
                    ter_max_dist = 0
                    if save_img:
                        if detect_terrain != 5:
                            filename = 'real_data_img_18072023_' + str(frame_counter) + '.png'
                            cv2.imwrite(os.path.join(save_path, filename), ter_char_img)
                            save_counter = save_counter + 1

                 

                    ter_char_img[ground_row - 4:ground_row + 3, :, 0] = 0
                    r, c = np.where(ter_char_img[:, :, 0] == 255)
                    check_img = ter_char_img.copy()
                    check_img = check_img.astype('uint8')
                    check_img = check_img.reshape((img_cols, img_rows))
                    # Estimate terrain characteristics - width, height, angle and distance to terrain.
                    if len(r) != 0:
                        Terrain_H = (ground_row - np.min(r)) * div_factor
                        Terrain_W = (np.max(c) - np.min(c)) * div_factor
                        Terrain_Dist = min_distance + (np.min(c) * div_factor)
                        print("Obstacle Height: ", Terrain_H, np.min(r))
                        print("Rows and columns: ", r, c)
                        print("Obstacle Width: ", np.max(c), np.min(c), Terrain_W)
                        print("Obstacle Distance: ", Terrain_Dist)
                        iy = np.min(r)
                        ix = np.min(c[np.where(r == np.min(r))])

                        jy = np.max(r)
                        jx = np.min(c[np.where(r == np.max(r))])

                        Terrain_Angle = math.atan2((jy - iy), (jx - ix)) * 180 / math.pi
                        print("Angle: ", Terrain_Angle)

                    

                        if detect_terrain == 1 or detect_terrain == 2:
                            hist_r, hist_index = np.histogram(r, np.arange(np.min(r), np.max(r)))
                            if len(hist_r) != 0:
                                hist_r = hist_r/np.max(hist_r)

                                print(hist_r, hist_index)
                                # print(hist_index[np.where(hist_r > 0.7)])
                                # Finding local maxima
                                loc_max = argrelextrema(hist_r, np.greater_equal)
                                # Display Local maxima
                                print("Local Maxima:\n", hist_index[loc_max], "\n")

                     
                                if detect_terrain == 1:
                                    lm = loc_max[0]
                                else:
                                    lm = reversed(loc_max[0])
                                for iter in lm:
                                    row_iter = hist_index[iter]
                                    #print(iter)
                                    if hist_r[iter] >= 0.15:
                                        col_range = c[np.where(r == row_iter)]
                                        if len(col_range) != 0:
                                            print(col_range)
                                            print("Corner was Identified: ", np.max(col_range), np.min(col_range))
                                            Terrain_H = (ground_row - row_iter) * div_factor
                                            Terrain_W = (np.max(col_range) - np.min(col_range)) * div_factor
                                            Terrain_Dist = min_distance + (np.min(col_range) * div_factor)
                                            print("Stair height: ", Terrain_H)
                                            print("Stair Width: ", Terrain_W)
                                            print("Stair Distance: ", Terrain_Dist)
                                # print(np.histogram(c, np.arange(0, 100)))

                        if detect_terrain == 2 or detect_terrain == 6:
                            gr_char_img[0:ground_row - 5, :, 0] = 0
                            gr_char_img[ground_row + 4:img_rows-1, :, 0] = 0
                            r, c = np.where(gr_char_img[:, :, 0] == 255)
                            Terrain_Dist = min_distance
                            if len(c) != 0:
                                # hist_c, hist_index = np.histogram(c, np.arange(np.min(c), np.max(c)))
                                hist_c, hist_index = np.histogram(c, np.arange(0, img_cols-1))
                                hist_c = hist_c / np.max(hist_c)
                                print("Desc_stairs characteristics: ", hist_c, hist_index)
                                # if len(np.where(hist_c == 0)) !=0:
                                #     print(np.min(hist_index[np.where(hist_c == 0)]))
                                #     Terrain_Dist = Terrain_Dist + (np.min(hist_index[np.where(hist_c == 0)]) * div_factor)
                            print("TD: ", Terrain_Dist)

                        disp = " A: " + str(round(Terrain_Angle,2)) + " Dist: " + str(round(Terrain_Dist,2))+ " H: " + str(round(Terrain_H,2)) + " W: " + str(round(Terrain_W,2))

                    else:
                        print("No r found")
                       

                    if detect_queue.queue_size() == 10:
                        # print("Here 1")
                        detect_queue.dequeue()
                        detect_queue.enqueue(detect_terrain, ter_min_dist, ter_max_dist)
                    else:
                        # print("Here 2")
                        detect_queue.enqueue(detect_terrain, ter_min_dist, ter_max_dist)
                    # print("Queue", detect_queue.max_elem())
                    # detect_terrain = detect_queue.max_elem()
                    queue_terrain, ter_min_dist, ter_max_dist = detect_queue.avg_terrain_characteristics()
                    # disp = ' ' + str(ter_min_dist) + ' ' + str(ter_max_dist)
                    if save_img:
                        # cv2.imwrite(os.path.join(save_fill_path, filename), binary_im_arr)
                        with open('noise_terrain_char.csv', mode='a') as employee_file:
                            employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"',
                                                         quoting=csv.QUOTE_MINIMAL)

                            print(np.concatenate((detect_terrain.reshape(1,), queue_terrain.reshape(1,), prediction.reshape(7,), noiseless_prediction.reshape(7,))))
                            employee_writer.writerow(np.concatenate((detect_terrain.reshape(1,), queue_terrain.reshape(1,), prediction.reshape(7, ), noiseless_prediction.reshape(7,))))
                            # employee_writer.writerow(['Erica Meyers', 'IT', 'March'])
                        if save_counter > 150:
                            save_img = False
                            save_counter = 0

                    disp = ''
                    detect_terrain = queue_terrain
                    
                    if detect_terrain == 0:
                        disp = ''
                    if detect_terrain == correct_terrain:
                        correct_detect = correct_detect + 1
                    hist_detect[detect_terrain] = hist_detect[detect_terrain] + 1
                    print("Queue Detected Terrain ", detect_terrain, terrain_classes[detect_terrain])
                    # ******************************************************************
                    # cv2.putText(color_image, terrain_classes[detect_queue.max_elem()] + disp, org=(50, 50),
                    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=5)
                    cv2.putText(color_image, terrain_classes[detect_terrain] + disp, org=(50, 50),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=5)
                    cv2.imshow('Color Stream', color_image)
                    # plt.imshow(check_img)
                    # plt.show()
                    cv2.imshow('Binary image', mask)
                    plt.draw()
                    plt.pause(0.0001)
                    plt.clf()
                    if video:
                        mask = np.array(mask, dtype='uint8')
                        corner_mask = np.array(corner_mask, dtype='uint8')
                        resize_binary = cv2.resize(mask, (640, 480))
                        corner_binary = cv2.resize(corner_mask, (640, 480))
                        images = np.hstack((color_image, resize_binary))
                        # images = np.hstack((resize_binary, corner_binary))
                        out_vid.write(images)
                else:
                    points_to_show = np.zeros((100, 3))
            else:
                points_to_show = np.zeros((100, 3))
        else:
            points_to_show = np.zeros((100, 3))

        # disp = terrain_classes[detect_terrain] + str(line_angle)
        # disp = "Angle -  X: " + str(round(combinedangleX, 2)) + "   Y: " + str(
        #     round(combinedangleY, 2)) + terrain_classes[np.argmax(prediction)]
        print("Frame counter: ", frame_counter, " Correct Detection: ", correct_detect, hist_detect)
        # cv2.imshow('Color Stream', color_image1)

        process_time = datetime.now() - dt0
        print("FPS = {0}".format(int(1 / process_time.total_seconds())))

        key = cv2.waitKey(1)

        if key & 0xFF == ord('a'):
            plane_flag = 0

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()

            break
    accuracy=(correct_detect/frame_counter)*100
    print("Total frames processed: ",frame_counter)
    print("Correctly classified frames: ",correct_detect)
    print("Accuracy%: ",accuracy)
    pipeline.stop()
    if video is not None:
        out_vid.release()
