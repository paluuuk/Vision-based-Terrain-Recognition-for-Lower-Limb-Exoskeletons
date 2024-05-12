import pyzed.sl as sl
import numpy as np
import cv2
import datetime
from matplotlib import pyplot as plt
import math
import argparse
import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from datetime import datetime
from scipy.signal import argrelextrema
# from Khbit import KBHit

import csv

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
    # sess = K.get_session()
    sess = tf.compat.v1.keras.backend.get_session()
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
# *******************************************************************************

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Environment or terrain classification into 7 classes")
    parser.add_argument('-v', '--video', help="Video file save path", default=None)
    args = parser.parse_args()

    video = args.video
    if video:
        frame_width = 640 * 2
        frame_height = 480
        fps = 30
        # out_vid = cv2.VideoWriter(video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
        out_vid = cv2.VideoWriter(video, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30, (frame_width, frame_height))

    terrain_classes = ['Level Ground', 'Asc_St', 'Des_St', 'Up Slope', 'Down Slope', 'Obstacle',
                       'Gap']

    detect_terrain = 0
    detect_queue = Queue()
    image_shape = (100, 100, 1)
    correct_terrain = 5 #terrain testing for
    frame_counter = 0
    correct_detect = 0
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

    model_path = './best_model.h5'
    checkpoint = ModelCheckpoint(model_path,
                                 verbose=1, monitor='val_acc',
                                 save_best_only=True, mode='auto')

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    # load the best model
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    calc_net_size()
    
    camera_height = 0.7
    # Initialize the ZED camera
    init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                                    coordinate_units=sl.UNIT.METER,
                                    coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
                                    depth_mode=sl.DEPTH_MODE.NEURAL)
                                    
    init_params.depth_stabilization = True
    zed = sl.Camera()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print("ZED Camera initialization failed. Exit program.")
        exit(1)
    
    tracking_params = sl.PositionalTrackingParameters() #set parameters for Positional Tracking
    tracking_params.enable_imu_fusion = True 
    status = zed.enable_positional_tracking(tracking_params) #enable Positional Tracking
    
    if status != sl.ERROR_CODE.SUCCESS:
        print("Enable Positional Tracking : "+repr(status)+". Exit program.")
        zed.close()
        exit()

    camera_pose = sl.Pose()
    camera_info = zed.get_camera_information()
    py_translation = sl.Translation()
    pose_data = sl.Transform()

    # get camera intrinsics
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    intr = calibration_params.left_cam

    width = camera_info.camera_configuration.resolution.width
    height = camera_info.camera_configuration.resolution.height

    # Create empty point cloud in grid format
    u = np.arange(width)
    v = np.arange(height)
    u_mat, v_mat = np.meshgrid(u, v)
    # [h,w,3]
    pcd_grid = np.zeros((height, width, 3))
    # Initialize ZED IMU variables
    zed_imu = sl.IMUData()

    # *******************************************************************************
    # runtime = sl.RuntimeParameters()
    runtime_params = sl.RuntimeParameters(confidence_threshold=50, texture_confidence_threshold=70)
    runtime_params.enable_fill_mode = True
    aligned_image = sl.Mat
    plt.ion()
    # *******************************************************************************
    imu_only = True
    if imu_only:
        sensors_data = sl.SensorsData()
    while True:

        detect_terrain = 0
        frame_counter += 1

        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            depth_frame = sl.Mat()
            display_depth_frame = sl.Mat()
            zed.retrieve_measure(depth_frame, sl.MEASURE.DEPTH)
            zed.retrieve_image(display_depth_frame, sl.VIEW.DEPTH)
            depth_array = depth_frame.get_data()
            display_depth_array = display_depth_frame.get_data()

            color_image = sl.Mat()
            zed.retrieve_image(color_image, sl.VIEW.LEFT)
            color_image_ocv = color_image.get_data()
            color_image_ocv = cv2.cvtColor(color_image_ocv, cv2.COLOR_RGB2BGR)

            # zed.retrieve_image(aligned_image, sl.VIEW.LEFT)
            # zed.retrieve_measure(depth_frame, sl.MEASURE.DEPTH)
            # zed.retrieve_image(color_image, sl.VIEW.LEFT)

            # For positional tracking info:
            tracking_state = zed.get_position(camera_pose,sl.REFERENCE_FRAME.WORLD) #Get the position of the camera in a fixed reference frame (the World Frame)
            text_rotation = 'No info'
            if imu_only :
                if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
                    rotation = sensors_data.get_imu_data().get_pose().get_euler_angles()
                    text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
            else : 
                if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    #Get rotation and translation and displays it
                    rotation = camera_pose.get_rotation_vector()
                    translation = camera_pose.get_translation(py_translation)
                    text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                    text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))
                    pose_data = camera_pose.pose_data(sl.Transform())
                    
            print("Text rotation: ", text_rotation)
            camera_info = zed.get_camera_information()
            calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters

            # For the left camera
            left_cam_intrinsics = calibration_params.left_cam
            fx = left_cam_intrinsics.fx
            fy = left_cam_intrinsics.fy
            cx = left_cam_intrinsics.cx
            cy = left_cam_intrinsics.cy
            
            print("parameters: ", fx, fy, cx, cy)
            print(np.min(depth_array), np.max(depth_array))
            # From depth image, populate the point cloud grid created earlier
            pcd_grid[:, :, 0] = (u_mat - cx) / fx * depth_array
            pcd_grid[:, :, 1] = (v_mat - cy) / fy * depth_array
            pcd_grid[:, :, 2] = depth_array

            sd_pcd = pcd_grid.reshape(-1, 3)
            index = ~np.all(sd_pcd == 0, axis=1)
            sd_pcd = sd_pcd[index]

            num_samples = int(sd_pcd.shape[0]/30)
            # downsample the point cloud
            classify_pts = sd_pcd[np.linspace(0, sd_pcd.shape[0]-1, num_samples).astype(int), :]

            # cv2.imshow('Color Stream', color_image_ocv)
            ter_min_dist = 0
            ter_max_dist = 0

            if text_rotation != 'No info':
                if classify_pts.shape[0] != 0:
                    print("Frame Begin")
                    # frame_counter = frame_counter + 1
                    # ******************************************************************
                    # Inclination angle of camera to rotate the pointcloud
                    x_compensation_angle = np.rad2deg(round(rotation[2], 2))
                    compensation_angle = np.rad2deg(round(rotation[0], 2)) #round(combinedangleZ, 2) + 90
                    output_pts = classify_pts.transpose()
                    output_pts[~np.isfinite(output_pts)] = 1.0
                    
                    # output_pts = np.nan_to_num(classify_pts.transpose(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    print("OP Limits 0: ", np.min(output_pts[:, 0]), np.max(output_pts[:, 0]))
                    print("OP Limits 1: ", np.min(output_pts[:, 1]), np.max(output_pts[:, 1]))
                    print("OP Limits 2: ", np.min(output_pts[:, 2]), np.max(output_pts[:, 2]))

                    conv_mat = np.array([
                        [np.cos(np.deg2rad(abs(x_compensation_angle))),
                        -np.sin(np.deg2rad(abs(x_compensation_angle))) * np.cos(np.deg2rad(abs(compensation_angle))),
                        np.sin(np.deg2rad(abs(x_compensation_angle))) * np.sin(np.deg2rad(abs(compensation_angle)))],
                        [-np.sin(np.deg2rad(abs(x_compensation_angle))),
                        np.cos(np.deg2rad(abs(x_compensation_angle))) * np.cos(np.deg2rad(abs(compensation_angle))),
                        np.cos(np.deg2rad(abs(x_compensation_angle))) * np.sin(np.deg2rad(abs(compensation_angle)))],
                        [0, -np.sin(np.deg2rad(abs(compensation_angle))), np.cos(np.deg2rad(abs(compensation_angle)))]
                    ])
                    points_to_classify = np.matmul(conv_mat, output_pts)  # pts.transpose(), conv_mat
                    points_to_classify = points_to_classify.transpose()

                    print("PC Limits 0: ", np.min(points_to_classify[:, 0]), np.max(points_to_classify[:, 0]))
                    print("PC Limits 1: ", np.min(points_to_classify[:, 1]), np.max(points_to_classify[:, 1]))
                    print("PC Limits 2: ", np.min(points_to_classify[:, 2]), np.max(points_to_classify[:, 2]))
                    print(points_to_classify)
                    print("Width Limits: ", np.min(points_to_classify[:, 0]), np.max(points_to_classify[:, 0]))
                    side_constraint = (points_to_classify[:, 0] >= -0.1) & (points_to_classify[:, 0] <= 0.1)
                    print(points_to_classify[side_constraint])
                    reduced_dim_points = points_to_classify[side_constraint]
                    # ******************************************************************
                    if reduced_dim_points.shape[0] != 0:
                        # Subtract the minimum y and z limits to make the range start from 0
                        # reduced_dim_points[:, 1] = reduced_dim_points[:, 1] - np.min(reduced_dim_points[:, 1])
                        min_distance = np.min(reduced_dim_points[:, 2])
                        print("Height Check ", np.min(reduced_dim_points[:, 1]), np.max(reduced_dim_points[:, 1]))
                        print("Distance Check ", np.min(reduced_dim_points[:, 2]), np.max(reduced_dim_points[:, 2]))
                        reduced_dim_points[:, 1] = reduced_dim_points[:, 1] - camera_height

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
                        height_constraint = (dist_points[:, 1] >= -0.5) & (dist_points[:, 1] <= 0.5)  # dist_points[:, 1] < 1
                        points_to_consider = dist_points[height_constraint]
                        points_to_show = dist_points[height_constraint]
                        compute_characteristics = points_to_consider
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
                            binary_im_arr[points_to_consider[:, 1].astype(int), points_to_consider[:, 2].astype(int), :] = 255
                            mask[points_to_consider[:, 1].astype(int), points_to_consider[:, 2].astype(int)] = (255, 255, 255)

                            binary_im_arr = (binary_im_arr / 255).astype('float32')
                            prediction = model.predict(np.reshape(binary_im_arr, (1, 100, 100, 1)).astype('float32'))
                            print("Predicted", prediction, np.argmax(prediction))
                            detect_terrain = np.argmax(prediction)



                            if detect_queue.queue_size() == 10:
                                
                                # print("Here 1")
                                detect_queue.dequeue()
                                detect_queue.enqueue(detect_terrain, ter_min_dist, ter_max_dist)
                            else:
                                # print("Here 2")
                                detect_queue.enqueue(detect_terrain, ter_min_dist, ter_max_dist)

                            queue_terrain, ter_min_dist, ter_max_dist = detect_queue.avg_terrain_characteristics()
                            detect_terrain = queue_terrain

                            if detect_terrain == correct_terrain:
                                correct_detect+=1

                            cv2.putText(color_image_ocv, terrain_classes[detect_terrain], org=(50, 50),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=5)
                            cv2.imshow('Color Stream', color_image_ocv)

                            cv2.imshow('Binary image', mask)
                            if video:

                                mask = np.array(mask, dtype='uint8')
                                corner_mask = np.array(corner_mask, dtype='uint8')
                                corner_binary = cv2.resize(corner_mask, (640, 480))
                                resize_binary = cv2.resize(mask, (640, 480))
                                color_out = cv2.resize(color_image_ocv, (640, 480))
                                combined_frames = np.hstack((color_out, resize_binary))
                                out_vid.write(combined_frames)

                            
                            # if video:
                            #     # Normalize and color-map the depth data
                            #     # normalized_depth = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                            #     # normalized_depth = np.uint8(normalized_depth)
                            #     # depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
                            #     # depth_colormap_resized = cv2.resize(depth_colormap, (640, 480))
                            #     depth_colormap_resized = cv2.resize(display_depth_array[:,:,0:3], (640, 480))
                                
                                
                            #     resize_binary = cv2.resize(mask, (640, 480))
                                
                            #     # Prepare the color stream
                            #     color_out = cv2.resize(color_image_ocv, (640, 480))
                            #     print("size: ", depth_colormap_resized.shape, resize_binary.shape, color_out.shape)
                            #     # Combine all frames horizontally
                            #     combined_frames = np.hstack((color_out, resize_binary))
                            #     final_frames = np.hstack((combined_frames, depth_colormap_resized))
                                
                            #     # Write the combined frame to the video
                            #     out_vid.write(final_frames)


            # cv2.imshow('Depth image', depth_colormap_resized)
            plt.imshow(display_depth_array, cmap="plasma")
            # print("Total frames processed: ",frame_counter)
            # print("Correctly classified frames: ",correct_detect)
            
            key = cv2.waitKey(1)

            if key & 0xFF == ord('a'):
                plane_flag = 0

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()

                break
            plt.draw()
            plt.pause(0.0001)
            plt.clf()
    # Disable positional tracking and close the camera
    accuracy=(correct_detect/frame_counter)*100
    print("Total frames processed: ",frame_counter)
    print("Correctly classified frames: ",correct_detect)
    print("Accuracy%: ",accuracy)
    zed.disable_positional_tracking()
    zed.close()
    if video is not None:
        out_vid.release()