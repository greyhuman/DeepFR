import pyrealsense2 as rs
import numpy as np
import cv2
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

pipeline.start(config)
out_metadata = []

try:
    while(True):
        frames = pipeline.wait_for_frames()
        infrared_frame1 = frames.get_infrared_frame(1)
        infrared_frame2 = frames.get_infrared_frame(2)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if color_frame:
            out_metadata.append([frames.get_frame_metadata(rs.frame_metadata_value.backend_timestamp),
                    frames.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp),
                    frames.get_frame_metadata(rs.frame_metadata_value.frame_timestamp),
                    frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival),
                    frames.get_frame_metadata(rs.frame_metadata_value.frame_counter)])
            color_image = np.asanyarray(color_frame.get_data())
            infrared_image1 = np.asanyarray(infrared_frame1.get_data())
            infrared_image2 = np.asanyarray(infrared_frame2.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            cv2.imshow('img',depth_image)
            cv2.imshow('img0',color_image)

            cv2.imshow('img1',infrared_image1)
            cv2.imshow('img2',infrared_image2)
            cv2.waitKey(1)

except KeyboardInterrupt:
    pass
