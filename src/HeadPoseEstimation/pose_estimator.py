"""Estimate head pose according to the facial landmarks"""
import numpy as np
import math

import cv2


class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, p1 = 0, p2 = 68, img_size=(480, 640)):
        self.size = img_size

        # 3D model points.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ]) / 4.5

        self.model_points_68 = self._get_full_model_points()
        self.model_points_p12 = self.model_points_68[p1: p2]
        # self.model_points_68 = self.model_points_68[p1: p2]

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])
        # self.r_vec = None
        # self.t_vec = None

    def _get_full_model_points(self, filename='assets/model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        # model_points *= 4
        model_points[:, -1] *= -1

        return model_points

    def solve_pose(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeefs)

        # (success, rotation_vector, translation_vector) = cv2.solvePnP(
        #     self.model_points,
        #     image_points,
        #     self.camera_matrix,
        #     self.dist_coeefs,
        #     rvec=self.r_vec,
        #     tvec=self.t_vec,
        #     useExtrinsicGuess=True)
        return (rotation_vector, translation_vector)

    def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """


        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_p12, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_p12,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def solve_pose_by_5_points(self, image_points):
        """
        Solve pose from 5 mtcnn image points
        Return (rotation_vector, translation_vector) as pose.
        """

        # point1 = (self.model_points_68[37] + self.model_points_68[38] + self.model_points_68[39] +     # right eye
        #           self.model_points_68[40] + self.model_points_68[41] + self.model_points_68[42]) / 6  # center

        point1 = (self.model_points_68[38] + self.model_points_68[39] +     # right eye
                  self.model_points_68[41] + self.model_points_68[42]) / 4  # center

        # point2 = (self.model_points_68[43] + self.model_points_68[44] + self.model_points_68[45] +     # left eye
        #           self.model_points_68[46] + self.model_points_68[47] + self.model_points_68[48]) / 6  # center
        point2 = (self.model_points_68[44] + self.model_points_68[45] +     # left eye
                  self.model_points_68[47] + self.model_points_68[48]) / 4  # center
        # model_points_5 = np.array(list([point1, point2, self.model_points_68[34], self.model_points_68[49], self.model_points_68[55]]))
        # model_points_5 = np.array(list([self.model_points[3], self.model_points[2], self.model_points[0], self.model_points[5], self.model_points[4]]))
        model_points_5 = np.array(list(
            [point2, point1, self.model_points[0], self.model_points[5],
             self.model_points[4]]))

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                model_points_5, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points_5,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        pts = []
        rear_size = 75
        rear_depth = 0
        pts.append((0, 0, 0))
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        pts.append((0, 0, 100))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        pts = np.array(pts, dtype=np.float).reshape(-1, 3)
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        (pts2d, _) = cv2.projectPoints(pts,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        #print(pts2d)
        pts2d = np.int32(pts2d.reshape(-1, 2))

        diff_lr = pts2d[0][0] - pts2d[1][0]
        angle_lr = math.atan(diff_lr / 100)
        angle_lr = angle_lr * 180 / math.pi

        pts2d00str = str(pts2d[0][0])
        pts2d10str = str(pts2d[1][0])
        pts2dstr = pts2d00str + "," + pts2d10str
        cv2.putText(image, pts2dstr, (520, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255),
                    1)
        text_lr = str(angle_lr)
        col_lr = (0, 0, 255)

        '''if (angle_lr > 40):
            text_lr = "OUT"
            col_lr = (0, 0, 255)
        else:
            text_lr = "IN"
            col_lr = (0, 255, 0)'''

        diff_ud = pts2d[0][1] - pts2d[1][1]
        angle_ud = math.atan(diff_ud / 100)
        angle_ud = angle_ud * 180 / math.pi
        pts2d01str = str(pts2d[0][1])
        pts2d11str = str(pts2d[1][1])
        pts2d1str = pts2d01str + "," + pts2d11str
        cv2.putText(image, pts2d1str, (520, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0),
                    1)

        text_ud = str(angle_ud)
        col_ud = (0, 255, 0)

        '''if (angle_ud > 40):
            text_ud = "OUT"
            col_ud = (0, 0, 255)
        else:
            text_ud = "IN"
            col_ud = (0, 255, 0)'''

        cv2.putText(image, text_lr, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col_lr,
                    1)
        cv2.putText(image, text_ud, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col_ud,
                    1)

        #cv2.polylines(image, [pts2d], True, (255, 0, 0), line_width, cv2.LINE_AA)
        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)
        return self.solve_direction(image, angle_lr, angle_ud)

    def solve_direction(self, image, angle_lr, angle_ud):
        th_lr = 20
        th_ud = 20
        textdir = None
        if (angle_lr <= th_lr and angle_lr >= -th_lr and angle_ud >= 5):
            textdir = "UP"
        elif (angle_lr <= th_lr and angle_lr >= -th_lr and angle_ud <= -20):
            textdir = "DOWN"
        elif (angle_ud <= th_ud and angle_ud >= -th_ud and angle_lr >= 35):
            textdir = "RIGHT"
        elif (angle_ud <= th_ud and angle_ud >= -th_ud and angle_lr <= -35):
            textdir = "LEFT"
        elif (angle_ud <= th_ud and angle_ud >= -th_ud and angle_lr <= th_lr and angle_lr >= -th_lr):
            textdir = "STRAIGHT ON"
        else:
            textdir = "unknown"

        cv2.putText(image, textdir, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0),
                    1)
        return textdir

    def get_pose_marks(self, marks):
        """Get marks ready for pose estimation from 68 marks"""
        pose_marks = []
        pose_marks.append(marks[30])    # Nose tip
        pose_marks.append(marks[8])     # Chin
        pose_marks.append(marks[36])    # Left eye left corner
        pose_marks.append(marks[45])    # Right eye right corner
        pose_marks.append(marks[48])    # Left Mouth corner
        pose_marks.append(marks[54])    # Right mouth corner
        return pose_marks
