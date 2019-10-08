from FaceboxDetector.face_detector import FaceDetector
from Landmarking.mark_detector import MarkDetector
from .pose_estimator import PoseEstimator
from .stabilizer import Stabilizer

import numpy as np
import time
import cv2

class HeadPoseEstimator:
    def __init__(self, sample_frame):
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(50, 50),
                         maxLevel=4,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.face_detector = FaceDetector()

        # Introduce mark_detector to detect landmarks.
        self.mark_detector = MarkDetector()

        self.p_st = 17
        self.p_end = 56
        self.n_p = self.p_end-self.p_st

        # Introduce pose estimator to solve pose. Get one frame to setup the
        # estimator according to the image size.
        height, width = sample_frame.shape[:2]
        self.pose_estimator = PoseEstimator(p1=self.p_st, p2=self.p_end, img_size=(height, width))

        # Introduce scalar stabilizers for pose.
        self.pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=0.1,
            cov_measure=0.1) for _ in range(6)]

        self.fd, self.lm, self.hpe = 0, 0, 0
        self.cnt = 0
        self.fbStartCount = 0
        self.fCount = 0
        self.mCount = 0

        self.sumErr = np.tile(-1, self.n_p)
        self.errCap = 30
        self.old_facebox = []
        self.p0 = np.array(1)
        self.old_frame = sample_frame
        self.color = np.random.randint(0, 255, (68, 3))

    def get_direction(self, frame):
        self.fCount += 1
        # Read frame, crop it, flip it, suits your needs.
        #frame_got, frame = cam.read()
        #if frame_got is False:
        #    break
        clean_frame = frame.copy()

        # If frame comes from webcam, flip it so it looks like a mirror.
        #if video_src == 0:
        #frame = cv2.flip(frame, 2)
        '''print("original: frame")
        print(frame)
        print("original: old_frame")
        print(self.old_frame)
        print("original: old_facebox")
        print(self.old_facebox)'''
        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose
        mask = np.zeros_like(frame)
        if (self.sumErr[0] == -1 or self.checkErr(self.sumErr, self.errCap)):
            #print (cnt)
            #mask = np.zeros_like(frame)
            self.cnt = 0
            self.fbStartCount += 1
            # self.sumErr = np.tile(0, self.n_p)
            self.sumErr = self.sumErr.astype('float64')
            self.sumErr = np.reshape(self.sumErr, (len(self.sumErr), 1))

            t = time.clock()
            faceboxes = self.face_detector.get_faceboxes(frame)
            if (len(faceboxes) > 0):
                self.old_facebox = facebox = faceboxes[0:1]
            else:
                facebox = None
                self.sumErr[0] = -1
            self.fd += time.clock() - t

            if facebox is not None:
                t = time.clock()

                marks = self.mark_detector.detect_marks(frame, facebox)
                marks = marks[0][self.p_st:self.p_end]

                self.lm += time.clock() - t

                self.p0 = np.array(marks)
                self.p0 = self.p0.astype('float32')
                self.p0 = np.reshape(self.p0, (len(self.p0), 1, 2))

                # Uncomment following line to show raw marks.
                self.mark_detector.draw_marks(
                   frame, marks, color=(0, 255, 0))

                # Try pose estimation with 68 points.
                t = time.clock()
                pose = self.pose_estimator.solve_pose_by_68_points(marks)
                self.hpe += time.clock() - t

                self.draw_pose(frame, pose, self.pose_estimator, self.pose_stabilizers, True)

        else:
            self.cnt += 1
            t = time.clock()
            old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            '''print("old_gray")
            print(old_gray)
            print("frame_gray")
            print(frame_gray)
            print("p0")
            print(self.p0)
            print("lk_params")
            print(self.lk_params)'''
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, self.p0, None, **self.lk_params)
            self.sumErr += err

            # Select good points
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            '''print("good_new")
            print(good_new)
            print("good_old")
            print(good_old)'''
            offset = [0, 0]
            # draw the tracks
            npoints = 0
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                offset[0] += a - c
                offset[1] += b - d
                npoints += 1
                mask = cv2.line(mask, (a, b), (c, d), self.color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, self.color[i].tolist(), -1)
            offset[0] /= npoints
            offset[1] /= npoints
            '''print("offset0")
            print(offset[0])
            print("offset1")
            print(offset[1])'''
            self.old_facebox[0][0] += offset[0]
            self.old_facebox[0][1] += offset[1]
            self.old_facebox[0][2] += offset[0]
            self.old_facebox[0][3] += offset[1]
            facebox = self.old_facebox
            self.fd += time.clock() - t

            if (len(good_new) < self.n_p):
                self.mCount += 1
                t = time.clock()
                '''print("frame")
                print(frame)
                print("facebox")
                print(facebox)'''
                marks = self.mark_detector.detect_marks(frame, facebox)
                marks = marks[0][self.p_st:self.p_end]

                self.lm += time.clock() - t
                # p0 = np.array([marks[39], marks[42], marks[33], marks[48], marks[54]])#, marks[8]])
                self.p0 = np.array(marks)
                self.p0 = self.p0.astype('float32')
                self.p0 = np.reshape(self.p0, (len(self.p0), 1, 2))
            else:
                marks = good_new
            t = time.clock()
            pose = self.pose_estimator.solve_pose_by_68_points(marks)
            self.hpe += time.clock() - t

            self.draw_pose(frame, pose, self.pose_estimator, self.pose_stabilizers, True)

        self.old_frame = clean_frame
        # Show preview.
        #img = cv2.add(frame, mask)
        #cv2.imshow('frame', frame)
        return frame
        # cv2.imshow("Preview", frame)
        '''if cv2.waitKey(10) == 27:
            break'''

    def draw_pose(self, frame, pose, pose_estimator, pose_stabilizers = None, stable = False):
        if stable:
            # Stabilize the pose.
            stabile_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                stabile_pose.append(ps_stb.state[0])
            stabile_pose = np.reshape(stabile_pose, (-1, 3))

            pose_estimator.draw_annotation_box(
                frame, stabile_pose[0], stabile_pose[1], color=(128, 255, 128))
        else:
            pose_estimator.draw_annotation_box(
                frame, pose[0], pose[1], color=(255, 128, 128))


    def checkErr(self, array, cap):
        for x in array:
            if x[0] > cap:
                return True
        return False

'''
if __name__ == '__main__':
    #main()
    cam = cv2.VideoCapture(0)
    _, sample_frame = cam.read()

    f = HeadPoseEstimator(sample_frame)
    while True:
        frame_got, frame = cam.read()
        if frame_got is False:
            break
        frame = f.get_direction(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) == 27:
            break'''
