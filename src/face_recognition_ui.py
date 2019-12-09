import cv2
import numpy as np
import sys
import os
import math
import time
from threading import Thread
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal
from HeadPoseEstimation.head_pose_estimation import HeadPoseEstimator
import faceRecognition

APP_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

class Memorize(QtCore.QObject):
    state_change_signal = QtCore.pyqtSignal(str, int)
    current_direction = "unknown"
    head_directions = ["RIGHT", "LEFT", "STRAIGHT ON", "DOWN", "UP"]
    msgs_directions = ["Turn your head right", "Turn your head left", "Look straight on", "Tilt your head down", "Put your head up"]
    state_imgs = {}
    status_direction = 0
    start_point_time = -1.0
    diff_time = 0.8
    cropped_h = 320
    cropped_w = 280
    border = 30
    face_detector = None

    def __init__(self, parent = None):
        super(Memorize, self).__init__(parent)

    def set_face_detector(self, fd):
        self.face_detector = fd

    def memorize(self, image, current_fb):
        if current_fb is None:
            return image
        height, width, _ = image.shape
        #print(self.current_facebox)
        area_facebox = (current_fb[2] - current_fb[0]) * (current_fb[3] - current_fb[1])
        area_image = self.cropped_h * self.cropped_w
        percent = ( area_facebox / area_image ) * 100


        ht = int((height - self.cropped_h) / 2 )
        wt = int((width - self.cropped_w) / 2 )
        #print("r=" + str(self.r_ok) + " | l=" + str(self.l_ok) + " | s=" + str(self.s_ok))
        #print("dir=" + self.current_direction)


        if not self.check_image(image[ht - self.border:ht + self.cropped_h + self.border, wt - self.border:wt + self.cropped_w + self.border]):
            cv2.putText(image, "Put your face closer to the frame's center", (int(width / 3), height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0),
                        1)
            self.reset_wait_step()
            return image

        msg = None
        if percent > 40 and percent < 90:
            if self.next_step(0):
                self.write_image_step(image, 1)
            elif self.next_step(1):
                self.write_image_step(image, 2)
            elif self.next_step(2):
                self.write_image_step(image, 3)
            elif self.next_step(3):
                self.write_image_step(image, 4)
            elif self.next_step(4):
                self.write_image_step(image, 5)
            else:
                '''cv2.putText(image, "Bad direction", (int(0), height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255),
                            1)'''
                self.reset_wait_step()
        else:
            msg = "Put your face in the frame"
            '''cv2.putText(image, "Area bad", (int(0), height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255),
                        1)'''
            self.reset_wait_step()


        if self.status_direction != 5 and msg == None:
            msg = self.msgs_directions[self.status_direction]
            '''cv2.putText(image, self.msgs_directions[self.status_direction], (int(width / 3), height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0),
                        1)'''
        if msg != None:
            cv2.putText(image, msg, (int(width / 3), height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0),
                        1)
        cv2.putText(image, str(percent), (int(0), height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0),
                    1)
        #cv2.rectangle(image, (0, 0), (width, height_top_bot_corner), (0,0,0),cv2.FILLED) # top corner
        #cv2.rectangle(image, (0, height_top_bot_corner), (width_left_right_corner, height_top_bot_corner + h), (0,0,0),cv2.FILLED) # left corner
        #cv2.rectangle(image, (0, height - height_top_bot_corner), (width, height), (0,0,0), cv2.FILLED) # botton corner
        #cv2.rectangle(image, (width - width_left_right_corner, height_top_bot_corner), (width, height_top_bot_corner + h), (0,0,0),cv2.FILLED) # right corner
        #image = image[height_top_bot_corner:height_top_bot_corner+h, width_left_right_corner:width_left_right_corner+w]
        return image

    def check_image(self, img):
        faceboxes, landmarks = self.face_detector.get_faceboxes(img)
        facebox = None
        l = None

        if len(faceboxes) >= 1:
            facebox = faceboxes[0]
        #else:
        #    print("Null facebox")

        if len(landmarks) >= 1:
            l = landmarks[0]
        #else:
        #    print("Null landmarks")

        return facebox and l

    def reset_wait_step(self):
        #print("Reset")
        self.state_change_signal.emit('r', self.status_direction)
        self.start_point_time = -1.0

    def next_step(self, ok):
        return self.current_direction == self.head_directions[ok] and self.status_direction == ok

    def write_image_step(self, image, ok):
        #print(self.head_directions[ok - 1])
        #time.sleep(2)
        #cv2.imwrite("/tmp/" + self.head_directions[ok - 1].lower() + ".jpg", image[ht:ht+self.cropped_h, wt:wt+self.cropped_w])
        if self.start_point_time == -1.0:
            self.start_point_time = time.time()
            #print("start time")image[ht - 20 :ht+self.cropped_h + 20, wt - 20 :wt+self.cropped_w + 20]
            self.state_change_signal.emit('y', ok - 1)
            return
        cur_time = time.time()

        if cur_time - self.start_point_time > self.diff_time:
            #print("OK - 3 sec")
            '''thread = Thread(target=lambda: [
                self.state_imgs.update({self.head_directions[ok - 1].lower() : image[ht:ht+self.cropped_h, wt:wt+self.cropped_w]}),
                self.change_status_direction(ok),
                self.state_change_signal.emit('g', ok - 1)])
            thread.start()'''
            height, width, _ = image.shape
            ht = int((height - self.cropped_h) / 2 )
            wt = int((width - self.cropped_w) / 2 )

            self.state_imgs.update({self.head_directions[ok - 1].lower() : image[ht - self.border:ht + self.cropped_h + self.border, wt - self.border:wt + self.cropped_w + self.border]})
            self.change_status_direction(ok)
            self.state_change_signal.emit('g', ok - 1)
            self.start_point_time = -1.0
        #state_imgs.update({self.head_directions[ok - 1].lower() : image[ht:ht+self.cropped_h, wt:wt+self.cropped_w]})
        #self.state_change_signal.emit(ok - 1)
        #self.status_direction = ok
    def change_status_direction(self, ok):
        self.status_direction = ok

    def reset(self):
        self.status_direction = 0
        self.current_direction = "unknown"
        self.state_imgs.clear()

    def crop_image_with_fill(self, image):
        height, width, _ = image.shape
        if height <= self.cropped_h or width <= self.cropped_w:
            print ("Incorrect height or width of memorize window")
            return
        height_top_bot_corner = int((height - self.cropped_h) / 2)
        width_left_right_corner = int((width - self.cropped_w) / 2)
        #cv2.rectangle(image, (0, 0), (width, height_top_bot_corner), (0,0,0),cv2.FILLED) # top corner
        #cv2.rectangle(image, (0, height_top_bot_corner), (width_left_right_corner, height_top_bot_corner + h), (0,0,0),cv2.FILLED) # left corner
        #cv2.rectangle(image, (0, height - height_top_bot_corner), (width, height), (0,0,0), cv2.FILLED) # botton corner
        #cv2.rectangle(image, (width - width_left_right_corner, height_top_bot_corner), (width, height_top_bot_corner + h), (0,0,0),cv2.FILLED) # right corner
        image[0:height_top_bot_corner, 0:width] = 255
        image[height_top_bot_corner:height_top_bot_corner + self.cropped_h, 0:width_left_right_corner] = 255
        image[height - height_top_bot_corner:height, 0:width] = 255
        image[height_top_bot_corner:height_top_bot_corner + self.cropped_h, width - width_left_right_corner:width] = 255
        #image = image[height_top_bot_corner:height_top_bot_corner+h, width_left_right_corner:width_left_right_corner+w]
        return image

class ShowVideo(QtCore.QObject):
    camera_port = 0
    camera = cv2.VideoCapture(camera_port)
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    current_facebox = None
    run_video = False
    mode = 0
    memorize_mode = False
    estimator = None
    known_face_encoding = []
    known_face_names = []
    memorize_module = Memorize()

    def __init__(self, parent = None):
        super(ShowVideo, self).__init__(parent)

    def add_person_by_image_path(self, img_path, name):
        img = faceRecognition.load_image_file(img_path)
        faceboxes, landmarks = self.estimator.face_detector.get_faceboxes(img)
        facebox = faceboxes[0] if len(faceboxes) >= 1 else None
        l = landmarks[0] if len(landmarks) >= 1 else None
        if not facebox:
            print("bad facebox")
        if not l:
            print("bad landmark")
        return self.add_person(img, facebox, l, name)

    def add_person(self, img, facebox, landmarks, name):
        if facebox and landmarks:
            upd_facebox = [[int(facebox[1]), int(facebox[2]), int(facebox[3]), int(facebox[0])]]
            my_face_encoding = faceRecognition.face_encodings(img, landmarks, upd_facebox)[0]
            return my_face_encoding, name
        return None, None

    @QtCore.pyqtSlot()
    def startVideo(self):
        self.run_video = True
        _, sample_image = self.camera.read()

        if self.estimator is None:
            self.estimator = HeadPoseEstimator(sample_frame=sample_image, enable_draw=False)
            self.memorize_module.set_face_detector(self.estimator.face_detector)

        '''face_encoding_templ, name_templ = self.add_person("/home/anastasiia/Documents/DeepFR/Nastya_train_12.jpg",
                                                          "NASTYA")
        self.known_face_encoding.append(face_encoding_templ)
        self.known_face_names.append(name_templ)'''

        while self.run_video:
            ret, image = self.camera.read()
            if ret is False:
                break
            res_image = None
            if self.mode == 0:
                res_image = self.memorize_module.crop_image_with_fill(image)
                if self.memorize_mode:
                    res_image = self.memorize_module.memorize(res_image, self.current_facebox)
                res_image = self.get_access(res_image)
            elif self.mode == 1:
                res_image = self.face_recog(image)

            if res_image is not None:
                color_swapped_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
                height, width, _ = color_swapped_image.shape
                qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)

                self.VideoSignal.emit(qt_image)
            else:
                break

    def get_access(self, image):
        image, self.memorize_module.current_direction, faceboxes = self.estimator.get_direction(image)
        self.current_facebox = faceboxes[0] if len(faceboxes) >= 1 else None
        return image

    def face_recog(self, image):
        faceboxes, landmarks = self.estimator.face_detector.get_faceboxes(image)
        # facebox = faceboxes[0] if len(faceboxes) >= 1 else None
        for i in range(len(faceboxes)):
            facebox = faceboxes[i]
            if facebox is not None:
                self.current_facebox = facebox
                l = landmarks[i]
                upd_facebox = [[int(facebox[1]), int(facebox[2]), int(facebox[3]), int(facebox[0])]]
                name = "Unknown"
                if self.known_face_encoding:
                    face_encodings = faceRecognition.face_encodings(image, l, upd_facebox)

                    matches = faceRecognition.compare_faces(self.known_face_encoding, face_encodings[0], 0.55)

                    '''
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]
                    '''


                    face_distances = faceRecognition.face_distance(self.known_face_encoding, face_encodings[0])
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                cv2.rectangle(image,
                              (int(facebox[0]), int(facebox[1])),
                              (int(facebox[2]), int(facebox[3])), (0, 255, 0))
                dist = math.fabs(int(facebox[0]) - int(facebox[2])) / 3
                cv2.putText(image, name, (int(facebox[0] + dist), int(facebox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0),
                            1)
        return image

class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self.fill_screen_cap()


    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0,0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        self.setWindowTitle('DeepFR')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()

    def fill_screen_cap(self):

        image = np.zeros((300, 600, 3), np.uint8)

        color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = color_swapped_image.shape

        qimg = qt_image = QtGui.QImage(color_swapped_image.data,
                                width,
                                height,
                                color_swapped_image.strides[0],
                                QtGui.QImage.Format_RGB888)
        self.setImage(qimg)


class Window(QtWidgets.QMainWindow):

    def __init__(self, stop_work_threads_func):
        QtWidgets.QWidget.__init__(self)
        self.stop_work_threads_func = stop_work_threads_func
        self.image_viewer = ImageViewer()
        #self.push_button1 = QtWidgets.QPushButton('Start')
        #self.push_button2 = QtWidgets.QPushButton('Stop')
        self.push_button3 = QtWidgets.QPushButton('Memorize me')
        self.push_button3.setDisabled(True)
        widget_1 = QtWidgets.QWidget()
        self.comboBox = QtWidgets.QComboBox(widget_1)
        self.comboBox.setGeometry(QtCore.QRect(80, 190, 85, 27))
        self.comboBox.setObjectName("selectMode")
        self.comboBox.addItem("Get access")
        self.comboBox.addItem("Face recognition mode")
        self.comboBox.setDisabled(True)
        self.horizontal_layout = QtWidgets.QHBoxLayout(widget_1)
        self.horizontal_layout.addWidget(self.comboBox)
        #self.label = QtWidgets.QLabel(widget_1)
        #self.label.setObjectName("label")
        #self.label.setText("TEST1")
        #self.horizontal_layout.addWidget(self.label)

        widget_2 = QtWidgets.QWidget()

        self.user_name = QtWidgets.QLineEdit()

        self.set_name_button = QtWidgets.QPushButton('Set')
        img_folger = os.path.join(APP_ROOT_PATH, 'assets/images/')
        self.pixmap_r = QtGui.QPixmap("/".join([img_folger, 'clipartRed.png']))
        self.pixmap_r = self.pixmap_r.scaled(14, 14)
        self.pixmap_g = QtGui.QPixmap("/".join([img_folger, 'clipartGreen.png']))
        self.pixmap_g = self.pixmap_g.scaled(14, 14)
        self.pixmap_y = QtGui.QPixmap("/".join([img_folger, 'clipartYellow.png']))
        self.pixmap_y = self.pixmap_y.scaled(14, 14)

        self.l_user_name = QtWidgets.QLabel('User name ')
        self.l_r = QtWidgets.QLabel()
        self.l_l = QtWidgets.QLabel()
        self.l_s = QtWidgets.QLabel()
        self.l_d = QtWidgets.QLabel()
        self.l_u = QtWidgets.QLabel()
        self.reset_memorize_gui()
        self.set_visible_memorize_gui(False)
        self.change_index_event(mode=0)
        self.horizontal_layout_f = QtWidgets.QHBoxLayout(widget_2)
        self.horizontal_layout_f.addWidget(self.l_user_name)
        self.horizontal_layout_f.addWidget(self.user_name)
        self.horizontal_layout_f.addWidget(self.set_name_button)
        self.horizontal_layout_f.addWidget(self.l_r)
        self.horizontal_layout_f.addWidget(self.l_l)
        self.horizontal_layout_f.addWidget(self.l_s)
        self.horizontal_layout_f.addWidget(self.l_d)
        self.horizontal_layout_f.addWidget(self.l_u)

        vertical_layout = QtWidgets.QVBoxLayout()
        vertical_layout.addWidget(self.image_viewer)
        vertical_layout.addWidget(widget_2)
        vertical_layout.addWidget(widget_1)
        #vertical_layout.addWidget(self.push_button1)
        vertical_layout.addWidget(self.push_button3)
        #vertical_layout.addWidget(self.push_button2)

        layout_widget = QtWidgets.QWidget()
        layout_widget.setLayout(vertical_layout)

        self.setCentralWidget(layout_widget)

    def reset_memorize_gui(self):
        self.user_name.setDisabled(True)
        self.set_name_button.setDisabled(True)
        self.l_r.setPixmap(self.pixmap_r)
        self.l_l.setPixmap(self.pixmap_r)
        self.l_s.setPixmap(self.pixmap_r)
        self.l_d.setPixmap(self.pixmap_r)
        self.l_u.setPixmap(self.pixmap_r)

    def set_state_for(self, name_state, signal_id):
        cur_signal = None
        pixmap = None
        if signal_id == 0:
            cur_signal = self.l_r
        elif signal_id == 1:
            cur_signal = self.l_l
        elif signal_id == 2:
            cur_signal = self.l_s
        elif signal_id == 3:
            cur_signal = self.l_d
        elif signal_id == 4:
            cur_signal = self.l_u
        else:
            return

        if name_state == 'g':
            pixmap = self.pixmap_g
        elif name_state == 'r':
            pixmap = self.pixmap_r
        elif name_state == 'y':
            pixmap = self.pixmap_y
        else:
            return

        cur_signal.setPixmap(pixmap)


    def set_visible_memorize_gui(self, value):
        self.user_name.setVisible(value)
        self.set_name_button.setVisible(value)
        self.l_r.setVisible(value)
        self.l_l.setVisible(value)
        self.l_s.setVisible(value)
        self.l_d.setVisible(value)
        self.l_u.setVisible(value)
        self.l_user_name.setVisible(value)

    def change_index_event(self, mode):
        if mode != 0:
            self.push_button3.setVisible(False)
            self.set_visible_memorize_gui(False)
            return
        self.push_button3.setVisible(True)


    def closeEvent(self, event):
        self.stop_work_threads_func()
        time.sleep(0.3)
        event.accept()


class App(QtCore.QObject):
    start_video_signal = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
        self.init_video_stream = False
        self.thread = QtCore.QThread()

        self.gui = Window(lambda : self.stop_all())
        self.vid = ShowVideo()

        self.vid.moveToThread(self.thread)
        self.thread.start()

        self.connect_signs()

        self.gui.show()
        self.start_video_signal.emit()

    def connect_signs(self):
        self.vid.VideoSignal.connect(self.gui.image_viewer.setImage)
        self.vid.VideoSignal.connect(self.start_video_stream)
        self.gui.comboBox.currentIndexChanged.connect(self.change_index)
        #self.gui.push_button1.clicked.connect(self.vid.startVideo)
        self.start_video_signal.connect(self.vid.startVideo)
        #self.gui.push_button2.clicked.connect(self.stop)
        self.gui.push_button3.clicked.connect(self.change_memorize_mode)
        self.gui.set_name_button.clicked.connect(self.save_images)
        self.vid.memorize_module.state_change_signal.connect(self.change_state)

    def start_video_stream(self, image):
        if not self.init_video_stream:
            gui = self.gui
            gui.push_button3.setDisabled(False)
            gui.comboBox.setDisabled(False)
            self.init_video_stream = True
            self.load_all_ds_images()


    def change_index(self, i):
        #self.gui.label.setText(str(i))
        self.vid.mode = i
        if i == 0:
            self.vid.memorize_mode = False
        self.gui.change_index_event(i)

    def stop(self):
        self.vid.run_video = False

    def stop_all(self):
        self.stop()
        self.thread.quit()

    def change_state(self, str_, ind):
        gui = self.gui
        gui.set_state_for(str_, ind)
        #print("set state for " + str_ + " , index = " + str(ind))
        if ind == 4 and str_ == 'g':
            gui.user_name.setDisabled(False)
            gui.set_name_button.setDisabled(False)

    def save_images(self):
        gui = self.gui
        memorize_module = self.vid.memorize_module
        vid = self.vid
        user_folder = os.path.join(APP_ROOT_PATH, 'assets/dataset/' + gui.user_name.text())
        os.mkdir(user_folder)
        for key, value in memorize_module.state_imgs.items():
            #print("key =" + key)
            #print("value =" + str(value))
            file_name = key + ".jpg"
            cv2.imwrite(os.path.join(user_folder, file_name), value)
            face_encoding_templ, name_templ = vid.add_person_by_image_path(img_path=os.path.join(user_folder, file_name),
                                                                           name=gui.user_name.text())
            #face_encoding_templ, name_templ = vid.add_person(value[0], value[1], value[2],
            #                                                  gui.user_name.text())
            if name_templ:
                vid.known_face_encoding.append(face_encoding_templ)
                vid.known_face_names.append(name_templ)
            else:
                print(key + ": Couldn't add a new person")
        memorize_module.reset()
        gui.reset_memorize_gui()

    def load_images_by_name(self, username):
        vid = self.vid
        username = username[0]
        user_folder = os.path.join(APP_ROOT_PATH, 'assets/dataset/' + username)
        onlyfiles = [f for f in os.listdir(user_folder) if os.path.isfile(os.path.join(user_folder, f))]
        for file_name in onlyfiles:
            print("img path = " +os.path.join(user_folder, file_name))
            print("user_name = " + username)
            face_encoding_templ, name_templ = vid.add_person_by_image_path(img_path=os.path.join(user_folder, file_name),
                                                                           name=username)
            #face_encoding_templ, name_templ = vid.add_person(value[0], value[1], value[2],
            #                                                  gui.user_name.text())
            if name_templ:
                vid.known_face_encoding.append(face_encoding_templ)
                vid.known_face_names.append(name_templ)
            else:
                print(key + ": Couldn't add a new person")

    def load_all_ds_images(self):
        ds_folder = os.path.join(APP_ROOT_PATH, 'assets/dataset/')
        dirs_name = [x[1] for x in os.walk(ds_folder) if x[1]]

        for dir_name in dirs_name:
            self.load_images_by_name(dir_name)

    def change_memorize_mode(self):
        self.vid.memorize_mode = not self.vid.memorize_mode
        self.gui.set_visible_memorize_gui(self.vid.memorize_mode)

if __name__ == '__main__':
    main_app = QtWidgets.QApplication(sys.argv)
    app = App(main_app)
    sys.exit(main_app.exec_())
