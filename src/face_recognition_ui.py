import cv2
import numpy as np
import sys
import math
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal
from HeadPoseEstimation.head_pose_estimation import HeadPoseEstimator
import faceRecognition

class ShowVideo(QtCore.QObject):
    camera_port = 0
    camera = cv2.VideoCapture(camera_port)
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    run_video = False
    mode = 0
    estimator = None
    known_face_encoding = []
    known_face_names = []

    def __init__(self, parent = None):
        super(ShowVideo, self).__init__(parent)

    def add_person(self, img_path, name):
        img = faceRecognition.load_image_file(img_path)
        faceboxes, landmarks = self.estimator.face_detector.get_faceboxes(img)
        facebox = faceboxes[0] if len(faceboxes) >= 1 else None
        l = landmarks[0]
        upd_facebox = [[int(facebox[1]), int(facebox[2]), int(facebox[3]), int(facebox[0])]]
        my_face_encoding = faceRecognition.face_encodings(img, l, upd_facebox)[0]
        return my_face_encoding, name

    @QtCore.pyqtSlot()
    def startVideo(self):
        self.run_video = True
        _, sample_image = self.camera.read()

        if self.estimator is None:
            self.estimator = HeadPoseEstimator(sample_image)

        face_encoding_templ, name_templ = self.add_person("/home/anastasiia/Documents/DeepFR/Nastya_train_12.jpg",
                                                          "NASTYA")
        self.known_face_encoding.append(face_encoding_templ)
        self.known_face_names.append(name_templ)

        while self.run_video:
            ret, image = self.camera.read()
            if ret is False:
                break
            res_image = None
            if self.mode == 0:
                res_image = self.get_access(image)
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
        image = self.estimator.get_direction(image)
        return image

    def face_recog(self, image):
        faceboxes, landmarks = self.estimator.face_detector.get_faceboxes(image)
        facebox = faceboxes[0] if len(faceboxes) >= 1 else None
        if facebox is not None:
            l = landmarks[0]
            upd_facebox = [[int(facebox[1]), int(facebox[2]), int(facebox[3]), int(facebox[0])]]
            face_encodings = faceRecognition.face_encodings(image, l, upd_facebox)
        
            matches = faceRecognition.compare_faces(self.known_face_encoding, face_encodings[0], 0.47)
            name = "Unknown"

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

        image = np.zeros((300, 300, 3), np.uint8)

        color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = color_swapped_image.shape

        qimg = qt_image = QtGui.QImage(color_swapped_image.data,
                                width,
                                height,
                                color_swapped_image.strides[0],
                                QtGui.QImage.Format_RGB888)
        self.setImage(qimg)


class Window(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        self.image_viewer = ImageViewer()
        self.push_button1 = QtWidgets.QPushButton('Start')
        self.push_button2 = QtWidgets.QPushButton('Stop')

        widget_1 = QtWidgets.QWidget()
        self.comboBox = QtWidgets.QComboBox(widget_1)
        self.comboBox.setGeometry(QtCore.QRect(80, 190, 85, 27))
        self.comboBox.setObjectName("selectMode")
        self.comboBox.addItem("Get access")
        self.comboBox.addItem("Face recognition mode")
        self.horizontal_layout = QtWidgets.QHBoxLayout(widget_1)
        self.horizontal_layout.addWidget(self.comboBox)
        #self.label = QtWidgets.QLabel(widget_1)
        #self.label.setObjectName("label")
        #self.label.setText("TEST1")
        #self.horizontal_layout.addWidget(self.label)

        vertical_layout = QtWidgets.QVBoxLayout()
        vertical_layout.addWidget(self.image_viewer)
        vertical_layout.addWidget(widget_1)
        vertical_layout.addWidget(self.push_button1)
        vertical_layout.addWidget(self.push_button1)
        vertical_layout.addWidget(self.push_button2)

        layout_widget = QtWidgets.QWidget()
        layout_widget.setLayout(vertical_layout)

        self.setCentralWidget(layout_widget)


class App(QtCore.QObject):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

        self.gui = Window()
        self.vid = ShowVideo()
        self.thread = QtCore.QThread()
        self.vid.moveToThread(self.thread)
        self.thread.start()

        self.connect_signs()
        self.gui.show()

    def connect_signs(self):
        self.vid.VideoSignal.connect(self.gui.image_viewer.setImage)
        self.gui.comboBox.currentIndexChanged.connect(self.change)
        self.gui.push_button1.clicked.connect(self.vid.startVideo)
        self.gui.push_button2.clicked.connect(self.stop)

    def change(self, i):
        #self.gui.label.setText(str(i))
        self.vid.mode = i
    def stop(self):
        self.vid.run_video = False

if __name__ == '__main__':
    main_app = QtWidgets.QApplication(sys.argv)
    app = App(main_app)
    sys.exit(main_app.exec_())
