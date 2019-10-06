import cv2
import numpy as np
import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal
from HeadPoseEstimation.head_pose_estimation import HeadPoseEstimator

class ShowVideo(QtCore.QObject):
    camera_port = 0
    camera = cv2.VideoCapture(camera_port)
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    run_video = False

    def __init__(self, parent = None):
        super(ShowVideo, self).__init__(parent)

    @QtCore.pyqtSlot()
    def startVideo(self):
        self.run_video = True
        _, sample_image = self.camera.read()
        estimator = HeadPoseEstimator(sample_image)
        while self.run_video:
            ret, image = self.camera.read()
            if ret is False:
                break
            image = estimator.get_direction(image)
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width, _ = color_swapped_image.shape

            #width = camera.set(CAP_PROP_FRAME_WIDTH, 1600)
			#height = camera.set(CAP_PROP_FRAME_HEIGHT, 1080)
			#camera.set(CAP_PROP_FPS, 15)

            qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)

            self.VideoSignal.emit(qt_image)


class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)


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

class Window(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        self.image_viewer = ImageViewer()
        self.push_button1 = QtWidgets.QPushButton('Start')
        self.push_button2 = QtWidgets.QPushButton('Stop')
        vertical_layout = QtWidgets.QVBoxLayout()

        vertical_layout.addWidget(self.image_viewer)
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
        self.gui.push_button1.clicked.connect(self.vid.startVideo)
        self.gui.push_button2.clicked.connect(self.stop)

    def stop(self):
        self.vid.run_video = False

if __name__ == '__main__':
    main_app = QtWidgets.QApplication(sys.argv)
    app = App(main_app)
    sys.exit(main_app.exec_())
