import tensorflow as tf
import cv2
import numpy as np
import os

class MarkDetector:
    """Facial landmark detector by Convolutional Neural Network"""

    def __init__(self, mark_model='models/frozen_inference_graph.pb', saved_model = False):
        """Initialization"""

        self.saved_model = saved_model
        if (self.saved_model == False):
            model_path,_ = os.path.split(os.path.realpath(__file__))
            mark_model = os.path.join(model_path, mark_model)
            # A face detector is required for mark detection.

            self.CNN_INPUT_SIZE = 128
            self.marks = None

            # Get a TensorFlow session ready to do landmark detection
            # Load a (frozen) Tensorflow model into memory.
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(mark_model, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            self.graph = detection_graph
            self.sess = tf.Session(graph=detection_graph)
        else:
            mark_model = 'models/pose_model'
            model_path, _ = os.path.split(os.path.realpath(__file__))
            mark_model = os.path.join(model_path, mark_model)

            # A face detector is required for mark detection.

            self.CNN_INPUT_SIZE = 128
            self.marks = None

            # Get a TensorFlow session ready to do landmark detection
            # Load a Tensorflow saved model into memory.
            self.graph = tf.Graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.graph, config=config)

            # Restore model from the saved_model file, that is exported by
            # TensorFlow estimator.
            tf.saved_model.loader.load(self.sess, ["serve"], mark_model)

    @staticmethod
    def draw_box(image, boxes, box_color=(255, 255, 255)):
        """Draw square boxes on image"""
        for box in boxes:
            cv2.rectangle(image,
                          (box[0], box[1]),
                          (box[2], box[3]), box_color)

    def detect_marks(self, image, faceboxes):
        arr_marks = []
        for facebox in faceboxes:
            # Get face image
            face_img = image[int(facebox[1]): int(facebox[3]),
                             int(facebox[0]): int(facebox[2])]
            if face_img is None:
                print("ERROR: Something has gone wrong.")

            face_img = cv2.resize(face_img, (self.CNN_INPUT_SIZE,self.CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            if (self.saved_model == False):
                """Detect marks from image"""
                # Get result tensor by its name.
                logits_tensor = self.graph.get_tensor_by_name('logits/BiasAdd:0')

                # Actual detection.
                predictions = self.sess.run(
                    logits_tensor,
                    feed_dict={'input_image_tensor:0': face_img})

                # Convert predictions to landmarks.
                marks = np.array(predictions).flatten()
                marks = np.reshape(marks, (-1, 2))
            else:
                # Get result tensor by its name.
                logits_tensor = self.graph.get_tensor_by_name(
                    'layer6/final_dense:0')

                # Actual detection.
                predictions = self.sess.run(
                    logits_tensor,
                    feed_dict={'image_tensor:0': [face_img]})

                # Convert predictions to landmarks.
                marks = np.array(predictions).flatten()[:136]
                marks = np.reshape(marks, (-1, 2))

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]
            arr_marks.append(marks)

        return arr_marks

    @staticmethod
    def draw_marks(image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 1, color, -1, cv2.LINE_AA)
