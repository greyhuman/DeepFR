FaceboxDetection

Interface:

    file_name: face_detector.py
    class_name: FaceDetector
    face_detection_func_name: get_faceboxes
    input: self, image (any resolution)
    output: array[n_faces, 5] -  (x1, y1, x2, y2, confidence)