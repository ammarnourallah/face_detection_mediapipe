# Import packages
import importlib
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
from postprocess import _tensors_to_detections, _weighted_non_max_suppression
import time

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    from tensorflow.lite.python.interpreter import load_delegate


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480)):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        # ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()
        self.img_size = np.asarray(self.frame.shape)[0:2]

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame  # [::-1]

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


imW, imH = 640, 480
CWD_PATH = os.getcwd()

EDGE_TPU = False
if EDGE_TPU:
    face_model_path = 'model/face-detector-quantized_edgetpu.tflite'
    face_interpreter = Interpreter(model_path=os.path.join(CWD_PATH, face_model_path),
                                   experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    face_model_path = 'model/face_detection_front.tflite'
    face_interpreter = Interpreter(model_path=os.path.join(CWD_PATH, face_model_path))

face_interpreter.allocate_tensors()

# Get model details
face_input_details = face_interpreter.get_input_details()[0]
face_output_details = face_interpreter.get_output_details()
height = face_input_details['shape'][1]
width = face_input_details['shape'][2]

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
# Initialize video stream
videostream = VideoStream(resolution=(imW, imH)).start()
time.sleep(1)
anchors = np.load('anchors.npy')

while True:
    # opencv
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    # Grab frame from video stream
    frame1 = videostream.read()
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    cut_img = np.zeros((640, 640, 3), np.uint8)
    cut_img[80:560] = frame
    frame_resized = cv2.resize(cut_img, (width, height), cv2.INTER_AREA)

    st1 = time.time()
    if EDGE_TPU:
        input_data = np.expand_dims(frame_resized, axis=0)
    else:
        input_data = np.expand_dims(frame_resized / 127.5 - 1, axis=0).astype(np.float32)

    face_interpreter.set_tensor(face_input_details['index'], input_data)
    face_interpreter.invoke()
    raw_box = face_interpreter.get_tensor(face_output_details[0]['index'])[0]
    raw_score = face_interpreter.get_tensor(face_output_details[1]['index'])[0]
    st2 = time.time()

    # 3. Postprocess the raw predictions:
    detections = _tensors_to_detections(raw_box, raw_score, anchors)

    # 4. Non-maximum suppression to remove overlapping detections:
    faces = _weighted_non_max_suppression(detections)
    print('Inference time: ', (st2 - st1) * 1000, ' Post-processing: ', (time.time() - st2) * 1000)

    for rc in faces:
        cv2.rectangle(frame, (int(rc[1]*imW), int(rc[0]*imW - 80)), (int(rc[3]*imW), int(rc[2]*imW - 80)), (0, 255, 0), 2)
        for i in range(6):
            cv2.circle(frame, (int(rc[i * 2 + 4] * imW), int(rc[i * 2 + 5] * imW - 80)), 2, (0, 0, 255), 2)
        # margin = (rc[3] - rc[1]) * 0.1
        # x1 = int(max(0, (rc[1] - margin) * imW))
        # y1 = int(max(0, rc[0] * imH))
        # x2 = int(min(imW, (rc[3] + margin) * imW))
        # y2 = int(min(imH, rc[2] * imH))

    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                cv2.LINE_AA)
    cv2.imshow('Face Detection', frame)
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
