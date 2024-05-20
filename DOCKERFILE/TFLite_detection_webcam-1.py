import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

class VideoStream:
    """Camera object that controls video streaming"""
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

class ObjectDetector:
    def __init__(self, model_dir, graph, labels, threshold, resolution, use_edgetpu):
        self.model_dir = model_dir
        self.graph = graph
        self.labels = labels
        self.threshold = threshold
        self.resolution = resolution
        self.use_edgetpu = use_edgetpu

        self.load_model()

    def load_model(self):
        # Load model
        self.interpreter = cv2.dnn.readNetFromTensorflow(os.path.join(self.model_dir, self.graph),
                                                          os.path.join(self.model_dir, self.labels))
        # Use Edge TPU if enabled
        if self.use_edgetpu:
            self.interpreter.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

    def detect_objects(self, frame):
        # Perform object detection
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True)
        self.interpreter.setInput(blob)
        detections = self.interpreter.forward()

        # Parse detections
        objects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.threshold:
                class_id = int(detections[0, 0, i, 1])
                label = self.labels[class_id]
                (h, w) = frame.shape[:2]
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                objects.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": (startX, startY, endX, endY)
                })
        return objects

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .pb file is located in', required=True)
    parser.add_argument('--graph', help='Name of the .pb file, if different than frozen_inference_graph.pb',
                        default='frozen_inference_graph.pb')
    parser.add_argument('--labels', help='Name of the labels file, if different than labels.txt',
                        default='labels.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support '
                                             'the resolution entered, errors may occur.', default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')
    args = parser.parse_args()

    model_dir = args.modeldir
    graph = args.graph
    labels_file = args.labels
    threshold = float(args.threshold)
    resolution = args.resolution
    use_edgetpu = args.edgetpu

    resolution = tuple(map(int, resolution.split('x')))

    with open(os.path.join(model_dir, labels_file), 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')

    vs = VideoStream().start()
    time.sleep(2.0)
    detector = ObjectDetector(model_dir, graph, labels, threshold, resolution, use_edgetpu)

    while True:
        frame = vs.read()
        objects = detector.detect_objects(frame)

        for obj in objects:
            startX, startY, endX, endY = obj['bbox']
            label = "{}: {:.2f}%".format(obj["label"], obj["confidence"] * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()