import numpy as np
import cv2 as cv
from ultralytics import YOLO
import os
import supervision as sv

class Annotator:
    def __init__(self):

        self.MODEL_DIR = '/Users/chrisgreen/Desktop/naruto/runs/detect/train2/weights'

        self.model = self.load_yolo_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )

    
    def load_yolo_model(self):
        """Loads the YOLOv8 model."""
        # Load the model
        model = YOLO(os.path.join(self.MODEL_DIR, 'best.pt'))

        # Return the model
        return model

    def detectionSupervision(self,frame):
        results = self.model(frame)#, agnostic_nms=True
        result=results[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # self.conter.conter(result,self.classNames())
        # print(bbox_id,"->id")
        
        labels = [self.CLASS_NAMES_DICT[x] for x in detections.class_id]
        frame = self.box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        return frame


    def generate_frames(self, video_path, file_name):
        cap = cv.VideoCapture(os.path.join(video_path, file_name))
        success, frame = cap.read()
        H, W, _ = frame.shape
        outname = file_name.split('.')[0] + "_detected." + file_name.split('.')[-1]
        out = cv.VideoWriter(os.path.join(video_path,outname), cv.VideoWriter_fourcc(*'XVID'), int(cap.get(cv.CAP_PROP_FPS)), (W, H))

        while True:
        
            ## read the camera frame
            success,frame=cap.read()
            if not success:
                break
            else:
                frame = self.detectionSupervision(frame)
                out.write(frame)

            
                
        

