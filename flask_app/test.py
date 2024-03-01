import os
import supervision as sv
from ultralytics import YOLO
import cv2



frame = cv2.imread('/Users/chrisgreen/Desktop/naruto/data/images/train/bird_IMG_14f422c48-4a9b-11ea-843c-0242ac1c0002.jpg')



model = YOLO('/Users/chrisgreen/Desktop/naruto/runs/detect/train2/weights/best.pt')  # load a custom model

box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )

CLASS_NAMES_DICT = model.model.names
results = model(frame)#, agnostic_nms=True
result=results[0]
detections = sv.Detections.from_ultralytics(result)


labels = [CLASS_NAMES_DICT[x] for x in detections.class_id]

print(labels)
"""""
labels = [f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections]


print(labels)



frame = box_annotator.annotate(
    scene=frame, 
    detections=detections, 
    labels=labels
)
"""