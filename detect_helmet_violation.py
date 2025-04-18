import cv2
import cvzone
import numpy as np
from load_model import load_model
from sort.sort import Sort

class DetectHelmetViolation:
    def __init__(self, helmet_model):
        # load custom trained model
        self.helmet_model = load_model(helmet_model)
        
        # initialize tracker
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # initialize list of violator ID & counter
        self.violation_counter = 0
        self.violator_id_list = []
        
    def start_detect(self, frame):
        processed_frame = self.detect_object(frame.copy())
        return processed_frame
    
    def detect_object(self, frame):
        results = self.helmet_model(frame.copy())
        objects = results.pandas().xyxy[0]
        
        rider_detections, no_helmet_detections = self.get_detections(objects)
        tracker_results = self.tracker.update(rider_detections)
        
        print(f"Rider Detection : \n{rider_detections}\n")
        print(f"No-helm Detection : \n{no_helmet_detections}\n")
        print(f"Tracker Results : \n{tracker_results}\n")
        
        processed_frame = self.draw_bounding_box(frame.copy(), tracker_results)
        processed_frame = self.check_helmet_violation(processed_frame, tracker_results, no_helmet_detections)
        
        return processed_frame
    
    def get_detections(self, objects):
        rider_detections = np.empty((0, 5))
        no_helmet_detections = np.empty((0, 5))
        
        for _, row in objects.iterrows():
            xmin = int(row["xmin"])
            ymin = int(row["ymin"])
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])
            confidence = float(f"{row['confidence']:.2f}")
            class_id = int(row["class"])
            name = row["name"]
            
            if confidence >= 0.7 and class_id == 2:
                print(f"Class: ({class_id}){name}")
                print(f"Confidence: {confidence}\n")
                
                coordinate_list = np.array([xmin, ymin, xmax, ymax, confidence])
                rider_detections = np.vstack([rider_detections, coordinate_list])
            
            if confidence >= 0.8 and class_id == 1:
                print(f"Class: ({class_id}){name}")
                print(f"Confidence: {confidence}\n")
                
                coordinate_list = np.array([xmin, ymin, xmax, ymax, confidence])
                no_helmet_detections = np.vstack([no_helmet_detections, coordinate_list])
        
        return rider_detections, no_helmet_detections
    
    def draw_bounding_box(self, frame, tracker_results):
        for result in tracker_results:
            xmin, ymin, xmax, ymax, idx = map(int, result)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cvzone.putTextRect(frame, f"{idx}", (max(0, xmin), max(35, ymin)), scale=0.8, thickness=1, offset=3)
        
        return frame
    
    def check_helmet_violation(self, frame, tracker_results, no_helmet_detections):
        for detection in no_helmet_detections:
            xmin, ymin, xmax, ymax, _ = map(int, detection)
            cx = int(xmin + xmax) // 2
            cy = int(ymin + ymax) // 2
            
            for result in tracker_results:
                rxmin, rymin, rxmax, rymax, idx = map(int, result)
                if rxmin <= cx <= rxmax and rymin <= cy <= rymax:
                    if idx not in self.helmet_violator_id_list:
                        self.helmet_violator_counter += 1
                        self.helmet_violator_id_list.append(idx)
                        cv2.rectangle(frame, (rxmin, rymin), (rxmax, rymax), (0, 0, 255), 2)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        print(f"Helmet violation detected! Rider ID: {idx}\nTotal Violations: {self.helmet_violator_counter}\nViolator list: {self.helmet_violator_id_list}\n")
                        
        return frame