import cv2
import numpy as np

class DetectCrosswalk:
    def __init__(self, travio_model, frame):
        self.travio_model = travio_model
        self.frame = frame.copy()
        self.area = []
        
        self.start_detect()
    
    def start_detect(self):
        halfed_frame = self.crop_to_half(self.frame)
        self.check_crosswalk(halfed_frame)
    
    def crop_to_half(self, frame):
        height, _ = frame.shape[:2]
        cropped_frame = frame[height // 2:, :]
        output_frame = np.zeros_like(frame)
        output_frame[height // 2:, :] = cropped_frame
        
        return output_frame
    
    def crop_to_trapezoid(self, frame):
        height, width = frame.shape[:2]
        region_of_interest_vertices = [
            (0, height),
            (width / 7, height / 5),
            (6 * width / 7, height / 2),
            (width, height)
        ]
        
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, np.array([region_of_interest_vertices], np.int32), 255)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        return masked_frame
    
    def check_crosswalk(self, crop_frame):
        results = self.travio_model(crop_frame)
        objects = results.pandas().xyxy[0]
        
        print("\n=== Detecting crosswalk boundary ===\n")
        print(f"📌 Detected objects:\n{objects}\n")
        
        if not objects.empty:
            if self.only_zebracross_detected(objects):
                self.process_zebracross(objects)
            else:
                print("⚠️ Another object detected, search again...\n")
        else:
            print("❌ No object detected!\n")
    
    def only_zebracross_detected(self, objects):
        for _, row in objects.iterrows():
            class_name = row["name"]
            if class_name != "zebracross":
                return False
        return True

    def process_zebracross(self, objects):
        for _, row in objects.iterrows():
            xmin, ymin, xmax, ymax = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            confidence = float(f"{row['confidence']:.2f}")
            
            if confidence > 0.5:
                cy = (ymin + ymax) // 2
            
                self.area.append({
                    "coords": np.array([
                        [xmin, cy],
                        [xmax, cy]
                    ], dtype=np.int32),
                    "north_count": 0,
                    "south_count": 0,
                    "status_dir": "Undefined",
                    "counted_idx": set()
                })
                
                print("✅ Zebra cross detected with clear object!")
                print(f"Zebra cross detected at: {self.area}\n")