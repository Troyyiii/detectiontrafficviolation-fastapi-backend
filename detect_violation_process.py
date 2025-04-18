import os
import cv2
import cvzone
import numpy as np
import const_settings as settings
from db import get_connection
from datetime import datetime
from detect_crosswalk import DetectCrosswalk
from deep_sort_realtime.deepsort_tracker import DeepSort

class DetectViolation:
    def __init__(self, travio_model, video_id = None):
        # Load custom trained YOLOv5 model
        self.travio_model = travio_model
        
        # Initiate Deep SORT tracker
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=0.5,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=False
        )
        
        # Initialize area for boundary detection
        self.area = []
        
        # Initialize trails dictionary
        self.trails = {}
        
        # Initialize crosswalk direction check flag
        self.crosswalk_dir_check = False
        
        # initialize traffic light status
        self.traffic_light_status = "Unknown"
        
        # Initialize list of violator ID & counter
        self.traffic_light_violator_list = []
        self.traffic_light_violator_counter = 0
        self.wrong_way_violator_list = []
        self.wrong_way_violator_counter = 0
        self.helmet_violator_list = []
        self.helmet_violator_counter = 0
        
        # Initialize list of clear vehicle
        self.traffic_light_clear_list = []
        
        self.video_id = video_id
    
    def start_detect(self, frame):
        if not self.area:
            detect_crosswalk = DetectCrosswalk(self.travio_model, frame)
            
            if detect_crosswalk.area:
                self.area = detect_crosswalk.area
            
            processed_frame = np.copy(frame)
            cvzone.putTextRect(processed_frame, "Detecting Crosswalk Boundary...", (20, 30), scale=0.8, thickness=1, offset=1)
            
            return processed_frame, self.traffic_light_status, self.traffic_light_violator_counter, self.wrong_way_violator_counter, self.helmet_violator_counter
        else:
            processed_frame = self.detect_object(frame)
            processed_frame = self.draw_bounding_area(processed_frame)
            
            cvzone.putTextRect(processed_frame, f"TV: {self.traffic_light_violator_counter}, WWV: {self.wrong_way_violator_counter}, HV: {self.helmet_violator_counter}", (20, 60), scale=0.8, thickness=1, offset=1)
            
            return processed_frame, self.traffic_light_status, self.traffic_light_violator_counter, self.wrong_way_violator_counter, self.helmet_violator_counter
    
    def detect_object(self, frame):
        results = self.travio_model(frame.copy())
        objects = results.pandas().xyxy[0]
        
        print("\n=== Detecting object ===\n")
        print(f"📌 Detected objects:\n{objects}\n")
        
        tracker_detections = self.set_tracker(objects)
        tracks = self.tracker.update_tracks(tracker_detections, frame=frame)
        
        processed_frame = self.set_traffic_light_status(objects, frame)
        processed_frame = self.draw_bounding_box_line_violation(processed_frame, tracks)
        processed_frame = self.check_helmet_violation(processed_frame, tracks, objects)
        
        return processed_frame

    def set_tracker(self, objects):
        tracker_detections = []
        
        print("\n🎯 Object Tracker:\n")
        
        for _, row in objects.iterrows():
            xmin, ymin, xmax, ymax = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            confidence = float(f"{row['confidence']:.2f}")
            class_id = int(row["class"])
            class_name = row["name"]
            
            if (class_name == "car" or class_name == "motorcycle") and confidence > 0.65:
                print(f"Class: ({class_id}) {class_name}")
                print(f"Confidence: {confidence}")
                print(f"Coordinate: {xmin} {ymin} {xmax} {ymax}\n")
                
                width, height = xmax - xmin, ymax - ymin
                tracker_detections.append(([xmin, ymin, width, height], confidence, {"class_id": class_id, "class_name": class_name, "confidence": confidence}))
                
        return tracker_detections
    
    def set_traffic_light_status(self, objects, frame):
        if self.crosswalk_dir_check:
            green_count, red_count, green_confidence_sum, red_confidence_sum, processed_frame = self.count_traffic_light(objects, frame.copy())
            self.update_traffic_light_status(green_count, red_count, green_confidence_sum, red_confidence_sum)
            
            cvzone.putTextRect(processed_frame, f"Traffic Light Status: {self.traffic_light_status}", (20, 30), scale=0.8, thickness=1, offset=1)
            cvzone.putTextRect(processed_frame, f"G: {green_count} {green_confidence_sum}, R: {red_count} {red_confidence_sum}", (20, 45), scale=0.8, thickness=1, offset=1)
            
            print(f"\n🚦 Traffic Light Status: {self.traffic_light_status}")
            print(f"Red: {red_count}")
            print(f"Green: {green_count}")
            print(f"Red Confidence: {red_confidence_sum}")
            print(f"Green Confidence: {green_confidence_sum}\n")

            return processed_frame
        
        else:
            return frame
    
    def count_traffic_light(self, objects, frame):
        green_count = 0
        red_count = 0
        green_confidence_sum = 0.0
        red_confidence_sum = 0.0
        
        for _, row in objects.iterrows():
            xmin, ymin, xmax, ymax = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            confidence = float(f"{row['confidence']:.2f}")
            class_name = row["name"]
            
            if confidence > 0.75:
                if class_name == "green-light" :
                    green_count += 1
                    green_confidence_sum += confidence
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                    cvzone.putTextRect(frame, f"{class_name}", (max(0, xmin), max(35, ymin)), scale=0.8, thickness=1, offset=1)
                if class_name == "red-light":
                    red_count += 1
                    red_confidence_sum += confidence
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                    cvzone.putTextRect(frame, f"{class_name}", (max(0, xmin), max(35, ymin)), scale=0.8, thickness=1, offset=1)
                    
        return green_count, red_count, green_confidence_sum, red_confidence_sum, frame
    
    def update_traffic_light_status(self, green_count, red_count, green_confidence_sum, red_confidence_sum):
        if green_count > 0 or red_count > 0:
            if green_count > red_count:
                self.traffic_light_status = "Green"
            elif red_count > green_count:
                self.traffic_light_status = "Red"
            else:
                green_avg_confidence = green_confidence_sum / green_count
                red_avg_confidence = red_confidence_sum / red_count

                if green_avg_confidence > red_avg_confidence:
                    self.traffic_light_status = "Green"
                elif red_avg_confidence > green_avg_confidence:
                    self.traffic_light_status = "Red"
                else:
                    self.traffic_light_status = "Unknown"

                print(f"⚠️ Traffic detected same value, count the average confidence. Results: Green: {green_avg_confidence}, Red: {red_avg_confidence}\n")
        else:
            self.traffic_light_status = "Unknown"
    
    def draw_bounding_box_line_violation(self, frame, tracks):        
        print("\n🎯 Tracker Results:\n")
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            track_data = track.get_det_class()
            class_id = track_data.get("class_id", "Unknown")
            class_name = track_data.get("class_name", "Unknown")
            confidence = track_data.get("confidence", None)
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = map(int, ltrb)
            
            # Calculate center of bounding box
            cx = int(xmin + xmax) // 2
            cy = int(ymin + ymax) // 2
            
            cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
            cvzone.putTextRect(frame, f"{track_id}", (max(0, xmin), max(35, ymin)), scale=0.8, thickness=1, offset=1)
            
            if self.video_id is not None:
                self.capture_violation(frame, (xmin, ymin, xmax, ymax), "Traffic Light")
            
            self.update_trails(track_id, cx, ymax)
            
            # Draw trails for each object
            if track_id in self.trails:
                for i in range(1, len(self.trails[track_id])):
                    thickness = int(np.sqrt(64 / float(len(self.trails[track_id]) - i)) * 1.5)
                    thickness = max(1, min(thickness, 4))
                    
                    cv2.line(frame, self.trails[track_id][i-1], self.trails[track_id][i], (255, 0, 0), thickness)
                    
                    # Get object direction
                    obj_dir = self.get_direction(self.trails[track_id][i-1], self.trails[track_id][i])
                    
                    cvzone.putTextRect(frame, f"{obj_dir}", (xmax, ymin), scale=0.8, thickness=1, offset=1)
                    
                    if self.crosswalk_dir_check is False:
                        cvzone.putTextRect(frame, "Waiting for boundary direction status...", (20, 30), scale=0.8, thickness=1, offset=1)
                        
                        for area in self.area:
                            if area["status_dir"] == "Undefined":
                                if self.do_lines_intersect(self.trails[track_id][i-1], self.trails[track_id][i], area["coords"][0], area["coords"][1]):
                                    if track_id not in area["counted_idx"]:
                                        if obj_dir == "North":
                                            area["north_count"] += 1
                                        elif obj_dir == "South":
                                            area["south_count"] += 1
                                        
                                        area["counted_idx"].add(track_id)
                                        
                                        if area["north_count"] >= 5:
                                            area["status_dir"] = "North"
                                        elif area["south_count"] >= 5:
                                            area["status_dir"] = "South"
                                        
                                        print(f"Area: {area['coords']}")
                                        print(f"North Count: {area['north_count']}")
                                        print(f"South Count: {area['south_count']}")
                                        print(f"Status: {area['status_dir']}\n")
                        
                        if all(area["status_dir"] != "Undefined" for area in self.area):
                            self.crosswalk_dir_check = True
                            print("✅ All areas already have direction status.\n")
                    
                    if self.crosswalk_dir_check:
                        for area in self.area:
                            # check traffic light violation
                            if area["status_dir"] == "North":
                                if self.do_lines_intersect(self.trails[track_id][i-1], self.trails[track_id][i], area["coords"][0], area["coords"][1]):
                                    if self.traffic_light_status == "Red":
                                        if track_id not in self.traffic_light_violator_list and track_id not in self.traffic_light_clear_list:
                                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                                            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                                            self.traffic_light_violator_list.append(track_id)
                                            self.traffic_light_violator_counter += 1
                                            if self.video_id is not None:
                                                self.capture_violation(frame, (xmin, ymin, xmax, ymax), "Traffic Light")
                                            print(f"\n🚩 Violator detected! ID: {track_id}\nTotal Violator: {self.traffic_light_violator_counter}\nViolator list: {self.traffic_light_violator_list}\n")
                                    if self.traffic_light_status == "Green":
                                        if track_id not in self.traffic_light_clear_list and track_id not in self.traffic_light_violator_list:
                                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                                            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                                            self.traffic_light_clear_list.append(track_id)
                            
                            # check wrong way violation
                            if obj_dir == "North":
                                if area["status_dir"] == "South":
                                    if self.do_lines_intersect(self.trails[track_id][i-1], self.trails[track_id][i], area["coords"][0], area["coords"][1]):
                                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                                        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                                        if track_id not in self.wrong_way_violator_list:
                                            self.wrong_way_violator_list.append(track_id)
                                            self.wrong_way_violator_counter += 1
                                            if self.video_id is not None:
                                                self.capture_violation(frame, (xmin, ymin, xmax, ymax), "Wrong Way")
                                            print("\n🚩 South line violated!")
                                            print(f"Wrong way violator detected! ID: {track_id}\nTotal Violator: {self.wrong_way_violator_counter}\nViolator list: {self.wrong_way_violator_list}\n")
                            
                            if obj_dir == "South":
                                if area["status_dir"] == "North":
                                    if self.do_lines_intersect(self.trails[track_id][i-1], self.trails[track_id][i], area["coords"][0], area["coords"][1]):
                                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                                        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                                        if track_id not in self.wrong_way_violator_list:
                                            self.wrong_way_violator_list.append(track_id)
                                            self.wrong_way_violator_counter += 1
                                            if self.video_id is not None:
                                                self.capture_violation(frame, (xmin, ymin, xmax, ymax), "Wrong Way")
                                            print("\n🚩 North line violated!")
                                            print(f"Wrong way violator detected! ID: {track_id}\nTotal Violator: {self.wrong_way_violator_counter}\nViolator list: {self.wrong_way_violator_list}\n")
            
            print(f"Track ID: {track_id}")
            print(f"Class: ({class_id}) {class_name}")
            print(f"Confidence: {confidence}")
            print(f"Bounding Box: {xmin}, {ymin}, {xmax}, {ymax}\n")
        
        if self.crosswalk_dir_check is False:
            print("🔍 Define Boundary Line Direction Status\n")
        
        return frame
    
    def check_helmet_violation(self, processed_frame, tracks, objects):
        for _, row in objects.iterrows():
            xmin, ymin, xmax, ymax = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            confidence = float(f"{row['confidence']:.2f}")
            class_name = row["name"]
            
            if class_name == "no-helm" and confidence > 0.7:
                cx = int(xmin + xmax) // 2
                cy = int(ymin + ymax) // 2
                
                cv2.rectangle(processed_frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    
                    track_id = track.track_id
                    track_data = track.get_det_class()
                    class_name = track_data.get("class_name", "Unknown")
                    ltrb = track.to_ltrb()
                    rxmin, rymin, rxmax, rymax = map(int, ltrb)
                    
                    if class_name == "motorcycle":
                        if rxmin <= cx <= rxmax and rymin <= cy <= rymax:
                            if track_id not in self.helmet_violator_list:
                                self.helmet_violator_counter += 1
                                self.helmet_violator_list.append(track_id)
                                cv2.rectangle(processed_frame, (rxmin, rymin), (rxmax, rymax), (0, 0, 255), 1)
                                cv2.rectangle(processed_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                                if self.video_id is not None:
                                    self.capture_violation(processed_frame, (xmin, ymin, xmax, ymax), "Helmet")
                                print(f"\n🚩 Helmet violation detected! Rider ID: {track_id}\nTotal Violations: {self.helmet_violator_counter}\nViolator list: {self.helmet_violator_list}\n")
                                
        return processed_frame
    
    def update_trails(self, track_id, cx, ymax):
        if track_id not in self.trails:
            self.trails[track_id] = []
        self.trails[track_id].append((cx, ymax))
    
    def get_direction(self, p1, p2):
        direction = ""
        
        if p1[1] > p2[1]:
            direction = "North"
        elif p1[1] < p2[1]:
            direction = "South"
        
        return direction
    
    def do_lines_intersect(self, p1, p2, q1, q2):
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)
    
    def draw_bounding_area(self, frame):
        if self.area:
            for area in self.area:
                coords = area["coords"]
                north_count = area["north_count"]
                south_count = area["south_count"]
                status_dir = area["status_dir"]
                
                if len(coords) > 1:
                    for i in range(len(coords) - 1):
                        cv2.line(frame, (coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]), (255, 255, 255), 2)
                
                cvzone.putTextRect(frame, f"{north_count} {south_count} {status_dir}", (coords[0][0], coords[0][1] - 5), scale=0.8, thickness=1, offset=1)
        
        return frame

    def capture_violation(self, frame, bbox, type, padding = 50):
        rxmin, rymin, rxmax, rymax = bbox
        xmin = max(rxmin - padding, 0)
        ymin = max(rymin - padding, 0)
        xmax = min(rxmax + padding, frame.shape[1])
        ymax = min(rymax + padding, frame.shape[0])
        
        timestamp = datetime.now().strftime("%H.%M.%S")
        cropped_frame = frame[ymin:ymax, xmin:xmax]
        
        conn = get_connection()
        cur = conn.cursor()
        
        select_query = """
            SELECT *
            FROM video_uploads
            WHERE id = %s;
        """
        cur.execute(select_query, (self.video_id,))
        video_data = cur.fetchone()
        video_path = video_data[1]
        video_dir = os.path.dirname(video_path)
        uuid = os.path.basename(video_dir)
        
        video_dir = os.path.join(settings.UPLOAD_DIR, uuid)
        
        if type == "Traffic Light":
            violation_dir = os.path.join(video_dir, "traffic_light_violations")
            file_name = f"{timestamp}_{self.traffic_light_violator_counter}.jpg"
        elif type == "Wrong Way":
            violation_dir = os.path.join(video_dir, "wrong_way_violations")
            file_name = f"{timestamp}_{self.wrong_way_violator_counter}.jpg"
        elif type == "Helmet":
            violation_dir = os.path.join(video_dir, "helmet_violations")
            file_name = f"{timestamp}_{self.helmet_violator_counter}.jpg"
        
        os.makedirs(violation_dir, exist_ok=True)
        file_path = os.path.join(violation_dir, file_name)
        
        cv2.imwrite(file_path, cropped_frame)
        
        
        query = """
            INSERT INTO capture_violations (video_id, violation_type, image_path)
            VALUES (%s, %s, %s)
            RETURNING id;
        """
        cur.execute(query, (self.video_id, type, file_path))
        conn.commit()
        cur.close()
        conn.close()