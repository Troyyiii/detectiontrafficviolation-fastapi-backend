import cv2
import time
import const_settings as settings
from detect_violation import DetectViolation

def main():
    start_time = time.time()
    capture = cv2.VideoCapture("./assets/test_video_cam.mp4")
    test_detection = DetectViolation(settings.TRAVIO_MODEL_PATH)
    
    # Output video file
    output_path = "./output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for AVI format
    fps = capture.get(cv2.CAP_PROP_FPS)
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (640, 360))
    
    while capture.isOpened():
        ret, frame = capture.read()
        
        if not ret:
            print("Error: failed to read frame")
            break
        
        default_frame = cv2.resize(frame, (640, int(640 * (9/16))))
        processed_frame = test_detection.start_detect(default_frame)
        
        cv2.imshow("Traffic Violation", processed_frame)
        
        # Save frame to output video
        output_video.write(processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    capture.release()
    output_video.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Output video saved to: {output_path}")

if __name__ == "__main__":
    main()