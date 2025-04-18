import os
import cv2
import yt_dlp
import asyncio
import const_settings as settings
from uuid import uuid4
from db import get_connection
from load_model import load_model
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
from detect_violation_process import DetectViolation
from fastapi import FastAPI, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Loading the model...")
        app.state.travio_model = load_model(settings.TRAVIO_MODEL_PATH)
        yield
        print("Clearing the model...")
        del app.state.travio_model
    except asyncio.exceptions.CancelledError as e:
        print(f"Error occured: {e.args}")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def main():
    content = """
        <body>
        <form action="/upload/" enctype="multipart/form-data" method="post">
        <input name="file" type="file">
        <input type="submit">
        </form>
        </body>
    """
    return HTMLResponse(content=content)

# API: Upload video file
@app.post('/upload/')
async def upload_file(file: UploadFile):
    try:
        file_name = file.filename
        unique_id = str(uuid4())
        
        uuid_dir = os.path.join(settings.UPLOAD_DIR, unique_id)
        os.makedirs(uuid_dir, exist_ok=True)
        file_path = os.path.join(uuid_dir, file_name)
        
        with open(file_path, 'wb') as f:
            f.write(await file.read())
        
        conn = get_connection()
        cur = conn.cursor()
        query = """
            INSERT INTO video_uploads (file_path, file_name)
            VALUES (%s, %s)
            RETURNING id;
        """
        cur.execute(query, (file_path, file_name))
        video_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        
        return {
            "status": "success",
            "id": video_id,
            "filename": file_name,
            "path": file_path
        }
        
    except Exception as e:
        print(f"Error occured: {e}")
        return {"status": "failed", "error": str(e)}

# API: Detect line violation by upload file
@app.websocket("/ws/detectViolation/")
async def detect_violation(websocket: WebSocket):
    await websocket.accept()
    
    try:
        video_id = await websocket.receive_text()
        
        conn = get_connection()
        cur = conn.cursor()
        query = """
            SELECT *
            FROM video_uploads
            WHERE id = %s;
        """
        cur.execute(query, (video_id,))
        video_data = cur.fetchone()
        
        video_path = video_data[1]
        
        capture = cv2.VideoCapture(video_path)
        detection = DetectViolation(app.state.travio_model, video_id)
        
        while capture.isOpened():
            ret, frame = capture.read()
            
            if not ret:
                print("Error: failed to read frame")
                break
            
            default_frame = cv2.resize(frame, (640, int(640 * (9/16))))
            processed_frame, traffic_light_status, traffic_light_violator_counter, wrong_way_violator_counter, helmet_violator_counter = detection.start_detect(default_frame.copy())
            
            _, default_buffer = cv2.imencode('.jpg', default_frame)
            _, processed_frame_buffer = cv2.imencode('.jpg', processed_frame)
            default_frame_bytes = default_buffer.tobytes()
            processed_frame_bytes = processed_frame_buffer.tobytes()
            
            result = {
                "traffic_light_status": traffic_light_status,
                "traffic_light_violator_counter": traffic_light_violator_counter,
                "wrong_way_violator_counter": wrong_way_violator_counter,
                "helmet_violator_counter": helmet_violator_counter
            }
            
            await websocket.send_bytes(default_frame_bytes)
            await websocket.send_bytes(processed_frame_bytes)
            await websocket.send_json(result)
            await asyncio.sleep(0.03)
        
    except Exception as e:
        print(f"Error occured: {e}")
        return {"status": "failed", "error": str(e)}
    
    finally:
        try:
            capture.release()
            await websocket.close()
            cur.close()
            conn.close()
        except RuntimeError:
            print("WebSocket already closed")

# API: youtube websocket stream
@app.websocket('/ws/streamViolation/')
async def stream_violation(websocket: WebSocket):
    await websocket.accept()
    
    try:
        url = await websocket.receive_text()
        
        if not url:
            await websocket.send_text("Error: URL is null or empty")
            return
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'no_warnings': True,
            'quiet': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_url = info['url']
        except Exception as e:
            await websocket.send_text(f"Error: Failed to extract video info - {str(e)}")
            return
        
        capture = cv2.VideoCapture(video_url)
        detection = DetectViolation(app.state.travio_model)
        
        while capture.isOpened():
            ret, frame = capture.read()
            
            if not ret:
                print("Error: failed to read frame")
                break
            
            default_frame = cv2.resize(frame, (640, int(640 * (9/16))))
            processed_frame, traffic_light_status, traffic_light_violator_counter, wrong_way_violator_counter, helmet_violator_counter = detection.start_detect(default_frame.copy())

            _, default_buffer = cv2.imencode('.jpg', default_frame)
            _, processed_frame_buffer = cv2.imencode('.jpg', processed_frame)
            default_frame_bytes = default_buffer.tobytes()
            processed_frame_bytes = processed_frame_buffer.tobytes()
            
            result = {
                "traffic_light_status": traffic_light_status,
                "traffic_light_violator_counter": traffic_light_violator_counter,
                "wrong_way_violator_counter": wrong_way_violator_counter,
                "helmet_violator_counter": helmet_violator_counter
            }
            
            await websocket.send_bytes(default_frame_bytes)
            await websocket.send_bytes(processed_frame_bytes)
            await websocket.send_json(result)
            await asyncio.sleep(0.03)
            
    except Exception as e:
        print(f"Error occured: {e}")
        return {"status": "failed", "error": str(e)}
    
    finally:
        try:
            capture.release()
            await websocket.close()
        except RuntimeError:
            print("WebSocket already closed")