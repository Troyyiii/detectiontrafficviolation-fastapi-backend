import torch
import pathlib

def load_model(model_path):
    pathlib.PosixPath = pathlib.WindowsPath
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load trained model YOLOv5
        model = torch.hub.load("./yolov5", "custom", path=model_path, source="local", force_reload=True)
        model.to(device)

        print("✅ Model Successfully loaded!")
        print(f"🔍 Model running in: {device}")

        return model

    except Exception as e:
        print(f"❌ Failed load model: {e}")
        return None