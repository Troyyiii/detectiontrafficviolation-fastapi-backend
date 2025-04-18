import sys
from pathlib import Path

# project file path
ROOT = Path(__file__).resolve().parent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

# model path
MODEL_PATH = ROOT / 'models'
LINE_MODEL_PATH = MODEL_PATH / 'line_test_best100.pt'
HELMET_MODEL_PATH = MODEL_PATH / 'helm_test_best50.pt'
CROSSWALK_MODEL_PATH = MODEL_PATH / 'crosswalk_best50.pt'
TRAVIO_MODEL_PATH = MODEL_PATH / 's_75_v2.onnx'

# upload path
UPLOAD_DIR = ROOT / 'uploads'