PORT = 8082
CONF_CONTENT_THRESHOLD = 0.7
IOU_CONTENT_THRESHOLD = 0.7

# CONF_CORNER_THRESHOLD = 0.8
# IOU_CORNER_THRESHOLD = 0.5

CORNER_MODEL_PATH = "src/service/orc/weights/corner.pt"
CONTENT_MODEL_PATH = "src/service/orc/weights/content.pt"
FACE_MODEL_PATH = "src/service/orc/weights/face.pt"
OCR_MODEL_PATH = "src/service/orc/weights/seq2seq.pth"
OCR_CFG = 'src/service/orc/config/seq2seq_config.yml'
DEVICE = "cpu"  # or "cuda:0" if using GPU
# Config directory
UPLOAD_FOLDER = "src/service/uploads"
SAVE_DIR = "src/service/results"

