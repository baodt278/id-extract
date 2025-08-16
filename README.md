# ID Extract

A Vietnamese ID card information extraction system using computer vision and optical character recognition (OCR). This project automatically detects, aligns, and extracts text information from Vietnamese identification cards using YOLOv5 for object detection and VietOCR for text recognition.

## Features

- **Corner Detection**: Automatically detects the four corners of ID cards for perspective correction
- **Content Detection**: Identifies and localizes text fields within the ID card
- **Face Detection**: Locates face regions on the ID card
- **OCR Processing**: Extracts Vietnamese text from detected regions using VietOCR
- **Image Alignment**: Performs perspective transformation to correct card orientation
- **RESTful API**: FastAPI-based web service for easy integration

## Architecture

The system uses a multi-stage pipeline:

1. **Corner Detection**: YOLOv5 model detects the four corners of the ID card
2. **Perspective Correction**: Four-point transformation aligns the card image
3. **Content Detection**: Another YOLOv5 model identifies text regions within the aligned card
4. **OCR Recognition**: VietOCR processes each detected text region
5. **Post-processing**: Combines and formats the extracted information

## Project Structure

```
baodt278-id-extract/
├── requirements.txt          # Python dependencies
├── run.py                   # Application entry point
└── src/
    ├── __init__.py         # FastAPI app initialization
    ├── controller/
    │   ├── config.py       # Configuration parameters
    │   ├── main.py         # API endpoints and main logic
    │   └── utils.py        # Utility functions
    └── service/
        └── orc/
            ├── config/
            │   └── seq2seq_config.yml  # OCR model configuration
            └── weights/
                ├── content.pt          # Content detection model
                ├── corner.pt          # Corner detection model
                └── face.pt            # Face detection model
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/baodt278/id-extract.git
   cd id-extract
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights**:
   - Ensure the pre-trained model weights are placed in `src/service/orc/weights/`
   - The models include: `corner.pt`, `content.pt`, `face.pt`

## Configuration

Edit `src/controller/config.py` to adjust:

- **Model Paths**: Locations of the YOLOv5 and OCR model files
- **Thresholds**: Confidence and IoU thresholds for object detection
- **Device**: CPU or CUDA device for inference
- **Directories**: Upload and result folder paths

Key configuration parameters:
```python
PORT = 8082
CONF_CONTENT_THRESHOLD = 0.7
IOU_CONTENT_THRESHOLD = 0.7
DEVICE = "cpu"  # or "cuda:0" for GPU
```

## Usage

### Starting the Server

```bash
python run.py
```

The API server will start on `http://localhost:8082`

### API Endpoints

#### 1. Upload and Extract (Combined)
```bash
POST /uploader
```
Upload an ID card image and automatically extract information.

**Example using curl**:
```bash
curl -X POST "http://localhost:8082/uploader" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/id_card.jpg"
```

#### 2. Extract Only
```bash
POST /extract
```
Process a previously uploaded image.

### Response Format

Successful extraction returns:
```json
{
  "data": [
    "Field 1 text",
    "Field 2 text",
    "Field 3 text",
    // ... more extracted fields
  ]
}
```

Error responses include specific error codes:
- `401`: Corner detection failed
- `402`: Content detection failed (missing fields)
- `403`: No file selected
- `404`: Unsupported file format

## Model Details

### YOLOv5 Models
- **Corner Detection**: Identifies the four corners of ID cards
- **Content Detection**: Locates text fields and regions of interest
- **Face Detection**: Detects face regions (if needed)

### VietOCR Configuration
- **Model**: seq2seq architecture with VGG19 backbone
- **Vocabulary**: Extended Vietnamese character set including diacritics
- **Image Processing**: Resizes to 32px height for optimal recognition

## Dependencies

Key dependencies include:
- **FastAPI**: Web framework for the API
- **YOLOv5**: Object detection models
- **VietOCR**: Vietnamese text recognition
- **OpenCV**: Image processing
- **Pillow**: Image manipulation
- **NumPy**: Numerical computations
- **Shapely**: Geometric operations

## Error Handling

The system includes comprehensive error handling for:
- Missing or invalid files
- Failed corner detection
- Insufficient content detection
- OCR processing errors

## Performance Considerations

- **GPU Acceleration**: Set `DEVICE = "cuda:0"` for faster processing
- **Batch Processing**: The system processes one image at a time
- **Memory Management**: Automatically cleans up temporary files
- **Threshold Tuning**: Adjust confidence thresholds based on your dataset

## Development

### Adding New Features
1. Extend the detection pipeline in `main.py`
2. Add new model configurations in `config.py`
3. Implement utility functions in `utils.py`

### Model Training
- Corner and content detection models are YOLOv5 based
- OCR model uses VietOCR's seq2seq architecture
- Configuration files are provided for model customization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

Please check the repository for license information.

## Acknowledgments

- **YOLOv5**: For object detection capabilities
- **VietOCR**: For Vietnamese text recognition
- **FastAPI**: For the web framework
- **OpenCV**: For image processing utilities
