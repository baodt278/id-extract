import uvicorn

from src.controller.config import PORT

if __name__ == "__main__":
    uvicorn.run(
        "src:app", host="0.0.0.0", port=int(PORT), reload=True, debug=True
    )
