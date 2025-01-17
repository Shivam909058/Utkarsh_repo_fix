import cv2
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import easyocr
import numpy as np
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


PLATE_DIR = "number_plates"
if not os.path.exists(PLATE_DIR):
    os.makedirs(PLATE_DIR)

harcascade = "model/haarcascade_russian_plate_number.xml"


reader = easyocr.Reader(['en'], gpu=False, detect_network="craft")
plate_cascade = cv2.CascadeClassifier(harcascade)
if plate_cascade.empty():
    raise Exception("Error: Could not load cascade classifier")

def enhance_image(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    return gray

async def process_image(image_data):
    """Process the uploaded image and detect plates."""
    try:
        # Convert image to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image")
            return None, "Failed to decode image"

        # Resize if image is too large
        max_dimension = 1920
        height, width = img.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # Convert to grayscale and enhance
        gray = enhance_image(img)

        
        plates = []
        scale_factors = [1.1, 1.15, 1.2]
        min_neighbors_range = [3, 4, 5]
        
        for scale in scale_factors:
            for min_neighbors in min_neighbors_range:
                detected = plate_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale,
                    minNeighbors=min_neighbors,
                    minSize=(60, 20),
                    maxSize=(300, 100)
                )
                if len(detected) > 0:
                    plates = detected
                    break
            if len(plates) > 0:
                break

        if len(plates) == 0:
            logger.info("No plate detected in image")
            return None, "No plate detected"

        
        x, y, w, h = plates[0]
        
        
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(w + 2*padding, img.shape[1] - x)
        h = min(h + 2*padding, img.shape[0] - y)
        
        plate_img = img[y:y+h, x:x+w]

        
        plate_img = cv2.resize(plate_img, None, fx=2, fy=2)  # Upscale
        plate_gray = enhance_image(plate_img)
        
        
        filename = f"plate_{len(os.listdir(PLATE_DIR))}.jpg"
        filepath = os.path.join(PLATE_DIR, filename)
        cv2.imwrite(filepath, plate_img)  # Save original
        
        
        try:
            results = reader.readtext(
                plate_gray,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                batch_size=1,
                contrast_ths=0.3,
                adjust_contrast=0.5,
                width_ths=0.5,
                height_ths=0.5
            )
            
            if results:
                extracted_text = " ".join([text[1] for text in results])
                logger.info(f"OCR Result: {extracted_text}")
            else:
                extracted_text = "No text detected"
                logger.info("OCR found no text")
                
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            extracted_text = "OCR Failed"

        return filename, extracted_text

    except Exception as e:
        logger.error(f"Image Processing Error: {e}")
        return None, f"Error: {str(e)}"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/capture")
async def capture(file: UploadFile = File(...)):
    
    try:
        logger.info("Received image upload")
        contents = await file.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
            
        filename, extracted_text = await process_image(contents)
        
        if filename:
            return {"filename": filename, "text": extracted_text}
        else:
            raise HTTPException(status_code=400, detail=extracted_text)
            
    except Exception as e:
        logger.error(f"Capture Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/show/{filename}")
async def show_image(filename: str):
    file_path = os.path.join(PLATE_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Image not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
