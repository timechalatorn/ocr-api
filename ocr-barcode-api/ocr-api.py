import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import base64
import uvicorn
import json

from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize FastAPI app
app = FastAPI()

# Allow CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to extract the area defined by a drawn polygon
def extract_area(image, points):
    rect = cv2.boundingRect(cv2.convexHull(np.array(points)))
    x, y, w, h = rect
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image, rect

# Function to convert an image to base64
def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

# FastAPI endpoint to upload an image and process OCR
@app.post("/ocr/process/")
async def process_image(file: UploadFile = File(...), points: str = Form(...)):
    # Read the uploaded image
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # image = cv2.resize(image, (780, 540), interpolation=cv2.INTER_LINEAR)      tempolary commend
    clone = image.copy()

    # Parse points from the input
    try:
        points_list = json.loads(points)
        points = [(int(x), int(y)) for x, y in points_list]
    except Exception as e:
        return JSONResponse(content={"error": "Invalid points format", "details": str(e)}, status_code=400)

    # Ensure points are within image bounds
    height, width = image.shape[:2]
    for (x, y) in points:
        if not (0 <= x < width and 0 <= y < height):
            return JSONResponse(content={"error": f"Point ({x}, {y}) is out of image bounds"}, status_code=400)

    # Extract the specific area from the image
    cropped_image, rect = extract_area(clone, points)

    # Check if the cropped image is empty
    if cropped_image.size == 0:
        return JSONResponse(content={"error": "The cropped area is empty. Please check the coordinates."}, status_code=400)

    # Resize the cropped image for better OCR accuracy
    resized_image = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Perform OCR on the resized image
    result = ocr.ocr(resized_image, cls=True)

    # Initialize detected_text as an empty string
    detected_text = ""

    # Extract detected text from the OCR result
    if result and result[0]:
        detected_text = ''.join([line[1][0] for line in result[0]])

    # Draw the bounding box on the image
    for point in points:
        cv2.circle(clone, point, 5, (0, 255, 0), -1)
    cv2.polylines(clone, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the OCR result on the image with "Detected text: xxxxxxx"
    def draw_text(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2, text_color=(0, 0, 255), text_color_bg=(0, 0, 0)):
        x, y = pos
        for i, line in enumerate(text.split('\n')):
            (w, h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            cv2.rectangle(img, (x, y - h - 10), (x + w, y + 5), text_color_bg, -1)
            cv2.putText(img, line, (x, y), font, font_scale/2, text_color, font_thickness)
            y += h + 10

    wrapped_text = '\n'.join(detected_text[i:i+40] for i in range(0, len(detected_text), 40))
    draw_text(clone, f"Detected text: {wrapped_text}", (10, 30))

    # Convert the final image to base64
    result_base64 = image_to_base64(clone)

    return JSONResponse(content={"result": wrapped_text})

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
