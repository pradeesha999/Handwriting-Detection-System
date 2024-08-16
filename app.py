from flask import Flask, request, render_template, send_file, jsonify, Response
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import pytesseract
import easyocr
import threading
import time
import base64
import openai

app = Flask(__name__, static_folder='static')

openai.api_key = 'API-KEY'  # Replace with your OpenAI API key

def enhance_image(image):
    # Straighten the image
    straight_image = straighten_image(image)

    # Denoise the image
    denoised_image = cv2.fastNlMeansDenoisingColored(straight_image, None, 5, 5, 7, 21)

    # Upscale the image
    scale_factor = 2
    width = int(denoised_image.shape[1] * scale_factor)
    height = int(denoised_image.shape[0] * scale_factor)
    upscaled_image = cv2.resize(denoised_image, (width, height), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    block_size = 19  # Use an odd number greater than 1
    C_value = 3
    thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C_value)

    return upscaled_image, thresh_image

def straighten_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        angles.append(angle)

    median_angle = np.median(angles)

    def rotate_image(image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    angle_degrees = median_angle * (180 / np.pi)
    corrected_image = rotate_image(image, angle_degrees)
    return corrected_image

def perform_easyocr(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    text = "\n".join([text for (_, text, _) in result])
    return text

def perform_tesseract(image):
    text = pytesseract.image_to_string(image)
    return text

def correct_text_with_gpt(text1, text2):
    combined_text = f"EasyOCR Output:\n{text1}\n\nTesseract Output:\n{text2}\n\n"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that corrects OCR text."},
            {"role": "user", "content": f"Combine and correct the following OCR texts:\n\n{combined_text}\n\nCorrected Text:"}
        ],
        max_tokens=2000  # Adjust if needed
    )
    corrected_text = response.choices[0].message['content'].strip()
    return corrected_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400
    
    print("File uploaded:", file.filename)  # Debugging message
    
    # Read the image file
    in_memory_file = BytesIO()
    file.save(in_memory_file)
    in_memory_file.seek(0)
    image = np.array(Image.open(in_memory_file))

    # Enhance the image
    upscaled_image, thresh_image = enhance_image(image)

    # Save the original image in-memory
    original_image_io = BytesIO()
    Image.fromarray(image).save(original_image_io, format='PNG')
    original_image_io.seek(0)
    original_image_base64 = base64.b64encode(original_image_io.getvalue()).decode('utf-8')

    # Save the enhanced image in-memory
    enhanced_image_io = BytesIO()
    Image.fromarray(thresh_image).save(enhanced_image_io, format='PNG')
    enhanced_image_io.seek(0)
    enhanced_image_base64 = base64.b64encode(enhanced_image_io.getvalue()).decode('utf-8')

    # Store the enhanced image in a global variable to be accessed by the OCR endpoint
    global enhanced_image
    enhanced_image = thresh_image

    return jsonify({
        "original_image": original_image_base64,
        "enhanced_image": enhanced_image_base64
    })

@app.route('/ocr-progress', methods=['GET'])
def ocr_progress():
    def generate():
        progress = 0
        while progress < 100:
            time.sleep(0.5)  # Simulate time delay
            progress += 10
            yield f"data: {progress}\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/ocr', methods=['POST'])
def ocr_image():
    if enhanced_image is None:
        return jsonify({"error": "No enhanced image available"}), 400
    
    try:
        # Perform OCR using both EasyOCR and Tesseract
        easyocr_text = perform_easyocr(enhanced_image)
        tesseract_text = perform_tesseract(enhanced_image)

        print(f"EasyOCR Text: {easyocr_text}")  # Debugging message
        print(f"Tesseract Text: {tesseract_text}")  # Debugging message

        # Correct the detected text using GPT
        corrected_text = correct_text_with_gpt(easyocr_text, tesseract_text)

        print(f"Corrected Text: {corrected_text}")  # Debugging message

        # Annotate the image with the detected text
        annotated_image = enhanced_image.copy()
        boxes = pytesseract.image_to_boxes(annotated_image)
        for b in boxes.splitlines():
            b = b.split(' ')
            annotated_image = cv2.rectangle(annotated_image, (int(b[1]), int(b[2])), (int(b[3]), int(b[4])), (0, 0, 255), 2)
            annotated_image = cv2.putText(annotated_image, b[0], (int(b[1]), int(b[2]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Save the annotated image in-memory
        annotated_image_io = BytesIO()
        Image.fromarray(annotated_image).save(annotated_image_io, format='PNG')
        annotated_image_io.seek(0)

        # Encode the annotated image as base64
        annotated_image_base64 = base64.b64encode(annotated_image_io.getvalue()).decode('utf-8')

        return jsonify({
            "easyocr_text": easyocr_text,
            "tesseract_text": tesseract_text,
            "corrected_text": corrected_text,
            "annotated_image": annotated_image_base64
        })

    except Exception as e:
        print(f"Error during OCR process: {e}")
        return jsonify({"error": "Error occurred during OCR process"}), 500

if __name__ == '__main__':
    enhanced_image = None  # Global variable to store the enhanced image
    app.run(debug=True)
