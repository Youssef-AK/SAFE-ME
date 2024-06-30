import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2
import pytesseract

# Set the path for Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Text Detection using Tesseract OCR
def detect_text(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT, config=custom_config)
    n_boxes = len(d['level'])
    boxes = []
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:  # confidence threshold
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            boxes.append((x, y, w, h))
    return boxes

# Face Detection using OpenCV
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Data Obscuring Functions
def blur_area(image, boxes):
    draw = ImageDraw.Draw(image)
    for (x, y, w, h) in boxes:
        region = image.crop((x, y, x+w, y+h))
        region = region.filter(ImageFilter.GaussianBlur(15))
        image.paste(region, (x, y))
    return image

def pixelate_area(image, boxes, pixel_size=10):
    for (x, y, w, h) in boxes:
        region = image.crop((x, y, x+w, y+h))
        region = region.resize((pixel_size, pixel_size), Image.NEAREST)
        region = region.resize((w, h), Image.NEAREST)
        image.paste(region, (x, y))
    return image

def redact_area(image, boxes):
    draw = ImageDraw.Draw(image)
    for (x, y, w, h) in boxes:
        draw.rectangle((x, y, x+w, y+h), fill="black")
    return image

# Streamlit application
st.title('SAFE ME Project')
st.write('Upload an image to detect and obscure sensitive information.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('')

    text_boxes = detect_text(image)
    face_boxes = detect_faces(image)

    all_boxes = text_boxes + list(face_boxes)

    blur = blur_area(image.copy(), all_boxes)
    pixelate = pixelate_area(image.copy(), all_boxes)
    redact = redact_area(image.copy(), all_boxes)

    st.write('Blurred Image')
    st.image(blur, use_column_width=True)

    st.write('Pixelated Image')
    st.image(pixelate, use_column_width=True)

    st.write('Redacted Image')
    st.image(redact, use_column_width=True)
