import streamlit as st
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
# from mtcnn.mtcnn import MTCNN
from io import BytesIO


# Install Tesseract if not installed
if not os.path.exists('/usr/bin/tesseract'):
    st.error('Tesseract is not installed. Installing...')
    os.system('apt-get install tesseract-ocr')
    os.system('apt-get install libtesseract-dev')
    os.system('apt-get install tesseract-ocr-deu')
    st.success('Tesseract has been successfully installed.')


# Function to detect text in an image using Tesseract OCR
def detect_text(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Data Obscuring Functions
def blur_area(image, boxes):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    for (x, y, w, h) in boxes:
        region = pil_image.crop((x, y, x+w, y+h))
        region = region.filter(ImageFilter.GaussianBlur(15))
        pil_image.paste(region, (x, y))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def pixelate_area(image, boxes, pixel_size=10):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for (x, y, w, h) in boxes:
        region = pil_image.crop((x, y, x+w, y+h))
        region = region.resize((pixel_size, pixel_size), Image.NEAREST)
        region = region.resize((w, h), Image.NEAREST)
        pil_image.paste(region, (x, y))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def redact_area(image, boxes):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    for (x, y, w, h) in boxes:
        draw.rectangle((x, y, x+w, y+h), fill="black")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# Main function to process the image
def process_image(image, method):
    text_boxes = detect_text(image)
    face_boxes = detect_faces(image)
    
    # Combine all boxes (you can add more detection methods and their boxes here)
    all_boxes = text_boxes + face_boxes
    
    # Apply the selected obscuring method
    if method == 'Blur':
        processed_image = blur_area(image, all_boxes)
    elif method == 'Pixelate':
        processed_image = pixelate_area(image, all_boxes)
    elif method == 'Redact':
        processed_image = redact_area(image, all_boxes)
    
    return processed_image

# Streamlit UI
st.title('SAFE ME: Image Privacy Protection')
st.write('Upload an image and choose an obscuring method to protect sensitive information.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
method = st.selectbox('Select Obscuring Method', ('Blur', 'Pixelate', 'Redact'))

# Load the icon
icon = Image.open('icon.png')

# Display the icon
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(icon, use_column_width=True)

if uploaded_file is not None:
    # Read the uploaded image
    image = np.array(Image.open(uploaded_file))
    
    # Process the image
    processed_image = process_image(image, method)
    
    # Display the original and processed images
    st.image(image, caption='Original Image', use_column_width=True)
    st.image(processed_image, caption='Processed Image', use_column_width=True)
    
    # Provide download link for the processed image
    processed_pil_image = Image.fromarray(processed_image)
    buffer = BytesIO()
    processed_pil_image.save(buffer, format='JPEG')
    buffer.seek(0)
    st.download_button(label='Download Processed Image', data=buffer, file_name='processed_image.jpg', mime='image/jpeg')

    # Add share buttons
    st.markdown("<h3>Share via:</h3>", unsafe_allow_html=True)

    # Gmail icon and link
    st.markdown(
        """
        <a href="https://mail.google.com/mail/?view=cm&fs=1&to&su=Check%20out%20this%20image%20uploaded%20to%20Safe%20Me:%20{}" target="_blank"><img src="https://static.vecteezy.com/system/resources/previews/016/716/465/large_2x/gmail-icon-free-png.png" alt="Gmail" width="48"/></a>
        """.format(uploaded_file.name),
        unsafe_allow_html=True
    )

    # Telegram icon and link
    st.markdown(
        """
        <a href="https://t.me/share/url?url=&text=Check%20out%20this%20image%20uploaded%20to%20Safe%20Me:%20{}" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Telegram_logo.svg/2048px-Telegram_logo.svg.png" alt="Telegram" width="48"/></a>
        """.format(uploaded_file.name),
        unsafe_allow_html=True
    )
