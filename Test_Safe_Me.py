import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
import time

# Function to convert file to image
def file_to_image(file):
    image = np.array(Image.open(file))
    return image

# Function to blur specific areas in an image
def blur_area(image, boxes):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    for (x, y, w, h) in boxes:
        region = pil_image.crop((x, y, x+w, y+h))
        region = region.filter(ImageFilter.GaussianBlur(15))
        pil_image.paste(region, (x, y))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Function to pixelate specific areas in an image
def pixelate_area(image, boxes, pixel_size=10):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for (x, y, w, h) in boxes:
        region = pil_image.crop((x, y, x+w, y+h))
        region = region.resize((pixel_size, pixel_size), Image.NEAREST)
        region = region.resize((w, h), Image.NEAREST)
        pil_image.paste(region, (x, y))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Function to redact specific areas in an image
def redact_area(image, boxes):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    for (x, y, w, h) in boxes:
        draw.rectangle((x, y, x+w, y+h), fill="black")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

logo = Image.open("icon.png")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(logo, use_column_width=True)

st.write("")

col1, col2, col3 = st.columns([2, 2, 1])
with col2:
    st.title("Safe Me")

start_button = st.button("Start")

if start_button:
    st.empty()
    st.title("Welcome to Safe Me!")

st.write("")
uploaded_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert file to image
    image = file_to_image(uploaded_file)
    
    # Dummy function to detect sensitive areas (replace with your implementation)
    def detect_sensitive_areas(image):
        # This is a dummy function; replace with your actual detection logic
        # Here, we return a single box for demonstration purposes
        return [(100, 100, 200, 200)]  # Example box coordinates (x, y, width, height)
    
    # Detect sensitive areas in the image (dummy implementation)
    sensitive_boxes = detect_sensitive_areas(image)
    
    # Apply blurring
    image_blur = blur_area(image.copy(), sensitive_boxes)
    
    # Apply pixelation
    image_pixelate = pixelate_area(image.copy(), sensitive_boxes, pixel_size=10)
    
    # Apply redaction
    image_redact = redact_area(image.copy(), sensitive_boxes)
    
    with st.spinner('Obscuring in Progress...'):
        time.sleep(2)

        # Display original image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("You are now protected. Please copy your hash key or share it... ")

        # Display blurred image
        st.image(image_blur, caption="Blurred Image", use_column_width=True)

        # Display pixelated image
        st.image(image_pixelate, caption="Pixelated Image", use_column_width=True)

        # Display redacted image
        st.image(image_redact, caption="Redacted Image", use_column_width=True)

        # Dummy hash key
        st.write("Your hash key: 755ef97e0d59d280f3e73c3bedff351e")

        # Add share buttons
        st.write("<h3>Share via:</h3>", unsafe_allow_html=True)

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
