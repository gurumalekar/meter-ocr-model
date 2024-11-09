import os
import shutil
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import string
import pandas as pd
from io import BytesIO
import streamlit as st  # type: ignore

# Import ReportLab for PDF generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as ReportLabImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_CENTER

# Define constants
alphabet = string.digits + string.ascii_lowercase + '.'
blank_index = len(alphabet)

# Define paths
models_folder = 'models'  # Ensure this folder contains the required models

# Model paths
meter_classifier_model_path = os.path.join(models_folder, 'meter_classifier.pth')
yolo_model_path = os.path.join(models_folder, 'yolo-screen-obb-grayscale.pt')
ocr_model_path = os.path.join(models_folder, 'model_float16.tflite')
screen_quality_classifier_model_path = os.path.join(models_folder, 'screen_classifier_grayscale_cnn.pth')  # New classifier

# CNN
class CNNBinaryClassifier(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 72 x 224

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 36 x 112

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Adjust if necessary
        )

        # Adjust input size if necessary
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 36 * 112, 512),  # Input size: 128 * 36 * 112 = 516096
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Output layer for 2 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the meter classifier model
@st.cache_resource
def load_meter_classifier(model_path):
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)  # Assuming 2 classes: 'meter' and 'no_meter'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Load the YOLO model
@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path).to(device)
    return model

# Load the OCR interpreter
@st.cache_resource
def load_ocr_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load the screen quality classifier model
@st.cache_resource
def load_screen_quality_classifier(model_path):
    model = CNNBinaryClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Define transformations for the classifiers
common_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize([0.485, 0.456, 0.406],   # Normalize with ImageNet mean and std
                         [0.229, 0.224, 0.225])
])
screen_clf_transform = transforms.Compose([
    transforms.Resize((288, 896)),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat channel to create 3 channels
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize for 3-channel input
])

def prepare_input(image_path):
    """
    Prepares the image for OCR by resizing and normalizing.
    """
    input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_data is None:
        raise ValueError(f"Failed to read image '{image_path}' for OCR.")
    input_data = cv2.resize(input_data, (200, 31))
    input_data = input_data[np.newaxis]
    input_data = np.expand_dims(input_data, 3)
    input_data = input_data.astype('float32') / 255
    return input_data

def predict_ocr(image_path, interpreter):
    """
    Performs OCR on the provided image using the TensorFlow Lite model.
    """
    input_data = prepare_input(image_path)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def order_points(pts):
    """
    Orders points in the following order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of points (x + y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    # Difference of points (x - y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

def perform_classification(image, model, transform, class_names):
    """
    Classifies the uploaded image using the provided model.
    """
    # Apply transformations
    image_transformed = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_transformed)
        _, prediction = torch.max(outputs, 1)
        predicted_class = class_names[prediction.item()]
    return predicted_class

def perform_yolo_detection(image, yolo_model, device='cpu'):
    """
    Performs YOLO detection on the image (converted to grayscale) and returns the best bounding box.
    """
    # Convert the image to grayscale while preserving the original shape
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Duplicate the grayscale image across three channels to match YOLO's expected input shape
    grayscale_image = cv2.merge([grayscale_image] * 3)

    # Run YOLO model
    results = yolo_model(grayscale_image)

    # Process the results
    for result in results:
        boxes = result.obb.xyxyxyxy  # Oriented bounding boxes
        confs = result.obb.conf

        if len(boxes) == 0:
            return None, None

        # Select the box with the highest confidence
        max_conf_idx = np.argmax(confs.cpu().numpy())
        box = boxes[max_conf_idx].cpu().numpy().reshape((4, 2))
        confidence = confs[max_conf_idx].cpu().item()
        return box, confidence

def preprocess(img):
    grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(grayimg, (200, 31))
    return resized

def crop_image(image, box):
    """
    Crops the image using the provided bounding box with perspective transform.
    """
    pts = box.astype(np.float32)
    pts = order_points(pts)

    # Compute the width of the new image
    widthA = np.linalg.norm(pts[2] - pts[3])
    widthB = np.linalg.norm(pts[1] - pts[0])
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.linalg.norm(pts[1] - pts[2])
    heightB = np.linalg.norm(pts[0] - pts[3])
    maxHeight = max(int(heightA), int(heightB))

    # Ensure dimensions are valid
    if maxWidth <= 0 or maxHeight <= 0:
        raise ValueError("Invalid dimensions for cropping.")

    # Destination points for perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, dst)

    # Warp the image
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Rotate the image if it's taller than it is wide
    if warped.shape[1] < warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return preprocess(warped)

def perform_ocr(cropped_image, ocr_interpreter):
    """
    Performs OCR on the cropped image and returns the extracted text.
    """
    # Save the cropped image to a temporary buffer
    _, buffer = cv2.imencode('.png', cropped_image)
    cropped_bytes = BytesIO(buffer).read()

    # Save to a temporary file
    temp_path = 'temp_cropped.png'
    with open(temp_path, 'wb') as f:
        f.write(cropped_bytes)

    try:
        ocr_output = predict_ocr(temp_path, ocr_interpreter)
        # Decode the OCR output
        text = "".join([alphabet[int(index)] for index in ocr_output[0] if int(index) not in [blank_index, -1]])
        # Post-processing (if any)
        modified_text = ""
        for char in text:
            if char == "a":
                modified_text += "."
            else:
                modified_text += char

    except Exception as e:
        st.error(f"OCR failed: {e}")
        text = ""
        modified_text = ""
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return modified_text

def draw_bounding_box(image, box, color=(255, 255, 0), thickness=80):
    """
    Draws the bounding box on the image.
    """
    box = box.astype(int)
    for i in range(4):
        pt1 = tuple(box[i])
        pt2 = tuple(box[(i + 1) % 4])
        cv2.line(image, pt1, pt2, color, thickness)
    image = cv2.resize(image, (224, 224))
    return image

def calculate_bill(units_consumed):
    """
    Calculates the total bill based on the units consumed and the tariff plan.
    """
    fixed_charge = 100  # Rs.
    total_amount = fixed_charge
    units_remaining = units_consumed
    slabs = [
        (30, 1.90),   # First 30 units
        (45, 3.00),   # Next 45 units (units 31-75)
        (50, 4.50),   # Next 50 units (units 76-125)
        (100, 6.00),  # Next 100 units (units 126-225)
        (25, 8.75)    # Last 25 units (units 226-250)
    ]

    for slab_units, rate in slabs:
        if units_remaining > 0:
            units_in_slab = min(units_remaining, slab_units)
            total_amount += units_in_slab * rate
            units_remaining -= units_in_slab
        else:
            break

    if units_remaining > 0:
        # Assuming the rate for units beyond 250 is same as the last slab rate
        rate = slabs[-1][1]
        total_amount += units_remaining * rate

    return total_amount

def generate_pdf(readings, usage, bill_amount, images, cropped_images):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    styleN = styles['Normal']
    styleH = styles['Heading1']
    center_style = ParagraphStyle(name='center', alignment=TA_CENTER, fontSize=18, fontName="Courier-Oblique")
    center_font = ParagraphStyle(name='center', alignment=TA_CENTER, fontName='Helvetica')
    normal_text = ParagraphStyle(
    name='normal_text',
    alignment=0,
    fontName='Times-Roman',
    fontSize=18
    )
    normal_text_2 = ParagraphStyle(
    name='normal_text',
    alignment=0,
    fontName='Times-Roman',
    fontSize=12
    )
    # Add company logo
    logo_path = 'company_logo.png'  # Adjust as needed
    if os.path.exists(logo_path):
        logo = ReportLabImage(logo_path, width=1*inch, height=1*inch)
        elements.append(logo)
        elements.append(Spacer(1, 12))

    # Add title
    elements.append(Paragraph("Dummy Electricity Bill", center_font))
    elements.append(Spacer(1, 12))

    # Add images and readings
    for idx, (reading, image, cropped_image) in enumerate(zip(readings, images, cropped_images), start=1):
        elements.append(Paragraph(f"Image {idx}", normal_text))
        elements.append(Spacer(1, 12))

        # # Save PIL image to BytesIO
        # image_buffer = BytesIO()
        # image.save(image_buffer, format='PNG')
        # image_buffer.seek(0)
        # reportlab_image = ReportLabImage(image_buffer, width=4*inch, height=4*inch)
        # elements.append(reportlab_image)
        # elements.append(Spacer(1, 12))

        # Save cropped image to BytesIO
        cropped_image_pil = Image.fromarray(cropped_image)
        cropped_image_buffer = BytesIO()
        cropped_image_pil.save(cropped_image_buffer, format='PNG')
        cropped_image_buffer.seek(0)
        reportlab_cropped_image = ReportLabImage(cropped_image_buffer, width=3*inch, height=1*inch)
        elements.append(reportlab_cropped_image)
        elements.append(Spacer(1, 12))

        elements.append(Paragraph(f"Reading: {reading} kWh", normal_text_2))
        elements.append(Spacer(1, 12))

    # Add calculations
    elements.append(Paragraph("Calculations:", styleH))
    elements.append(Spacer(1, 12))
    # elements.append(Paragraph(f"Reading from Image 1: {readings[0]}\nReading from Image 2: {readings[1]}", normal_text_2))
    # elements.append(Paragraph(f"", styleN))
    elements.append(Paragraph(f"Units Consumed = {readings[0]} - {readings[1]} = {usage}", center_style))
    elements.append(Spacer(1, 12))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Total Bill Amount: Rs. {bill_amount:.2f}", center_style))

    # Build PDF
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# Streamlit App
def main():
    st.set_page_config(page_title="Meter OCR App", layout="wide")

    # Create two columns for the header
    col1, col2 = st.columns([3, 8])
    with col2:
        st.write("#  HackAP Hackathon: Power Distribution")
        st.write("## Problem Statement 5: Electric Meter OCR Application")
    with col1:
        # Placeholder for company logo (Adjust the path to your logo file)
        st.image('company_logo.png', width=200)  # Replace 'company_logo.png' with your logo file

    # Load models
    with st.spinner("Loading models..."):
        meter_classifier = load_meter_classifier(meter_classifier_model_path)
        yolo_model = load_yolo_model(yolo_model_path)
        ocr_interpreter = load_ocr_interpreter(ocr_model_path)
        screen_quality_classifier = load_screen_quality_classifier(screen_quality_classifier_model_path)
    # st.success("Models loaded successfully!")  

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload two images",
        type=["png", "jpg", "jpeg", "bmp", "gif"],
        accept_multiple_files=True
    )

    if uploaded_files is not None and len(uploaded_files) == 2:
        ocr_results = []
        readings = []
        cropped_images = []
        images_np = []
        images_pil = []

        # Create three columns for images and billing calculations
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2]

        with col1:
            st.header("Image 1")
            image1 = Image.open(uploaded_files[0]).convert('RGB')
            st.image(image1, caption='Uploaded Image 1', use_column_width=False, width=320)
            images_np.append(np.array(image1))
            images_pil.append(image1)
        with col2:
            st.header("Image 2")
            image2 = Image.open(uploaded_files[1]).convert('RGB')
            st.image(image2, caption='Uploaded Image 2', use_column_width=False, width=320)
            images_np.append(np.array(image2))
            images_pil.append(image2)

        # Process both images
        for idx, (image_np, image) in enumerate(zip(images_np, images_pil), start=1):
            with cols[idx-1]:
                try:
                    box, confidence = perform_yolo_detection(image_np, yolo_model)

                    if box is not None:
                        # Draw bounding box on the image
                        image_with_box = image_np.copy()
                        image_with_box = draw_bounding_box(image_with_box, box)

                        # Crop the detected area
                        cropped_image = crop_image(image_np, box)
                        # Collect cropped images for further use if needed
                        cropped_images.append(cropped_image)

                        # Display cropped image
                        st.write(f"Meter Screen in Image {idx}:")
                        st.image(cropped_image, caption=f'Meter Screen {idx}', width=320)

                        # Perform Screen Quality Classification
                        # Convert cropped_image to PIL Image for consistency
                        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                        class_names_quality = ['ng', 'ok']  # Ensure correct order
                        predicted_class_quality = perform_classification(
                            cropped_pil,
                            screen_quality_classifier,
                            screen_clf_transform,
                            class_names_quality
                        )
                        if predicted_class_quality == 'ok':
                            st.write(f"Screen Quality of Image {idx} is okay.")
                        else:
                            st.write(f"Screen Quality of Image {idx} might not be okay.")
                        if predicted_class_quality == 'ok':
                            # Perform OCR
                            ocr_result = perform_ocr(cropped_image, ocr_interpreter)
                            st.write(f"#### Reading in Image {idx}: {ocr_result}")
                            ocr_results.append(ocr_result)
                            readings.append(float(ocr_result))
                        else:
                            st.warning(
                                f"The cropped meter screen in Image {idx} is not good. OCR will not be performed."
                            )
                            ocr_results.append(None)
                            readings.append(None)
                    else:
                        st.warning(f"Meter screen unclear/not detected in Image {idx}.")
                        ocr_results.append(None)
                        readings.append(None)
                except Exception as e:
                    st.error(f"An error occurred during processing Image {idx}: {e}")
                    ocr_results.append(None)
                    readings.append(None)

        # After processing both images, perform calculations
        if len(ocr_results) == 2:
            if readings[0] is not None and readings[1] is not None:
                try:
                    # Perform calculation (units consumed)
                    usage = readings[0] - readings[1]

                    # Display calculations in the third column
                    with col3:
                        st.write("## Bill Calculations:")
                        st.write(f"Reading from Image 1: **{readings[0]}**")
                        st.write(f"Reading from Image 2: **{readings[1]}**")
                        st.write(f"Units Consumed = {readings[0]} - {readings[1]} = **{usage}**")
                        st.write(f"Fixed Charges = Rs. 100/-")

                        # Calculate the bill amount
                        bill_amount = calculate_bill(usage)
                        st.write(f"Total Bill Amount = Rs. **{bill_amount:.2f}**")

                        # Generate PDF
                        pdf_data = generate_pdf(readings, usage, bill_amount, images_pil, cropped_images)

                        # Provide download button
                        st.download_button(
                            label="Download Bill as PDF",
                            data=pdf_data,
                            file_name="meter_bill.pdf",
                            mime='application/pdf'
                        )

                except ValueError:
                    st.error("OCR results could not be parsed into numbers for calculation.")
            else:
                st.warning("Could not perform calculations because one or both OCR readings are missing.")
    else:
        st.info("Please upload exactly two images.")

if __name__ == '__main__':
    main()
