import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import string
from io import BytesIO
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as ReportLabImage
)
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

# Define constants
alphabet = string.digits + string.ascii_lowercase + '.'
blank_index = len(alphabet)
models_folder = 'model'
sample_pairs_folder = 'sample_pairs'

# Model paths
meter_classifier_model_path = os.path.join(models_folder, 'meter_classifier.pth')
yolo_model_path = os.path.join(models_folder, 'yolo-screen-obb-grayscale.pt')
ocr_model_path = os.path.join(models_folder, 'model_float16.tflite')
screen_quality_classifier_model_path = os.path.join(models_folder, 'best_meter_clf.pt')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
@st.cache_resource
def load_meter_classifier(model_path):
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path).to(device)

@st.cache_resource
def load_ocr_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_screen_quality_classifier(model_path):
    return YOLO(model_path)

# Transformations
common_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

screen_clf_transform = transforms.Compose([
    transforms.Resize((288, 896)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# OCR preparation
def prepare_input(image_path):
    input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_data is None:
        raise ValueError(f"Failed to read image '{image_path}' for OCR.")
    input_data = cv2.resize(input_data, (200, 31))
    input_data = input_data[np.newaxis, ..., np.newaxis].astype('float32') / 255
    return input_data

def predict_ocr(image_path, interpreter):
    input_data = prepare_input(image_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

# Image processing
def crop_image(image, box):
    pts = order_points(box.astype(np.float32))
    widthA, widthB = np.linalg.norm(pts[2] - pts[3]), np.linalg.norm(pts[1] - pts[0])
    maxWidth = max(int(widthA), int(widthB))
    heightA, heightB = np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[0] - pts[3])
    maxHeight = max(int(heightA), int(heightB))
    if maxWidth <= 0 or maxHeight <= 0:
        raise ValueError("Invalid dimensions for cropping.")
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    if warped.shape[1] < warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY), (200, 31))

def perform_yolo_detection(image, yolo_model):
    grayscale_image = cv2.merge([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)] * 3)
    results = yolo_model(grayscale_image)
    for result in results:
        boxes, confs = result.obb.xyxyxyxy, result.obb.conf
        if len(boxes):
            max_conf_idx = np.argmax(confs.cpu().numpy())
            return boxes[max_conf_idx].cpu().numpy().reshape((4, 2)), confs[max_conf_idx].cpu().item()
    return None, None

# Generate PDF
def generate_pdf(prev_reading, curr_reading, usage, bill_amount, images, cropped_images):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    center_style = ParagraphStyle(name='center', alignment=TA_CENTER, fontSize=18)
    normal_text = ParagraphStyle(name='normal_text', fontSize=12)

    # Add title
    elements.append(Paragraph("Electricity Bill", center_style))
    elements.append(Spacer(1, 24))

    # Display readings and bill details
    elements.append(Paragraph(f"Previous Reading: {prev_reading} kWh", normal_text))
    elements.append(Paragraph(f"Current Reading: {curr_reading} kWh", normal_text))
    elements.append(Paragraph(f"Units Consumed: {usage} kWh", normal_text))
    elements.append(Paragraph(f"Total Bill Amount: Rs. {bill_amount:.2f}", normal_text))

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# Streamlit app
def main():
    st.title("Meter OCR App")
    meter_classifier = load_meter_classifier(meter_classifier_model_path)
    yolo_model = load_yolo_model(yolo_model_path)
    ocr_interpreter = load_ocr_interpreter(ocr_model_path)
    screen_quality_classifier = load_screen_quality_classifier(screen_quality_classifier_model_path)

    uploaded_prev = st.file_uploader("Upload Previous Month's Image", type=["png", "jpg", "jpeg"])
    uploaded_curr = st.file_uploader("Upload Current Month's Image", type=["png", "jpg", "jpeg"])

    if uploaded_prev and uploaded_curr:
        prev_image = Image.open(uploaded_prev).convert('RGB')
        curr_image = Image.open(uploaded_curr).convert('RGB')

        # Process image pair
        # Add logic to handle classification, detection, OCR, and PDF generation here

if __name__ == '__main__':
    main()
