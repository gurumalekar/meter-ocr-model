##libgl1-mesa-glx
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
sample_pairs_folder = 'sample_pairs'  # Folder containing sample image pairs

# Model paths
meter_classifier_model_path = os.path.join(models_folder, 'meter_classifier.pth')
yolo_model_path = os.path.join(models_folder, 'yolo-screen-obb-grayscale.pt')
ocr_model_path = os.path.join(models_folder, 'model_float16.tflite')
screen_quality_classifier_model_path = os.path.join(models_folder, 'screen_classifier_grayscale.pth')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the meter classifier model
@st.cache_resource
def load_meter_classifier(model_path):
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
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
    try:
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"An error occurred while loading the screen quality classifier: {e}")
        raise

# Define transformations for the classifiers
common_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

screen_clf_transform = transforms.Compose([
    transforms.Resize((288, 896)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def prepare_input(image_path):
    input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_data is None:
        raise ValueError(f"Failed to read image '{image_path}' for OCR.")
    input_data = cv2.resize(input_data, (200, 31))
    input_data = input_data[np.newaxis]
    input_data = np.expand_dims(input_data, 3)
    input_data = input_data.astype('float32') / 255
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
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perform_classification(image, model, transform, class_names):
    image_transformed = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_transformed)
        _, prediction = torch.max(outputs, 1)
        predicted_class = class_names[prediction.item()]
    return predicted_class

def perform_yolo_detection(image, yolo_model, device='cpu'):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.merge([grayscale_image] * 3)
    results = yolo_model(grayscale_image)
    for result in results:
        boxes = result.obb.xyxyxyxy
        confs = result.obb.conf
        if len(boxes) == 0:
            return None, None
        max_conf_idx = np.argmax(confs.cpu().numpy())
        box = boxes[max_conf_idx].cpu().numpy().reshape((4, 2))
        confidence = confs[max_conf_idx].cpu().item()
        return box, confidence
    return None, None

def preprocess(img):
    grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(grayimg, (200, 31))    
    return resized

def crop_image(image, box):
    pts = box.astype(np.float32)
    pts = order_points(pts)
    widthA = np.linalg.norm(pts[2] - pts[3])
    widthB = np.linalg.norm(pts[1] - pts[0])
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(pts[1] - pts[2])
    heightB = np.linalg.norm(pts[0] - pts[3])
    maxHeight = max(int(heightA), int(heightB))
    if maxWidth <= 0 or maxHeight <= 0:
        raise ValueError("Invalid dimensions for cropping.")
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    if warped.shape[1] < warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return preprocess(warped)

def perform_ocr(cropped_image, ocr_interpreter):
    _, buffer = cv2.imencode('.png', cropped_image)
    cropped_bytes = buffer.tobytes()
    temp_path = 'temp_cropped.png'
    with open(temp_path, 'wb') as f:
        f.write(cropped_bytes)
    try:
        ocr_output = predict_ocr(temp_path, ocr_interpreter)
        text = "".join([alphabet[int(index)] for index in ocr_output[0] if int(index) not in [blank_index, -1]])
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

def draw_bounding_box(image, box, color=(255, 0, 0), thickness=22):
    box = box.astype(int)
    for i in range(4):
        pt1 = tuple(box[i])
        pt2 = tuple(box[(i + 1) % 4])
        cv2.line(image, pt1, pt2, color, thickness)
    return image

def calculate_bill(units_consumed):
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
        rate = slabs[-1][1]
        total_amount += units_remaining * rate

    return total_amount

def generate_pdf(previous_reading, current_reading, usage, bill_amount, images, cropped_images):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    center_style = ParagraphStyle(name='center', alignment=TA_CENTER, fontSize=18, fontName="Courier-Oblique")
    center_font = ParagraphStyle(name='center', alignment=TA_CENTER, fontName='Helvetica')
    normal_text = ParagraphStyle(
        name='normal_text',
        alignment=0,
        fontName='Times-Roman',
        fontSize=12
    )
    normal_text_2 = ParagraphStyle(
        name='normal_text_2',
        alignment=0,
        fontName='Times-Roman',
        fontSize=10
    )

    # Add company logo
    logo_path = 'company_logo.png'
    if os.path.exists(logo_path):
        logo = ReportLabImage(logo_path, width=1*inch, height=1*inch)
        elements.append(logo)
        elements.append(Spacer(1, 12))

    # Add title
    elements.append(Paragraph("Electricity Bill", center_font))
    elements.append(Spacer(1, 12))

    # Add images and readings
    elements.append(Paragraph("Meter Reading Details", normal_text))
    elements.append(Spacer(1, 12))

    # Display previous and current images
    if images:
        for idx, (image, cropped_image) in enumerate(zip(images, cropped_images), start=1):
            month = "Previous" if idx == 1 else "Current"
            elements.append(Paragraph(f"{month} Month Image", normal_text))
            elements.append(Spacer(1, 12))

            # Add uploaded image
            uploaded_image_pil = image
            uploaded_image_buffer = BytesIO()
            uploaded_image_pil.save(uploaded_image_buffer, format='PNG')
            uploaded_image_buffer.seek(0)
            reportlab_uploaded_image = ReportLabImage(uploaded_image_buffer, width=3*inch, height=2*inch)
            elements.append(reportlab_uploaded_image)
            elements.append(Spacer(1, 12))

            # Add cropped image
            if cropped_image is not None:
                cropped_image_pil = Image.fromarray(cropped_image)
                cropped_image_buffer = BytesIO()
                cropped_image_pil.save(cropped_image_buffer, format='PNG')
                cropped_image_buffer.seek(0)
                reportlab_cropped_image = ReportLabImage(cropped_image_buffer, width=3*inch, height=1*inch)
                elements.append(reportlab_cropped_image)
                elements.append(Spacer(1, 12))

                # Add reading
                if idx == 1:
                    elements.append(Paragraph(f"Previous Reading: {previous_reading} kWh", normal_text_2))
                else:
                    elements.append(Paragraph(f"Current Reading: {current_reading} kWh", normal_text_2))
                elements.append(Spacer(1, 12))

    # Add calculations
    elements.append(Spacer(1, 24))
    elements.append(Paragraph("Bill Calculations:", styles['Heading1']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Units Consumed = {current_reading} - {previous_reading} = {usage} kWh", center_style))
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(f"Total Bill Amount: Rs. {bill_amount:.2f}", center_style))

    # Build PDF
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

def process_image_pair(previous_image, current_image, meter_classifier, yolo_model, ocr_interpreter, screen_quality_classifier):
    # Initialize lists to store images and cropped images
    uploaded_images = [previous_image, current_image]
    cropped_images = [None, None]
    readings = {}
    
    # Create two columns for side-by-side display
    display_cols = st.columns(2)
    
    for idx, (label, image, display_col) in enumerate(zip(['Previous', 'Current'], uploaded_images, display_cols)):
        with display_col:
            st.subheader(f"{label} Month Image")

            try:
                image_np = np.array(image)

                # Meter Classification
                class_names_meter = ['meter', 'no_meter']
                predicted_class_meter = perform_classification(image, meter_classifier, common_transform, class_names_meter)
                if predicted_class_meter == 'meter':
                    st.success("Meter device is **visible** in the image.")
                else:
                    st.warning("Meter device might **not be visible** or multiple devices present.")

                # YOLO Detection
                try:
                    box, confidence = perform_yolo_detection(image_np, yolo_model)

                    if box is not None:
                        st.write(f"**Detection Confidence:** {confidence:.2f}")

                        # Draw bounding box on the image
                        image_with_box = image_np.copy()
                        image_with_box = draw_bounding_box(image_with_box, box)
                        st.image(image_with_box, caption='Image with Bounding Box', width=300)

                        # Crop the detected area
                        cropped_image = crop_image(image_np, box)
                        cropped_images[idx] = cropped_image

                        st.image(cropped_image, caption='Cropped Meter Screen', width=300)

                        # Screen Quality Classification
                        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                        class_names_quality = ['ok', 'ng']
                        predicted_class_quality = perform_classification(
                            cropped_pil,
                            screen_quality_classifier,
                            screen_clf_transform,
                            class_names_quality
                        )
                        if predicted_class_quality == 'ok':
                            st.success("Screen quality is **Readable**.")
                        else:
                            st.warning("Screen quality is **Unreadable** (Not Good).")

                        # Perform OCR if screen quality is OK
                        if predicted_class_quality == 'ok':
                            ocr_result = perform_ocr(cropped_image, ocr_interpreter)
                            st.write("### OCR Result:")
                            st.write(f"#### {ocr_result}")

                            # Editable OCR Result
                            reading = st.number_input(
                                f"Edit {label} Meter Reading (kWh):",
                                value=float(ocr_result) if ocr_result.replace('.', '', 1).isdigit() else 0.0,
                                min_value=0.0,
                                step=1.0,
                                key=f'reading_{label}'
                            )
                            readings[label] = reading
                        else:
                            st.warning("OCR skipped due to poor screen quality.")
                            readings[label] = 0.0
                    else:
                        st.warning("Meter screen unclear/not detected in the image.")
                        cropped_images[idx] = None
                        readings[label] = 0.0
                except Exception as e:
                    st.error(f"An error occurred during YOLO detection or cropping: {e}")
                    cropped_images[idx] = None
                    readings[label] = 0.0
            except Exception as e:
                st.error(f"An error occurred while processing the {label} month image: {e}")
                cropped_images[idx] = None
                readings[label] = 0.0

    return uploaded_images, cropped_images, readings

# Streamlit App
def main():
    st.set_page_config(page_title="Meter OCR App", layout="wide")
    import base64

    # Initialize readings
    readings = {}

    # Function to load and encode image
    def load_image_as_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    # Check if company_logo.png exists
    if os.path.exists("company_logo.png"):
        logo_base64 = load_image_as_base64("company_logo.png")
        logo_html = f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{logo_base64}" width="150" style="margin-right: 50px;">
                <div>
                    <h1>AI-Powered Meter Display Interpretation System</h1>
                    <h3>HackAP Hackathon: Power Distribution - Problem Statement 5</h3>
                </div>
            </div>
        """
        st.markdown(logo_html, unsafe_allow_html=True)
    else:
        st.title("AI-Powered Meter Display Interpretation System")
        st.subheader("HackAP Hackathon: Power Distribution - Problem Statement 5")

    # Load models
    with st.spinner("Loading models..."):
        meter_classifier = load_meter_classifier(meter_classifier_model_path)
        yolo_model = load_yolo_model(yolo_model_path)
        ocr_interpreter = load_ocr_interpreter(ocr_model_path)
        screen_quality_classifier = load_screen_quality_classifier(screen_quality_classifier_model_path)

    st.success("Models loaded successfully!")

    st.header("Load Sample Image Pairs or Upload Your Own")

    # Button to load sample pairs
    sample_buttons = st.columns(3)
    sample_pairs = [
        ("Sample Pair 1", "a1.jpg", "a2.jpg"),
        ("Sample Pair 2", "a3.jpg", "a4.jpg"),
        ("Sample Pair 3", "a5.jpg", "a6.jpg")
    ]

    sample_loaded = False  # Flag to check if sample images are loaded

    for idx, (label, img1, img2) in enumerate(sample_pairs):
        if sample_buttons[idx].button(label):
            sample_prev_path = os.path.join(sample_pairs_folder, img1)
            sample_curr_path = os.path.join(sample_pairs_folder, img2)
            if os.path.exists(sample_prev_path) and os.path.exists(sample_curr_path):
                # Load images
                prev_image = Image.open(sample_prev_path).convert('RGB')
                curr_image = Image.open(sample_curr_path).convert('RGB')
                
                # Process image pair
                uploaded_images, cropped_images, readings = process_image_pair(
                    previous_image=prev_image,
                    current_image=curr_image,
                    meter_classifier=meter_classifier,
                    yolo_model=yolo_model,
                    ocr_interpreter=ocr_interpreter,
                    screen_quality_classifier=screen_quality_classifier
                )
                sample_loaded = True
            else:
                st.error("Sample images not found in the 'sample_pairs' folder.")

    if not sample_loaded:
        st.write("---")
        st.subheader("Or Upload Your Own Meter Images")

        # File uploaders for previous and current images
        prev_col, curr_col = st.columns(2)
        with prev_col:
            uploaded_prev = st.file_uploader(
                "Upload Previous Month's Meter Image",
                type=["png", "jpg", "jpeg", "bmp", "gif"],
                key='previous'
            )
        with curr_col:
            uploaded_curr = st.file_uploader(
                "Upload Current Month's Meter Image",
                type=["png", "jpg", "jpeg", "bmp", "gif"],
                key='current'
            )

        if uploaded_prev and uploaded_curr:
            st.header("Processing Uploaded Images")

            # Load images
            prev_image = Image.open(uploaded_prev).convert('RGB')
            curr_image = Image.open(uploaded_curr).convert('RGB')

            # Process image pair
            uploaded_images, cropped_images, readings = process_image_pair(
                previous_image=prev_image,
                current_image=curr_image,
                meter_classifier=meter_classifier,
                yolo_model=yolo_model,
                ocr_interpreter=ocr_interpreter,
                screen_quality_classifier=screen_quality_classifier
            )

    # Editable input boxes and billing calculations
    if 'Previous' in readings and 'Current' in readings:
        previous_reading = readings['Previous']
        current_reading = readings['Current']

        # Input validation
        if current_reading < previous_reading:
            st.error("Current reading cannot be less than previous reading.")
        else:
            usage = current_reading - previous_reading
            bill_amount = calculate_bill(usage)

            st.header("Billing Calculations")
            billing_col1, billing_col2 = st.columns(2)
            with billing_col1:
                st.write(f"**Previous Reading:** {previous_reading} kWh")
                st.write(f"**Current Reading:** {current_reading} kWh")
            with billing_col2:
                st.write(f"**Units Consumed:** {usage} kWh")
                st.write(f"**Total Bill Amount:** Rs. **{bill_amount:.2f}**")

            # Generate PDF
            pdf_data = generate_pdf(
                previous_reading=previous_reading,
                current_reading=current_reading,
                usage=usage,
                bill_amount=bill_amount,
                images=uploaded_images,
                cropped_images=cropped_images
            )

            # Provide download button
            st.download_button(
                label="Download Bill as PDF",
                data=pdf_data,
                file_name="meter_bill.pdf",
                mime='application/pdf'
            )
    else:
        st.info("Please upload images for both previous and current months or select a sample pair to proceed.")

if __name__ == '__main__':
    main()
