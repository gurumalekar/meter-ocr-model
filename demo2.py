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

# Define constants
alphabet = string.digits + string.ascii_lowercase + '.'
blank_index = len(alphabet)
models_folder = 'model'
sample_pairs_folder = 'sample_pairs'  # Folder containing sample images

# Model paths
meter_classifier_model_path = os.path.join(models_folder, 'meter_classifier.pth')
yolo_model_path = os.path.join(models_folder, 'yolo-screen-obb-grayscale.pt')
ocr_model_path = os.path.join(models_folder, 'model_float16.tflite')
screen_quality_classifier_model_path = os.path.join(models_folder, 'best_meter_clf.pt')

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
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"An error occurred while loading the screen quality classifier: {e}")
        raise
    return model


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

def draw_bounding_box(image, box, color=(255, 0, 0), thickness=2):
    box = box.astype(int)
    for i in range(4):
        pt1 = tuple(box[i])
        pt2 = tuple(box[(i + 1) % 4])
        cv2.line(image, pt1, pt2, color, thickness)
    return image

def perform_yolo_classification(crop, model):
    image = cv2.resize(crop, (31, 31))
    result = model(image)
    conf_scores = result[0].probs.data.cpu().detach().numpy()
    if len(conf_scores) != 2:
        raise ValueError("Input must be a list or tuple of two values.")
    ng_score, ok_score = conf_scores
    return "ok" if ok_score > ng_score else "ng"

def process_image(image, meter_classifier, yolo_model, ocr_interpreter, screen_quality_classifier):
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
                st.image(image_with_box, caption='Image with Bounding Box', width=600)

                # Crop the detected area
                cropped_image = crop_image(image_np, box)

                st.image(cropped_image, caption='Cropped Meter Screen', width=600)

                # Screen Quality Classification
                class_names_quality = ['ok', 'ng']
                predicted_class_quality = perform_yolo_classification(
                    cropped_image,
                    screen_quality_classifier,
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
                else:
                    st.warning("OCR skipped due to poor screen quality.")
            else:
                st.warning("Meter screen unclear/not detected in the image.")
        except Exception as e:
            st.error(f"An error occurred during YOLO detection or cropping: {e}")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

    return image, cropped_image

# Streamlit App
def main():
    st.set_page_config(page_title="Meter OCR App")
    import base64

    # Initialize session state for processing
    if 'processing' not in st.session_state:
        st.session_state['processing'] = False

    # Function to load and encode image
    def load_image_as_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    # Check if company_logo.png exists
    if os.path.exists("company_logo.png"):
        logo_base64 = load_image_as_base64("company_logo.png")
        logo_html = f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{logo_base64}" width="150" style="margin-right: 20px;">
                <div>
                    <h1 style="margin-bottom: 5px;">AI-Powered Meter Display Interpretation System</h1>
                    <h3 style="margin-top: 0;">HackAP Hackathon: Power Distribution - Problem Statement 5</h3>
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

    st.header("Load Sample Images or Upload Your Own")

    # Dropdown menu for selecting mode
    mode = st.selectbox("Select Mode", ["Explore Sample Images", "Upload Your Own Image"])

    if mode == "Explore Sample Images":
        # Button to load sample images
        sample_images = [
            ("Sample Image 1", "a1.jpg"),
            ("Sample Image 2", "a2.jpg"),
            ("Sample Image 3", "a3.jpg")
        ]

        for idx, (label, img) in enumerate(sample_images):
            sample_image_path = os.path.join(sample_pairs_folder, img)
            if os.path.exists(sample_image_path):
                if st.button(label):
                    # Set processing state
                    st.session_state['processing'] = True

                    # Load image
                    image = Image.open(sample_image_path).convert('RGB')

                    # Process image
                    uploaded_image, cropped_image = process_image(
                        image=image,
                        meter_classifier=meter_classifier,
                        yolo_model=yolo_model,
                        ocr_interpreter=ocr_interpreter,
                        screen_quality_classifier=screen_quality_classifier
                    )

                    # Reset processing state
                    st.session_state['processing'] = False
            else:
                st.error("Sample image not found in the 'sample_pairs' folder.")

    elif mode == "Upload Your Own Image":
        st.subheader("Upload Your Image")
        # File uploader for the image
        uploaded_image_file = st.file_uploader(
            "Upload Meter Image",
            type=["png", "jpg", "jpeg", "bmp", "gif"],
            key='uploaded_image'
        )

        if uploaded_image_file:
            if st.button("Process Uploaded Image"):
                # Set processing state
                st.session_state['processing'] = True

                # Load image
                image = Image.open(uploaded_image_file).convert('RGB')

                # Process image
                uploaded_image, cropped_image = process_image(
                    image=image,
                    meter_classifier=meter_classifier,
                    yolo_model=yolo_model,
                    ocr_interpreter=ocr_interpreter,
                    screen_quality_classifier=screen_quality_classifier
                )

                # Reset processing state
                st.session_state['processing'] = False
        else:
            st.info("Please upload an image to proceed.")

if __name__ == '__main__':
    main()
