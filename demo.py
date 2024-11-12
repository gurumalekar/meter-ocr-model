import os
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import string
import torch
import torch.nn as nn
import streamlit as st

alphabet = string.digits + string.ascii_lowercase + '.'
blank_index = len(alphabet)
models_folder = 'model'
meter_classifier_model_path = os.path.join(models_folder, 'meter_classifier.pth')
yolo_model_path = os.path.join(models_folder, 'yolo-screen-obb-grayscale.pt')
ocr_model_path = os.path.join(models_folder, 'model_float16.tflite')
screen_quality_classifier_model_path = os.path.join(models_folder, 'screen_classifier_grayscale.pth')

class CNNBinaryClassifier(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 36 * 112, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

@st.cache_resource
def load_yolo_model(model_path):
    model = YOLO(model_path).to(device)
    return model

@st.cache_resource
def load_ocr_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_meter_classifier(model_path):
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

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

def preprocess(img):
    grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(grayimg, (200, 31))    
    return resized

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

def draw_bounding_box(image, box, color=(255, 0, 255), thickness=11):
    box = box.astype(int)
    for i in range(4):
        pt1 = tuple(box[i])
        pt2 = tuple(box[(i + 1) % 4])
        cv2.line(image, pt1, pt2, color, thickness)
    return image

def main():
    st.set_page_config(page_title="Meter OCR App", layout="wide")
    st.title("📊 Meter OCR Application")
    st.write("### Note that this is a demo and some models are lite versions of actual models. Performance may vary.")
    with st.spinner("Loading models..."):
        meter_classifier = load_meter_classifier(meter_classifier_model_path)
        yolo_model = load_yolo_model(yolo_model_path)
        ocr_interpreter = load_ocr_interpreter(ocr_model_path)
        screen_quality_classifier = load_screen_quality_classifier(screen_quality_classifier_model_path)
    st.success("Models loaded successfully!")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "gif"])
    if uploaded_file is not None:
        st.header("Processing Image")
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        cols = st.columns(5)
        with cols[0]:
            st.write("### Uploaded Image")
            st.image(image, caption='Uploaded Image', use_column_width=True)
        class_names_meter = ['meter', 'no_meter']
        predicted_class_meter = perform_classification(image, meter_classifier, common_transform, class_names_meter)
        with cols[1]:
            st.write("### Meter Classification")
            if predicted_class_meter == 'meter':
                st.success("Meter device is **visible** in the image.")
            else:
                st.warning("Meter device might **not be visible** or multiple devices present.")
        try:
            box, confidence = perform_yolo_detection(image_np, yolo_model)
            if box is not None:
                with cols[2]:
                    st.write("### YOLO Detection")
                    st.write(f"**Detection Confidence:** {confidence:.2f}")
                    image_with_box = image_np.copy()
                    image_with_box = draw_bounding_box(image_with_box, box)
                    st.image(image_with_box, caption='Image with Bounding Box', use_column_width=True)
            else:
                st.warning("Meter screen unclear/not detected in the image.")
        except Exception as e:
            st.error(f"An error occurred during YOLO detection: {e}")
        if box is not None:
            try:
                cropped_image = crop_image(image_np, box)
                with cols[3]:
                    st.write("### Cropped Meter Screen")
                    st.image(cropped_image, caption='Cropped Meter Screen', use_column_width=True)
                cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                class_names_quality = ['ok', 'ng']
                predicted_class_quality = perform_classification(cropped_pil, screen_quality_classifier, screen_clf_transform, class_names_quality)
                with cols[4]:
                    st.write("### Screen Quality Classification")
                    if predicted_class_quality == 'ok':
                        st.success("Screen quality is **OK**.")
                    else:
                        st.warning("Screen quality is **NG** (Not Good).")
                with cols[2]:
                    if predicted_class_quality == 'ok':
                        ocr_result = perform_ocr(cropped_image, ocr_interpreter)
                        st.write("# OCR Result:")
                        st.write(f"## {ocr_result}")
                    else:
                        st.warning("OCR skipped due to poor screen quality.")
            except Exception as e:
                st.error(f"An error occurred during cropping or OCR: {e}")
    else:
        st.info("Please upload an image to proceed.")

if __name__ == '__main__':
    main()
