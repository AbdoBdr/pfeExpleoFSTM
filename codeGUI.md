import numpy as np
import requests
from PIL import Image, ImageDraw
import gradio as gr
import cv2
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from tensorflow.keras.optimizers import Adam
from io import BytesIO
import torch
# Paths and settings for models
ROAD_MODEL_PATH = 'mobilenet_v2.keras'
WEATHER_MODEL_PATH = r"C:\Users\Desktop\Snow_fog_29\mode27mai_adam.h5"
OBJECT_MODEL_PATH = r"C:\Users\Desktop\yolo-model\
yolov8x-seg.pt
"
VEHICLE_IMAGE_PATH = "
https://github.com/Images/blob/main/image%20
(5).png?raw=true"
ADDITIONAL_IMAGE_URL = "
https://github.com/Images/blob/main/image%20
(6).png?raw=true"
ROAD_TARGET_WIDTH = 224
ROAD_TARGET_HEIGHT = 224
WEATHER_TARGET_WIDTH = 250
WEATHER_TARGET_HEIGHT = 250
OBJECT_TARGET_SIZE = 640
ROAD_LABELS = ["Autoroute", "Nationale", "Tunnel", "Urbain"]
WEATHER_LABELS = ["Snowy", "Foggy", "Rainy", "Clear"]
# Icons for LEDs
icon_urls = {
    "Diurne": "
https://github.com//Images/blob/main/diurne.png?raw=true
",
    "Low Beam": "
https://github.com//Images/blob/main/croisement.png?raw=true
",
    "High Beam": "
https://github.com//Images/blob/main/Route.png?raw=true
",
    "Front Fog Light": "
https://github.com/Images/blob/main/Brouillard_Avant.png?raw=true
",
    "Rear Fog Light": "
https://github.com/Images/blob/main/Brouillard_Arri
ère.png?raw=true",
    "Parking Light": "
https://github.com/Images/blob/main/Position.png?raw=true
"
}
# Load and compile models
road_model = load_model(ROAD_MODEL_PATH, compile=False)
road_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
weather_model = load_model(WEATHER_MODEL_PATH, compile=False)
weather_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# Load YOLO model for object detection
object_model = YOLO(OBJECT_MODEL_PATH)
def preprocess_image(image, width, height, color_mode='RGB'):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((width, height))
    img_array = np.array(image)
    if color_mode == 'L':
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_array = img_array.reshape((height, width, 1))
    img_array = img_array / 255.0
    return img_array
def predict_conditions(image):
    road_img_array = preprocess_image(image, ROAD_TARGET_WIDTH, ROAD_TARGET_HEIGHT)
    weather_img_array = preprocess_image(image, WEATHER_TARGET_WIDTH, WEATHER_TARGET_HEIGHT, 'L')
    road_prediction = road_model.predict(np.expand_dims(road_img_array, axis=0))
    weather_prediction = weather_model.predict(np.expand_dims(weather_img_array, axis=0))
    road_type = ROAD_LABELS[np.argmax(road_prediction)]
    weather_condition = WEATHER_LABELS[np.argmax(weather_prediction)]
    hls_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HLS)
    l_channel = hls_image[:, :, 1]
    lightness = np.mean(l_channel)
    is_daytime = lightness > 120
    return road_type, weather_condition, is_daytime
def predict_objects(image):
    try:
        results = object_model.predict(source=image, imgsz=OBJECT_TARGET_SIZE, iou=0.45)
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
        return im
    except Exception as e:
        print(f"Error during object detection: {e}")
        return image
def pad_image_to_even_dimensions(image, divisor=32):
    h, w = image.shape[:2]
    new_h = (h + divisor - 1) // divisor * divisor
    new_w = (w + divisor - 1) // divisor * divisor
    padded_image = np.zeros((new_h, new_w, 3), dtype=image.dtype)
    padded_image[:h, :w, :] = image
    return padded_image
def predict_lane_segmentation(image):
    img_array = np.array(image)
    img_array = pad_image_to_even_dimensions(img_array)
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor =
img_tensor.to
('cpu')
    try:
        with torch.no_grad():
            fs_output = fs_model(img_tensor)
            line_output = line_model(img_tensor)
        fs_out = torch.argmax(fs_output, axis=1).cpu().numpy().squeeze()
        line_out = torch.argmax(line_output, axis=1).cpu().numpy().squeeze()
        fs_mask_uint8 = fs_out.astype('uint8')
        line_mask_uint8 = line_out.astype('uint8')
        fs_mask_resize = cv2.resize(fs_mask_uint8, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
        line_mask_resize = cv2.resize(line_mask_uint8, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
        copy_img = img_array.copy()
        img_array[fs_mask_resize == 1, :] = (255, 0, 125)
        img_array[line_mask_resize == 1, :] = (0, 0, 255)
        img_array[line_mask_resize == 2, :] = (38, 255, 255)
        opac_image = (img_array / 2 + copy_img / 2).astype(np.uint8)
        segmented_image = Image.fromarray(opac_image)
        return segmented_image
    except Exception as e:
        print(f"Error during lane segmentation: {e}")
        return image
def update_leds(road_type, weather_condition, is_daytime):
    led_status = {key: False for key in icon_urls}
    if is_daytime:
        if road_type in ["Autoroute", "Nationale", "Urbain"]:
            if weather_condition == "Clear":
                led_status["Diurne"] = True
            elif weather_condition == "Rainy":
                led_status["Low Beam"] = True
            elif weather_condition == "Foggy":
                led_status["Front Fog Light"] = True
                led_status["Low Beam"] = True
            elif weather_condition == "Snowy":
                led_status["Low Beam"] = True
                led_status["Rear Fog Light"] = True
        elif road_type == "Tunnel":
            led_status["Low Beam"] = True
    else:
        if road_type in ["Autoroute", "Nationale", "Urbain"]:
            if weather_condition == "Clear":
                led_status["Low Beam"] = True
                led_status["High Beam"] = True
            elif weather_condition == "Rainy":
                led_status["Low Beam"] = True
            elif weather_condition == "Foggy":
                led_status["Front Fog Light"] = True
                led_status["Low Beam"] = True
            elif weather_condition == "Snowy":
                led_status["Low Beam"] = True
                led_status["Rear Fog Light"] = True
        elif road_type == "Tunnel":
            led_status["Low Beam"] = True
    return led_status
def highlight_lights(image, road_type, weather_condition, is_daytime):
    try:
        response = requests.get(VEHICLE_IMAGE_PATH, stream=True)
        vehicle_image = Image.open(response.raw).convert('RGBA')
    except Exception as e:
        print(f"Failed to load vehicle image: {e}")
        return image
    draw = ImageDraw.Draw(vehicle_image)
    light_color = "yellow" if is_daytime else "orange"
    if is_daytime:
        drl_coords = [(18, 121, 24, 133), (11, 149, 17, 164), (9, 177, 14, 192),
                      (668, 121, 674, 133), (674, 149, 680, 164), (677, 177, 682, 192)]
        for x0, y0, x1, y1 in drl_coords:
            draw.rectangle([x0, y0, x1, y1], fill=light_color)
        if weather_condition in ["Snowy", "Foggy", "Rainy"]:
            fog_lights = [(631, 198, 9), (60, 198, 9)]
            for x, y, radius in fog_lights:
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=light_color)    
    else:
        if weather_condition == "Clear":
            if road_type in ["Autoroute", "Tunnel"]:
                high_beam_coords = [(631, 48, 651, 62), (50, 48, 70, 62)]
                for x0, y0, x1, y1 in high_beam_coords:
                    draw.rectangle([x0, y0, x1, y1], fill=light_color)
            elif road_type in ["Nationale", "Urbain"]:
                low_beam_lights = [(93, 58, 9), (117, 59, 9), (141, 60, 9),
                                   (555, 63, 9), (581, 61, 9), (606, 60, 9)]
                for x, y, radius in low_beam_lights:
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=light_color)
        elif weather_condition in ["Snowy", "Foggy", "Rainy"]:
            fog_lights = [(631, 198, 9), (60, 198, 9)]
            for x, y, radius in fog_lights:
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=light_color)
            if road_type in ["Autoroute", "Tunnel"]:
                high_beam_coords = [(631, 48, 651, 62), (50, 48, 70, 62)]
                for x0, y0, x1, y1 in high_beam_coords:
                    draw.rectangle([x0, y0, x1, y1], fill=light_color)
            elif road_type in ["Nationale", "Urbain"]:
                low_beam_lights = [(93, 58, 9), (117, 59, 9), (141, 60, 9),
                                   (555, 63, 9), (581, 61, 9), (606, 60, 9)]
                for x, y, radius in low_beam_lights:
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=light_color)
    return vehicle_image
def fetch_additional_image(road_type, weather_condition, is_daytime):
    response = requests.get(ADDITIONAL_IMAGE_URL)
    additional_image = Image.open(BytesIO(response.content)).convert('RGBA')
    draw = ImageDraw.Draw(additional_image)
    light_color = "Orange"
    if not is_daytime:
        draw.rectangle([509, 194, 627, 201], fill=light_color)
        draw.rectangle([58, 195, 178, 201], fill=light_color)
    if weather_condition in ["Snowy", "Foggy", "Rainy"]:
        draw.rectangle([476, 332, 643, 349], fill=light_color)
        draw.rectangle([47, 330, 216, 349], fill=light_color)
    return additional_image
def classify_and_highlight(image):
    road_type, weather_condition, is_daytime = predict_conditions(image)
    object_detection_image = predict_objects(image)
    lane_segmentation_image = predict_lane_segmentation(image)
    highlighted_image = highlight_lights(image, road_type, weather_condition, is_daytime)
    additional_image = fetch_additional_image(road_type, weather_condition, is_daytime)
    leds = update_leds(road_type, weather_condition, is_daytime)
    led_html = "<div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 20px;'>"
    for led_type, url in icon_urls.items():
        color = "#00FF00" if leds[led_type] else "#CCCCCC"
        led_html += f"""
<div style='text-align: center; margin: 10px;'>
<div style='position: relative;'>
<img src='{url}' style='height: 50px; border-radius: 50%; box-shadow: 0 0 10px {color};'>
<div style='
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    border-radius: 50%;
                    box-shadow: 0 0 20px {color};
                    mix-blend-mode: color;
                '></div>
</div>
<div style='margin-top: 5px; font-size: 12px; font-weight: bold; color: black; background-color: {color}; padding: 5px; border-radius: 5px;'>{led_type}</div>
</div>
        """
    led_html += "</div>"
    classification_results = f"<div class='result-box'>Classification: Road Type: {road_type} | Weather Condition: {weather_condition} | Day/Night: {'Day' if is_daytime else 'Night'}</div>"
    return classification_results, highlighted_image, led_html, additional_image, object_detection_image, lane_segmentation_image
css = """
body { background-color: #1c1c1c; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #f5f5f5; }
.gradio-container { max-width: 900px; margin: 0 auto; padding: 20px; background-color: #2e2e2e; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.5); }
.gradio-container h1 { text-align: center; color: #ffffff; font-size: 24px; font-weight: bold; margin-bottom: 20px; }
.gradio-container .gradio-input, .gradio-container .gradio-output { margin-bottom: 20px; }
.gradio-container .gradio-input label, .gradio-container .gradio-output label { font-weight: bold; color: #ffffff; }
.gradio-container .gradio-output img { border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.5); }
.result-box { background-color: #444; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(255,255,255,0.5); color: #ffffff; font-size: 18px; font-weight: bold; text-align: center; }
"""
js_code = """
function switchTab() {
    document.querySelectorAll('.tab-item')[1].click();
}
document.getElementById('switch_button').onclick = switchTab;
"""
def create_app():
    with gr.Blocks(css=css) as app:
        with gr.Tab("Page 1") as page1:
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload an Image")
                    switch_button = gr.Button("Show Object Detection and Lane Segmentation", elem_id="switch_button")
                with gr.Column():
                    classification_results = gr.HTML()
                    highlighted_image = gr.Image()
                    led_html = gr.HTML()
                    additional_image = gr.Image()
            def on_image_change(image):
                results = classify_and_highlight(image)
                return results[:-1] + (gr.update(visible=True),)
            image_input.change(on_image_change, inputs=image_input, outputs=[classification_results, highlighted_image, led_html, additional_image, switch_button])
        with gr.Tab("Page 2") as page2:
            with gr.Row():
                object_detection_image = gr.Image()
                lane_segmentation_image = gr.Image()
            def show_object_detection_and_lane(image):
                return classify_and_highlight(image)[-2:]
            switch_button.click(show_object_detection_and_lane, inputs=image_input, outputs=[object_detection_image, lane_segmentation_image])
    return app
# Create and launch the app
app = create_app()
app.launch(share=True)