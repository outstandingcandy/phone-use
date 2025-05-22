import base64
import cv2 as cv
import numpy as np
from mss import mss
from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionAll
import Quartz
import pyautogui
import time
import json


def get_window_dimensions(hwnd):
    window_info_list = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionIncludingWindow, hwnd
    )
    for window_info in window_info_list:
        window_id = window_info[Quartz.kCGWindowNumber]
        if window_id == hwnd:
            bounds = window_info[Quartz.kCGWindowBounds]
            width = bounds["Width"]
            height = bounds["Height"]
            left = bounds["X"]
            top = bounds["Y"]
            return {"top": top, "left": left, "width": width, "height": height}
    return {"top": 0, "left": 0, "width": 0, "height": 0}


def window_capture(window_name, image_path="screenshot.png"):
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)
    for window in window_list:
        # print(window.get('kCGWindowName', ''))
        if window_name.lower() == window.get("kCGWindowName", "").lower():
            hwnd = window["kCGWindowNumber"]
            print("found window id %s" % hwnd)
    monitor = get_window_dimensions(hwnd)
    with mss() as sct:
        screenshot = np.array(sct.grab(monitor))
        # screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)
        # cv.imshow('Computer Vision', screenshot)
        # resize the image
        screenshot = cv.resize(
            screenshot, (screenshot.shape[1] // 2, screenshot.shape[0] // 2)
        )
        # save the image
        cv.imwrite(image_path, screenshot)
    return monitor["top"], monitor["left"]


def take_phone_screenshot(image_path="screenshot.png"):
    top = 0
    left = 0
    # top, left = window_capture(window_name="iPhone Mirroring", image_path=image_path)
    # resize the image to half size
    # image_path = "/Users/tangjiee/Project/browser-use/Xnip2025-05-13_22-00-28.jpg"
    grounded_image_path = image_path.split(".")[0] + "_grounded.png"
    detections = ground_element(input_path=image_path, output_path=grounded_image_path)
    with open(grounded_image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image, top, left, detections

def get_encoded_cropped_images(image_path, detections):
    # Read the image file
    import cv2
    import base64
    from PIL import Image
    import numpy as np
    import io
    
    # Read original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Convert to RGB (for PIL)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create image content list
    image_content_list = []
    
    # Process each detection and crop the image
    for i, detection in enumerate(detections):
        try:
            # Get bounding box
            box = detection["box"]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Ensure coordinates are within image bounds
            height, width = original_image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # Skip if invalid dimensions
            if x1 >= x2 or y1 >= y2:
                print(f"Skipping invalid box: {box}")
                continue
            
            # Crop the image
            cropped_image = original_image_rgb[y1:y2, x1:x2]
            
            # Convert to PIL Image
            pil_cropped = Image.fromarray(cropped_image)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_cropped.save(buffer, format="JPEG")
            buffer.seek(0)

            # Save the cropped image
            cropped_image_path = f"cropped_{x1}_{y1}_{x2}_{y2}.jpg"
            pil_cropped.save(cropped_image_path)
            
            # Encode to base64
            cropped_encoded = base64.b64encode(buffer.read()).decode("utf-8")
            
            # Add to content list
            image_content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{cropped_encoded}"
                }
            })
            
            print(f"Created crop {i+1} with dimensions {x2-x1}x{y2-y1}")
            
        except Exception as e:
            print(f"Error cropping detection {i}: {e}")
    return image_content_list


def ground_element(input_path="screenshot.png", output_path="output.png"):
    import requests
    import torch
    from PIL import Image
    import numpy as np
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    # model_id = "IDEA-Research/grounding-dino-tiny"
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Make sure the image is properly loaded and converted
    try:
        # Open the image and explicitly convert to RGB
        pil_image = Image.open(input_path).convert('RGB')

        # Check if the image is valid
        if pil_image.width == 0 or pil_image.height == 0:
            raise ValueError(f"Invalid image dimensions: {pil_image.width}x{pil_image.height}")

        # Check for input boxes and buttons
        text_labels = [["app icon", "input box"]]

        # Process the image
        inputs = processor(images=pil_image, text=text_labels, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.1,
            text_threshold=0.1,
            target_sizes=[pil_image.size[::-1]]
        )

        result = results[0]
        detections = []
        for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
            box = [round(x, 2) for x in box.tolist()]
            detection = {
                "object": labels,
                "confidence": round(score.item(), 3),
                "box": box
            }
            detections.append(detection)
            print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
        
        # Convert PIL image to OpenCV format for drawing
        opencv_image = np.array(pil_image)
        opencv_image = opencv_image[:, :, ::-1].copy()  # RGB to BGR for OpenCV
        # Scale the image to high resolution
        opencv_image = cv.resize(opencv_image, (opencv_image.shape[1] * 2, opencv_image.shape[0] * 2))

        
        # Draw the boxes on the OpenCV image, add the cordinates below the boxes
        for box in result["boxes"]:
            box = [int(x) for x in box.tolist()]
            cv.rectangle(opencv_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv.putText(opencv_image, f"({box[0]}, {box[1]}, {box[2]}, {box[3]})", (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

        # Save the annotated image
        cv.imwrite(output_path, opencv_image)

        return detections

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        # Try alternative approach with explicit channel handling
        try:
            # Try with OpenCV
            import cv2
            cv_image = cv2.imread(input_path)
            if cv_image is None:
                print(f"Failed to load image with OpenCV from {input_path}")
                return []
                
            # Convert from BGR to RGB
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image_rgb)
            
            # Now try processing again
            inputs = processor(images=pil_image, text=text_labels, return_tensors="pt").to(device)
            # ... rest of the processing code
            
        except Exception as inner_e:
            print(f"Alternative approach also failed: {str(inner_e)}")
            return []


def add_description_to_cropped_images(image_content_list, detections):
    system_prompt = f"""You are a professional image annotator. You need to provide detailed descriptions for the elements in the mobile interface screenshots. 
            The screenshots are from a mobile app, and you need to identify and describe the elements in the images. 
            The elements include buttons, input boxes, and other UI components. 
            Please provide a detailed description of each element, including its function and any relevant information. 
            The descriptions should be clear and concise, suitable for someone who is not familiar with the app.
            The screenshots are provided in the following format: the first image is the full screenshot, and the subsequent images are cropped sections of the screenshot. 
            Each cropped image corresponds to a specific element in the mobile interface. 
            Please ensure that you provide descriptions for all elements in the cropped images as well.
            Please provide description information for the cropped images in the mobile interface screenshots below, and retain the original information.
            Output the result in JSON format, and the JSON format is as follows:
        {{
            "elements": [
                {{
                    "name": "app icon",
                    "description": "The app icon with the name below it.",
                    "box": [x1, y1, x2, y2]
                }},
                {{
                    "name": "input box",
                    "description": "The input box where the user can enter text.",
                    "box": [x1, y1, x2, y2]
                }}
            ]
        }}
            """
    # Create message with all images - full image and crops
    import json
    detections_json = json.dumps(detections, indent=2)

    # Add all images to the message
    message_contents = [
        {
            "type": "text",
            "text": f"The detail of the input cropped images are as follows: {detections_json}."
                "Output all the cropped images with descriptions in JSON format. Do not miss any cropped images."
                "Do not add any elements that are not in the cropped images."
                "The output list shoud be in the same order as the input cropped image list."
                "The output list size should be the same as the input cropped image list."
        }
    ]
    message_contents.extend(image_content_list)
    
    # Continue with the API call
    import openai
    import re
    from openai.types.chat import (
        ChatCompletionUserMessageParam,
        ChatCompletionSystemMessageParam,
    )
    import os

    # model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    model = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    base_url = "http://127.0.0.1:4000/v1"
    # model = "qwen-vl-max"
    # base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key = os.environ.get("ALIYUN_API_KEY", "sk")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            ChatCompletionSystemMessageParam(
                role="system",
                content=system_prompt,
            ),
            ChatCompletionUserMessageParam(role="user", content=message_contents),
        ],
    )
    text_response = response.choices[0].message.content
    print(text_response)
    
    import re
    import json

    response_text = text_response
    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1

    if json_start >= 0 and json_end > json_start:
        bboxes = json.loads(response_text[json_start:json_end])
        print("Successfully extracted bounding boxes")
        return bboxes
    else:
        raise ValueError("No valid JSON found in response")

def parse_phone_screenshot(image_path, detections):
    # Read the image file
    import cv2
    import base64
    from PIL import Image
    import numpy as np
    import io
    
    all_descriptions = []
    with open(image_path, "rb") as image_file:
        full_image_encoded = base64.b64encode(image_file.read()).decode("utf-8")
    max_cropped_images = 10
    for sub_detections in [detections[i:i + max_cropped_images] for i in range(0, len(detections), max_cropped_images)]:
        cropped_image_content_list = get_encoded_cropped_images(image_path, sub_detections)
        # Process each request_image_list
        image_content_list = [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{full_image_encoded}"
            }
        }]
        # Add the cropped images to the list
        image_content_list.extend(cropped_image_content_list)
        refined_bboxes = add_description_to_cropped_images(image_content_list, sub_detections)
        all_descriptions.extend(refined_bboxes)
    return all_descriptions


def move_mouse(x, y):
    pyautogui.moveTo(x, y)


def click_mouse():
    pyautogui.click()


def press_and_hold_mouse(t):
    pyautogui.mouseDown()
    time.sleep(t)
    pyautogui.mouseUp()


def drag_mouse(x, y):
    pyautogui.dragTo(x, y, duration=0.5)


def type_text(text):
    pyautogui.typewrite(text)


if __name__ == "__main__":
    # Example usage
    # screenshot, top, left = take_phone_screenshot(image_path="screenshot.png")
    # bboxes = ground_element("Xnip2025-05-13_22-00-28.jpg")
    screenshot_path = "screenshot_1.png"
    screenshot, top, left, detections = take_phone_screenshot(screenshot_path)
    touchable_elements = parse_phone_screenshot(screenshot_path, detections)
    print(touchable_elements)
