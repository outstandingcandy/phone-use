import base64
import cv2 as cv
import numpy as np
from mss import mss
from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionAll
import Quartz
import pyautogui
import time


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
    top, left = window_capture(window_name="iPhone Mirroring", image_path=image_path)
    # resize the image to half size
    image_path = "/Users/tangjiee/Project/browser-use/Xnip2025-05-13_22-00-28.jpg"
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image, top, left


def parse_phone_screenshot(encoded_image):
    system_prompt = """
            You are an expert computer vision system. First describe the image in accurate details, then analyze the provided images and return ONLY a JSON object containing bounding boxes. Be super precise and try to detect as many objects as possible.
        Be accurate and try to detect as many objects as possible. Really open your eyes and see the world.

        Follow these strict rules:
        1. Output MUST be valid JSON with no additional text
        2. Each detected object must have:
           - 'element': descriptive name of the object
           - 'bbox': [x1, y1, x2, y2] coordinates (x1, y1 are top-left, x2, y2 are bottom-right)
           - 'confidence': confidence score (0-1)
        3. Use this exact format:
           {
             "image_number": [
               {
                 "element": "object_name",
                 "bbox": [x1, y1, x2, y2],
                 "confidence": 0.95
               }
             ]
           }
        4. Coordinates must be precise and properly normalized
        5. DO NOT include any explanation or additional text
        """
    message_contents = [
        {
            "type": "text",
            "text": """You are an expert computer vision system. First describe the image in accurate details, then analyze the provided images and return ONLY a JSON object containing bounding boxes. Be super precise and try to detect as many objects as possible.
        Be accurate and try to detect as many objects as possible. Really open your eyes and see the world.

        Follow these strict rules:
        1. Output MUST be valid JSON with no additional text
        2. Each detected object must have:
           - 'element': descriptive name of the object
           - 'bbox': [x1, y1, x2, y2] coordinates, normalized to the image size (x1, y1 are top-left, x2, y2 are bottom-right)
           - 'confidence': confidence score (0-1)
        3. Use this exact format:
           {
             "image_number": [
               {
                 "element": "object_name",
                 "bbox": [x1, y1, x2, y2],
                 "confidence": 0.95
               }
             ]
           }
        4. Coordinates must be precise and properly normalized
        5. DO NOT include any explanation or additional text""",
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + encoded_image},
        },
    ]
    import openai
    import re
    from openai.types.chat import (
        ChatCompletionUserMessageParam,
        ChatCompletionSystemMessageParam,
    )

    import os

    # model = "qwen-vl-max"
    # base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    model = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    base_url="http://127.0.0.1:4000"
    api_key = os.environ.get("ALIYUN_API_KEY", "sk")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    # os.environ["OPENAI_API_BASE"] = "http://
    # os.environ["OPENAI_API_KEY"] = os.environ.get("ALIYUN_API_KEY", "sk")
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            # ChatCompletionSystemMessageParam(role="system", content=system_prompt),
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
    screenshot, top, left = take_phone_screenshot(image_path="screenshot.png")
