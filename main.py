import time
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import base64
import io
from PIL import Image
import torch
import numpy as np

# Existing imports
import numpy as np
import torch
from PIL import Image
import io

from utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)
import torch

# yolo_model = get_yolo_model(model_path='/data/icon_detect/best.pt')
# caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="/data/icon_caption_florence")

from ultralytics import YOLO

# if not os.path.exists("/data/icon_detect"):
#     os.makedirs("/data/icon_detect")

from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base", trust_remote_code=True
)

if torch.cuda.is_available():
    yolo_model = YOLO("weights/icon_detect/best.pt").to("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        "weights/icon_caption_florence",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")

    print('using gpu')
else:
    yolo_model = YOLO("weights/icon_detect/best.pt")
    model = AutoModelForCausalLM.from_pretrained(
        "weights/icon_caption_florence",
        trust_remote_code=True,
    )

    print('using cpu')

caption_model_processor = {"processor": processor, "model": model}
print("finish loading model!!!")

app = FastAPI()

class ProcessRequest(BaseModel):
    image_base64: str
    box_threshold: float = 0.05
    iou_threshold: float = 0.1

class ProcessResponse(BaseModel):
    image: str  # Base64 encoded image
    parsed_content_list: List[str]
    label_coordinates: Dict[str, List[float]]


def process(
    image_input: Image.Image, box_threshold: float, iou_threshold: float
) -> ProcessResponse:
    image_save_path = "imgs/saved_image_demo.png"
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }

    start_at = time.time()
    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_save_path,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=True,
    )
    end_at = time.time()
    print(f'ocr time: {end_at - start_at}')

    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
    )
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print("finish processing")
    parsed_content_list_str = "\n".join(parsed_content_list)

    # Encode image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # processed_image_save_path = "imgs/processed_image_demo.png"
    # image.save(processed_image_save_path)

    return ProcessResponse(
        image=img_str,
        parsed_content_list=parsed_content_list,
        label_coordinates=label_coordinates,
    )



@app.post("/process_image", response_model=ProcessResponse)
async def process_image(request: ProcessRequest):
    try:
        image_input = Image.open(io.BytesIO(base64.b64decode(request.image_base64))).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    response = process(image_input, request.box_threshold, request.iou_threshold)
    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
    # or logger.error(f'{exc}')
    print(request, exc_str)
    content = {'status_code': 10422, 'message': exc_str, 'data': None}
    return JSONResponse(content=content, status_code=422)