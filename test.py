import time
from PIL import Image
# from utils import check_ocr_box

for i in range(10):
  start_at = time.time()
  # results = check_ocr_box(
  #   'imgs/saved_image_demo.png',
  #   display_img=False,
  #   output_bb_format="xyxy",
  #   goal_filtering=None,
  #   easyocr_args={"paragraph": False, "text_threshold": 0.9},
  #   use_paddleocr=False,
  # )

  from main import process
  process(
    Image.open('imgs/test.png'),
    box_threshold=0.05,
    iou_threshold=0.1,
  )

  # from paddleocr import PaddleOCR
  # ocr = PaddleOCR()
  # results = ocr.ocr('test.png', cls=True)
  # for line in results[0]:
  #     print(line)

  # import easyocr

  # reader = easyocr.Reader(['en'])
  # result = reader.readtext('imgs/saved_image_demo.png')
  # print(result)

  end_at = time.time()
  print(f'total time: {end_at - start_at}')

# print(result)