import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import numpy as np
import cv2
import pytesseract as pt
from io import BytesIO
from pydantic import BaseModel
from typing import List
from PIL import Image
from datetime import datetime, timedelta
import os
import easyocr
from typing import Tuple
import shutil
import os


# # กำหนดค่า logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# โหลดโมเดล ONNX
try:
    net = ort.InferenceSession("./best.onnx")
except Exception as e:
    # logger.error(f"ไม่สามารถโหลดโมเดล ONNX: {str(e)}")
    raise

app = FastAPI()
UPLOAD_FOLDER = 'img_input/'

engine = create_engine('sqlite:///User_input.db')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@app.post("/upload-DB/")
async def upload_image(file: UploadFile = File(...)):
    session = SessionLocal()
    try:
        # สร้าง path และบันทึกไฟล์ลงในโฟลเดอร์ img_input
        file_location = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # บันทึก path ลงในฐานข้อมูล
        session.execute(text("INSERT INTO uploaded_files (file_path) VALUES (:file_path)"), {"file_path": file_location})
        session.commit()

        return {"file_path": file_location}
    except Exception as e:
        session.rollback()
        return {"error": str(e)}
    finally:
        session.close()
        # Use 'text()' to wrap the raw SQL command



# กำหนด middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตให้เข้าถึงจากทุกโดเมน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# กำหนดคลาสผลลัพธ์ป้ายทะเบียน
class PlateResultImg(BaseModel):

    plate: str  # หมายเลขป้ายทะเบียน
    # dateTime: str = ""   # วันและเวลาที่ตรวจพบ

INPUT_WIDTH = 640 
INPUT_HEIGHT = 640

def get_detections(image, net):
    # ตรวจสอบว่า image เป็น NumPy array หรือไม่ ถ้าไม่ใช่ให้แปลง
    if not isinstance(image, np.ndarray):
        image = np.array(image).any()  # แปลง PIL.Image เป็น NumPy array

    # 1. CONVERT IMAGE TO YOLO FORMAT
    image = image.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL (ONNX runtime)
    # สร้าง blob จาก input image
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    
    # ทำให้แน่ใจว่า input blob มีรูปแบบ (Batch, Channel, Height, Width)
    blob = np.array(blob, dtype=np.float32)

    # ดึงชื่อ input ของโมเดล
    input_name = net.get_inputs()[0].name

    # ใช้ onnxruntime เพื่อ run โมเดล
    preds = net.run(None, {input_name: blob})

    detections = preds[0]

    return input_image, detections

def non_maximum_supression(input_image,detections):

    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    # print('detection', detections)
    detections = detections[0]
    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        # print('confidence',confidence)
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)

    return boxes_np, confidences_np, index

def yolo_predictions(image,net):
    # step-1: detections
    input_image, detections = get_detections(image,net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    result_image = extract_text(image,boxes_np,confidences_np,index)
    
    return result_image

def extract_text(image,boxes_np,confidences_np,index):

    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        # print(conf_text)

    x, y, w, h = boxes_np[ind]
    roi = image[y:y + h, x:x + w]
    # print('roi', roi)

    # แก้ไขการตรวจสอบให้ถูกต้อง
    if (roi.shape[0] == 0 or roi.shape[1] == 0):
        return 'no number'

    h = int(h * 0.7)  # ปรับขนาดความสูงเป็น 70% ของความสูงเดิม
    roi = roi[:h, :]

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # print('gray_roi', gray_roi)
    reader = easyocr.Reader([('th'),('en')])
    
    text = reader.readtext(gray_roi)
    print(f"OCR Result: {text}")
    return text

def read_file_as_image(data) -> Tuple[np.ndarray, Tuple[int, int]]: # A function to read the image file as a numpy array
    img = Image.open(BytesIO(data)).convert('RGB') # Open the image and convert it to RGB color space
    img_resized = img.resize((640, 640), resample=Image.BICUBIC) # Resize the image to 180 x 180
    image = np.array(img_resized) # Convert the image to a numpy array
    return image

# API สำหรับอัปโหลดภาพ
@app.post("/upload-image/")
async def upload_file(file: UploadFile = File(...)) -> List[PlateResultImg]:

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="อนุญาตเฉพาะไฟล์ภาพเท่านั้น")

    image = read_file_as_image(await file.read())
    # file_data = await file.read()
    # image = Image.open(BytesIO(file_data))

    results = yolo_predictions(image,net)
    # ตรวจสอบผลลัพธ์จาก YOLO
    if isinstance(results, list) and len(results) > 0:
        # ใช้ป้ายทะเบียนแรกในผลลัพธ์ (กรณีที่มีหลายป้าย)
        plate_text = results[0][1]  # แก้ไขให้ตรงกับผลลัพธ์จาก yolo_predictions
        print(plate_text)
        return [PlateResultImg(plate=plate_text)]  # Return a list of PlateResult objects
        

    raise HTTPException(status_code=404, detail="ไม่พบป้ายทะเบียนในภาพ")


# ส่วนของวิดีโอ
# class PlateData(BaseModel):
#     plate: str  # หมายเลขป้ายทะเบียน
#     dateTime: str  # วันและเวลาที่ตรวจพบ

# class PlateResultVideo(BaseModel):
#     results: List[PlateData]  # รายการของข้อมูลป้ายทะเบียน

# def process_video(video_path, net):
#     try:
#         metadata = extract_metadata(video_path)

#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Unable to open video file: {video_path}")

#         fps = cap.get(cv2.CAP_PROP_FPS)

#         frame_number = 0
#         results = []
#         while True:
#             ret, frame = cap.read()

#             if not ret:
#                 break

#             frame_time = calculate_frame_time(metadata, frame_number, fps)
            
#             plate_result = yolo_prediction_vdo(frame, net, metadata, frame_time)
#             if plate_result:
#                 results.append(plate_result)
            
#             frame_number += 1

#         cap.release()

#         print("Processing completed.")

#         return results

#     except Exception as e:
#         print(f"Error in process_video: {str(e)}")
#         return []
    
# def extract_metadata(video_path):
#     try:
#         # ใช้ exiftool ในการดึงข้อมูล metadata
#         result = subprocess.run(['exiftool', '-json', video_path], capture_output=True, text=True, check=True)
#         metadata = json.loads(result.stdout)
#         date_fields = ['DateTimeOriginal', 'CreateDate', 'ModifyDate', 'CreationDate']
#         extracted_dates = {}
        
#         for field in date_fields:
#             if field in metadata[0]:
#                 extracted_dates[field] = metadata[0][field]
#             else:
#                 extracted_dates[field] = "ไม่พบ"
        
#         return extracted_dates

#     except subprocess.CalledProcessError as e:
#         print(f"เกิดข้อผิดพลาด: {e}")
#         return None
    
# def calculate_frame_time(metadata, frame_number, fps):
#     create_date_str = metadata.get('CreateDate', metadata.get('DateTimeOriginal', None))
    
#     if create_date_str:
#         # แปลง string ให้เป็น datetime object
#         create_date = datetime.strptime(create_date_str, '%Y:%m:%d %H:%M:%S')
        
#         # คำนวณเวลาในเฟรมปัจจุบันจาก frame number และ fps (frames per second)
#         frame_time = create_date + timedelta(seconds=frame_number / fps)

#         print(frame_time)
        
#         return frame_time.strftime('%Y-%m-%d %H:%M:%S')

#     else:
#         return "ไม่พบข้อมูลเวลา"
    
# def yolo_prediction_vdo(img,net,metadata, frame_time):
#     # step-1: detections
#     input_image, detections = get_detection_vdo(img,net)
#     # step-2: NMS
#     boxes_np, confidences_np, index = non_maximum_supression_vdo(input_image, detections)
#     # step-3: Drawings
#     result_img = extract_text_vdo(img,boxes_np,confidences_np,index,metadata, frame_time)
#     return result_img

# def get_detection_vdo(img,net):
#     # 1.CONVERT IMAGE TO YOLO FORMAT
#     image = img.copy()
#     row, col, d = image.shape

#     max_rc = max(row,col)
#     input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
#     input_image[0:row,0:col] = image

#     # 2. GET PREDICTION FROM YOLO MODEL
#     blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
#     net.setInput(blob)
#     preds = net.forward()
#     detections = preds[0]
    
#     return input_image, detections

# def non_maximum_supression_vdo(input_image,detections):
    
#     # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    
#     # center x, center y, w , h, conf, proba
#     boxes = []
#     confidences = []

#     image_w, image_h = input_image.shape[:2]
#     x_factor = image_w/INPUT_WIDTH
#     y_factor = image_h/INPUT_HEIGHT

#     for i in range(len(detections)):
#         row = detections[i]
#         confidence = row[4] # confidence of detecting license plate
#         if confidence > 0.4:
#             class_score = row[5] # probability score of license plate
#             if class_score > 0.25:
#                 cx, cy , w, h = row[0:4]

#                 left = int((cx - 0.5*w)*x_factor)
#                 top = int((cy-0.5*h)*y_factor)
#                 width = int(w*x_factor)
#                 height = int(h*y_factor)
#                 box = np.array([left,top,width,height])

#                 confidences.append(confidence)
#                 boxes.append(box)

#     # 4.1 CLEAN
#     boxes_np = np.array(boxes).tolist()
#     confidences_np = np.array(confidences).tolist()
    
#     # 4.2 NMS
#     index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)
    
#     return boxes_np, confidences_np, index

# displayed_results = []
# def extract_text_vdo(img,boxes_np,confidences_np,index, frame_time):
    
#     global displayed_results

#     for ind in index:
#         x,y,w,h =  boxes_np[ind]
#         bb_conf = confidences_np[ind]
    
#     x, y, w, h = boxes_np[ind]
#     roi = img[y:y+h, x:x+w]
    
#     if 0 in roi.shape:
#         return 'ไม่พบหมายเลข'
    
#     h = int(h * 0.7)  
#     roi = roi[:h, :]
    
#     gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     reader = easyocr.Reader([('th'),('en')])
    
#     text = reader.readtext(gray_roi)
#     thai_consonants_and_digits = re.sub(r'[^ก-ฮ0-9]', '', text)
    

#     # แสดงผลลัพธ์ OCR พร้อม Metadata
#     if text and bb_conf > 0.85 and len(re.findall(r'[ก-ฮ]', thai_consonants_and_digits)) == 2:
#         if re.search(r'[ก-ฮ]+\d{4}$', thai_consonants_and_digits):
#             if thai_consonants_and_digits not in displayed_results:
#                 # Display the result and its associated metadata
#                 print(f"OCR Result: {thai_consonants_and_digits} | Metadata: {frame_time}")
#                 displayed_results.append(thai_consonants_and_digits)

#     return (thai_consonants_and_digits,frame_time)

# def read_file_as_video(data) -> str:
#     # สร้างไฟล์ชั่วคราวสำหรับวิดีโอ
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
#         temp_file.write(data)
#         temp_file_path = temp_file.name

#     return temp_file_path

# # API สำหรับอัปโหลดวิดีโอ
# @app.post("/upload-video/")
# async def upload_video(file: UploadFile = File(...)) -> PlateResultVideo:
        
#         if not file.content_type.startswith("video/"):
#             raise HTTPException(status_code=400, detail="อนุญาตเฉพาะไฟล์วิดีโอเท่านั้น")

#         temp_file_path = read_file_as_video(await file.read())

#         plates = process_video(temp_file_path,net)

#         # # สมมติว่าคุณได้ป้ายทะเบียนออกมาจากการประมวลผล
#         # plates = [
#         #     {"plate": "จย 1559", "dateTime": "2024-10-07 14:30:00"},
#         #     {"plate": "กท 1234", "dateTime": "2024-10-07 15:00:00"},
#         #     {"plate": "2กย 896", "dateTime": "2024-10-07 15:45:00"},
#         # ]

#         results = [PlateData(**plate) for plate in plates]  # สร้างรายการของ PlateData

#         os.unlink(temp_file_path)  # ลบไฟล์ชั่วคราวเมื่อเสร็จสิ้น

#         return PlateResultVideo(results=results)

# @app.post("/upload-video/")
# async def upload_video(file: UploadFile = File(...)) -> PlateResultVideo:
#     return {'name':file}

    # if not file.content_type.startswith("video/"):
    #     raise HTTPException(status_code=400, detail="Only video files are allowed")

    # try:
    #     temp_file_path = read_file_as_video(await file.read())
    #     if not temp_file_path:
    #         raise HTTPException(status_code=500, detail="Failed to create temporary file")

    #     plates = process_video(temp_file_path, net)
    #     # results = [PlateData(plate=plate[0], dateTime=plate[1]) for plate in plates if isinstance(plate, tuple) and len(plate) == 2]
    #     results = [PlateData(**plate) for plate in plates]  # สร้างรายการของ PlateData
    #     print('results', results)

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    # finally:
    #     if temp_file_path and os.path.exists(temp_file_path):
    #         os.unlink(temp_file_path)

    # return PlateResultVideo(results=results)

# @app.post("/video/detect-faces")
# def detect_faces(file: UploadFile = File(...)):