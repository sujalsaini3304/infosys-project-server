from fastapi import FastAPI, File, UploadFile , HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import cv2
from ultralytics import YOLO
import json
from collections import defaultdict


from dotenv import load_dotenv
import bcrypt
from pydantic import BaseModel ,  EmailStr , Field
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import pytz

app = FastAPI()
load_dotenv()
tz = pytz.timezone("UTC")
DESIRED_TIMEZONE = pytz.timezone("Asia/Kolkata")

client = AsyncIOMotorClient(os.getenv('MONGODB_URI'))
db = client["infosysCrowdCountProject"]

# Loading YOLO model
model = YOLO("yolov8n.pt")

# Allow frontend to read header for website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Detection-Results", "X-Detection-Summary"]  
)


# Signup endpoint
class User(BaseModel):
    username:str
    email:str
    password:str

@app.post("/api/create/user")
async def create_user(payload: User):
    collection = db["user"]

    # 🔐 Check if the user already exists
    existing_user = await collection.find_one({"email": payload.email})
    if existing_user:
        raise HTTPException(status_code=409, detail="User already exists")

    hashed_password = bcrypt.hashpw(payload.password.encode('utf-8'), bcrypt.gensalt())

    item = {
        "username": payload.username,
        "email": payload.email,
        "password": hashed_password.decode('utf-8'),  # Store as string
        "is_email_verified": False,
        "created_at": datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    }

    db_response = await collection.insert_one(item)

    return {
        "message": "User created successfully",
        "id": str(db_response.inserted_id)
    }




# Login endpoint
class UserDetail(BaseModel):
    email : EmailStr
    password : str
    

@app.post("/api/verify/user")
async def fetch_data(userData: UserDetail):
    collection = db["user"]
    document = await collection.find_one({"email": userData.email})

    if document:
        document["_id"] = str(document["_id"])
        if(bcrypt.checkpw(userData.password.encode('utf-8'), document["password"].encode('utf-8'))):      
           return {"data": document , "message" : "Success" ,  "verify" : True}
        else:
            return {"data": document, "message": "Password Mismatch." , "verify" : False}
    else:
        return {"data": None, "message": "User not found." , "verify" : False}



# Update password endpoint    
class UserInfo(BaseModel):
    email : EmailStr
    password : str

@app.post("/api/update/password")
async def  update_password(payload : UserInfo ):
    collection = db["user"]
    user = await collection.find_one({"email": payload.email})
    if not user :
       raise HTTPException(status_code=404, detail="User not found.")

    if bcrypt.checkpw(payload.password.encode("utf-8"), user["password"].encode("utf-8")):
       raise HTTPException(status_code=400, detail="Using the old password.")

    hashed_password = bcrypt.hashpw(payload.password.encode('utf-8'), bcrypt.gensalt())
    result = await collection.update_one(
        {"email": payload.email},
        {"$set": {"password": hashed_password.decode('utf-8')}}
    )
    
    if result.modified_count > 0:
        return {
            "status" : "Success",
            "message" : "Password updated successfully.",
            "flag" : True
        }
    else:
        return {
            "status" : "Failed",
            "message" : "Password not changed.",
            "flag" : False
        }





@app.post("/upload/image")
async def detect_image(file: UploadFile = File(...)):
    upload_dir = "temp/uploads"
    output_dir = "temp/output"

     # Removes older files
    if os.path.exists("temp"):
       shutil.rmtree("temp")


    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Save uploaded image
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get predictions
    results = model(file_path)
    result = results[0]

    # Extract detections with confidence
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0]) * 100  # confidence in %
        label = model.names[cls_id]
        detections.append({"object": label, "confidence": round(conf, 2)})

    # Aggregate counts per object
    from collections import defaultdict
    summary_dict = defaultdict(lambda: {"count": 0, "avg_conf": 0})
    for det in detections:
        obj = det["object"]
        summary_dict[obj]["count"] += 1
        summary_dict[obj]["avg_conf"] += det["confidence"]

    # Final summary array
    summary = [
        {
            "object": obj,
            "count": data["count"],
            "avg_confidence": round(data["avg_conf"] / data["count"], 2)
        }
        for obj, data in summary_dict.items()
    ]

    # Save annotated image
    annotated_image = result.plot()
    output_file = os.path.join(output_dir, f"annotated_{file.filename}")
    cv2.imwrite(output_file, annotated_image)

    # Return both image and summary as JSON header
    response = FileResponse(output_file, media_type="image/jpeg")
    import json
    response.headers["X-Detection-Summary"] = json.dumps(summary)

    return response


@app.post("/upload/video")
async def detect_video(file: UploadFile = File(...)):
    upload_dir = "temp/uploads"
    output_dir = "temp/output"

     # Removes older files
    if os.path.exists("temp"):
       shutil.rmtree("temp")


    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Save uploaded video
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Open video
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video file with H.264 codec (browser-compatible)
    output_file = os.path.join(output_dir, f"annotated_{file.filename}")
    
    # Use H.264 codec for browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or try 'H264' or 'X264'
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    summary_dict = defaultdict(lambda: {"count": 0, "avg_conf": 0})

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the frame
        results = model(frame)
        result = results[0]

        # Annotate frame
        annotated_frame = result.plot()
        
        # Convert BGRA to BGR if needed
        if annotated_frame.shape[2] == 4:
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGRA2BGR)

        # Collect detections
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0]) * 100
            label = model.names[cls_id]
            summary_dict[label]["count"] += 1
            summary_dict[label]["avg_conf"] += conf

        # Write annotated frame to output video
        out.write(annotated_frame)

    cap.release()
    out.release()

    # Prepare final summary
    summary = [
        {
            "object": obj,
            "count": data["count"],
            "avg_confidence": round(data["avg_conf"] / data["count"], 2)
        }
        for obj, data in summary_dict.items()
    ]

    # Return video with summary in headers
    response = FileResponse(output_file, media_type="video/mp4")
    response.headers["X-Detection-Summary"] = json.dumps(summary)
    
    # Add CORS headers if needed
    response.headers["Access-Control-Expose-Headers"] = "X-Detection-Summary"
    
    return response



# Single endpoint
@app.post("/upload")
async def detect_media(file: UploadFile = File(...)):
    upload_dir = "temp/uploads"
    output_dir = "temp/output"

    # Removes older files
    if os.path.exists("temp"):
        shutil.rmtree("temp")

    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Save uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Determine file type based on extension
    file_extension = file.filename.lower().split('.')[-1]
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp']
    video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm']

    if file_extension in image_extensions:
        # Process as IMAGE
        # Get predictions
        results = model(file_path)
        result = results[0]

        # Extract detections with confidence
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0]) * 100  # confidence in %
            label = model.names[cls_id]
            detections.append({"object": label, "confidence": round(conf, 2)})

        # Aggregate counts per object
        summary_dict = defaultdict(lambda: {"count": 0, "avg_conf": 0})
        for det in detections:
            obj = det["object"]
            summary_dict[obj]["count"] += 1
            summary_dict[obj]["avg_conf"] += det["confidence"]

        # Final summary array
        summary = [
            {
                "object": obj,
                "count": data["count"],
                "avg_confidence": round(data["avg_conf"] / data["count"], 2)
            }
            for obj, data in summary_dict.items()
        ]

        # Save annotated image
        annotated_image = result.plot()
        output_file = os.path.join(output_dir, f"annotated_{file.filename}")
        cv2.imwrite(output_file, annotated_image)

        # Return both image and summary as JSON header
        response = FileResponse(output_file, media_type="image/jpeg")
        response.headers["X-Detection-Summary"] = json.dumps(summary)

        return response

    elif file_extension in video_extensions:
        # Process as VIDEO
        # Open video
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output video file with H.264 codec (browser-compatible)
        output_file = os.path.join(output_dir, f"annotated_{file.filename}")
        
        # Use H.264 codec for browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or try 'H264' or 'X264'
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        summary_dict = defaultdict(lambda: {"count": 0, "avg_conf": 0})

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO model on the frame
            results = model(frame)
            result = results[0]

            # Annotate frame
            annotated_frame = result.plot()
            
            # Convert BGRA to BGR if needed
            if annotated_frame.shape[2] == 4:
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGRA2BGR)

            # Collect detections
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0]) * 100
                label = model.names[cls_id]
                summary_dict[label]["count"] += 1
                summary_dict[label]["avg_conf"] += conf

            # Write annotated frame to output video
            out.write(annotated_frame)

        cap.release()
        out.release()

        # Prepare final summary
        summary = [
            {
                "object": obj,
                "count": data["count"],
                "avg_confidence": round(data["avg_conf"] / data["count"], 2)
            }
            for obj, data in summary_dict.items()
        ]

        # Return video with summary in headers
        response = FileResponse(output_file, media_type="video/mp4")
        response.headers["X-Detection-Summary"] = json.dumps(summary)
        
        # Add CORS headers if needed
        response.headers["Access-Control-Expose-Headers"] = "X-Detection-Summary"
        
        return response

    else:
        # Unsupported file type
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Please upload an image or video file."
        )






