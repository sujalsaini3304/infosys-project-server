from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File,Form , UploadFile , HTTPException 
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import cv2
from ultralytics import YOLO
import json
from collections import defaultdict
from typing import Optional
import time 


from dotenv import load_dotenv
import bcrypt
from pydantic import BaseModel ,  EmailStr , Field
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import pytz
import resend
import random

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
    # expose_headers=["X-Detection-Results", "X-Detection-Summary"]
    expose_headers=[
        "X-Detection-Summary",
        "X-Zone-Summary",
        "X-Processing-Time",
        "X-Detection-Results",
        "X-Frame-Density",
        "X-Zone-Density"
    ],  
)


@app.get("/ping")
async def ping():
    return {
        "message":"Server running."
    }



class ResetRequest(BaseModel):
    email: EmailStr


def generate_verification_code():
    """Generate 6-digit verification code"""
    return random.randint(100000, 999999)


@app.post("/api/send/auth/reset/password/email")
async def send_reset_email(payload: ResetRequest):
    resend.api_key = os.getenv("RESEND_API")
    users_collection = db["user"]

    # Check if user exists
    user = await users_collection.find_one({"email": payload.email})
    if not user:
        raise HTTPException(status_code=404, detail="No account found with this email address")

    reset_code = generate_verification_code()
    username = user["username"]

    # Email content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Password Reset Request</title>
        <style>
            body {{
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background-color: #f9fafb;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 540px;
                background: #ffffff;
                margin: 40px auto;
                border-radius: 12px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.05);
                border: 1px solid #e5e7eb;
                overflow: hidden;
            }}
            .header {{
                background: #2563eb;
                color: white;
                text-align: center;
                padding: 20px;
                font-size: 20px;
                font-weight: 600;
            }}
            .content {{
                padding: 30px;
                text-align: center;
            }}
            .greeting {{
                font-size: 16px;
                color: #111827;
                margin-bottom: 10px;
            }}
            .message {{
                font-size: 15px;
                color: #374151;
                margin: 8px 0 18px;
                line-height: 1.5;
            }}
            .code-box {{
                font-size: 30px;
                font-weight: bold;
                letter-spacing: 8px;
                color: #1f2937;
                background: #f3f4f6;
                padding: 14px 22px;
                border-radius: 8px;
                display: inline-block;
                border: 1px solid #d1d5db;
            }}
            .footer {{
                font-size: 12px;
                color: #6b7280;
                text-align: center;
                padding: 18px;
                border-top: 1px solid #e5e7eb;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">Password Reset Request</div>
            <div class="content">
                <p class="greeting">Hey , {username}</p>
                <p class="message">
                    We received a request to reset your password for your Crowd Count account.
                    Use the code below to continue:
                </p>
                <div class="code-box">{reset_code}</div>
                <p class="message">
                    If you didn’t request a password reset, please ignore this message.
                    Your account will remain secure.
                </p>
            </div>
            <div class="footer">
                This is an automated message from Crowd Count. Please do not reply.<br />
                &copy; 2025 Crowd Count using Video Analytics
            </div>
        </div>
    </body>
    </html>
    """

    try:
        params: resend.Emails.SendParams = {
            "from": "Crowd Count <crowd-count@sujalkumarsaini.me>",
            "to": [payload.email],
            "subject": "Reset your Crowd Count password",
            "html": html_content,
        }

        email = resend.Emails.send(params)
        print("Password reset email sent:", email)

        return {
            "success": True,
            "message": f"Password reset email sent to {payload.email}",
            "data": {
                "username": user["username"],
                "email": user["email"]
            },
            "code": reset_code,
        }

    except Exception as e:
        print("Error sending reset email:", e)
        raise HTTPException(status_code=500, detail="Failed to send password reset email")



# Signup process endpoint... 
class UserData(BaseModel):
    username: str
    email: EmailStr


def generate_verification_code():
    """Generate 6-digit verification code"""
    return random.randint(100000, 999999)

@app.post("/api/send/auth/email")
async def sendEmail(payload: UserData):
    resend.api_key = os.getenv("RESEND_API")
    code = generate_verification_code()
    

    # Email design
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background-color: #f9fafb;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 550px;
                background: #ffffff;
                margin: 50px auto;
                border-radius: 14px;
                box-shadow: 0 4px 18px rgba(0,0,0,0.08);
                overflow: hidden;
                border: 1px solid #e5e7eb;
            }}
            .header {{
                background: linear-gradient(90deg, #2563eb, #3b82f6);
                color: white;
                text-align: center;
                padding: 25px 20px;
                font-size: 22px;
                font-weight: 700;
                letter-spacing: 0.6px;
            }}
            .content {{
                padding: 35px 30px;
                text-align: center;
            }}
            .greeting {{
                font-size: 18px;
                color: #111827;
                margin-bottom: 12px;
            }}
            .code-box {{
                font-size: 34px;
                font-weight: 800;
                letter-spacing: 10px;
                color: #1f2937;
                background: #f3f4f6;
                display: inline-block;
                padding: 16px 28px;
                border-radius: 10px;
                margin: 25px 0;
                border: 2px dashed #2563eb;
            }}
            .message {{
                font-size: 16px;
                color: #374151;
                margin-top: 10px;
                line-height: 1.6;
            }}
            .footer {{
                text-align: center;
                font-size: 13px;
                color: #9ca3af;
                padding: 18px;
                border-top: 1px solid #e5e7eb;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">Crowd Count using Video Analytics</div>
            <div class="content">
                <p class="greeting">Hey , <b>{payload.username}</b>,</p>
                <p class="message">
                    We're thrilled to have you on board!<br>
                    Use the verification code below to confirm your email address
                </p>
                <div class="code-box">{code}</div>
                <p class="message">
                    If you didn’t request this email, no worries — simply ignore it.
                </p>
            </div>
            <div class="footer">
                Made with love by the Sujal Kumar Saini
                <br>
                &copy; 2025 Crowd Count using Video Analytics
            </div>
        </div>
    </body>
    </html>
    """

    try:
        params: resend.Emails.SendParams = {
            "from": "Crowd Count <crowd-count@sujalkumarsaini.me>",
            "to": [payload.email],
            "subject": "Verify Your Email - Crowd Count using Video Analytics",
            "html": html_content,
        }

        email = resend.Emails.send(params)
        print("Email sent:", email)

        return {
            "success": True,
            "message": f"Verification email sent to {payload.email}",
            "code": code, 
        }

    except Exception as e:
        print("Error sending email:", e)
        raise HTTPException(status_code=500, detail="Failed to send verification email")






# Serving and mounting file
app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str = ""):
    # Check if the requested path is a file in dist
    file_path = os.path.join("dist", full_path)
    
    # If it's a file that exists, serve it
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Otherwise, serve index.html (for React Router)
    return FileResponse("dist/index.html")




# Signup endpoint
class User(BaseModel):
    username:str
    email:str
    password:str

@app.post("/api/create/user")
async def create_user(payload: User):
    collection = db["user"]

    # Check if the user already exists
    existing_user = await collection.find_one({"email": payload.email})
    if existing_user:
        raise HTTPException(status_code=409, detail="User already exists")

    hashed_password = bcrypt.hashpw(payload.password.encode('utf-8'), bcrypt.gensalt())

    item = {
        "username": payload.username,
        "email": payload.email,
        "password": hashed_password.decode('utf-8'),  # Store as string
        "is_email_verified": True,
        "created_at": datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    }

    db_response = await collection.insert_one(item)

    return {
        "message": "User created successfully",
        "id": str(db_response.inserted_id)
    }



# Delete user from database
class DeleteUserRequest(BaseModel):
    email: EmailStr

@app.post("/api/delete/user")
async def delete_user(payload: DeleteUserRequest):
    collection = db["user"]

    try:
        result = await collection.delete_one({"email": payload.email})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="User not found")

        return {"success": True, "message": "User deleted successfully"}

    except HTTPException as e:
        raise e
    except Exception as e:
        print("Error deleting user:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")







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
    

# @app.post("/new/upload")
# async def detect_media(
#     file: UploadFile = File(...),
#     zones: Optional[str] = Form(None)
# ):
#     start_time = time.time()
#     upload_dir = "temp/uploads"
#     output_dir = "temp/output"

#     # Clean temp directory
#     if os.path.exists("temp"):
#         shutil.rmtree("temp")
#     os.makedirs(upload_dir, exist_ok=True)
#     os.makedirs(output_dir, exist_ok=True)

#     # Save file
#     file_path = os.path.join(upload_dir, file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     file_extension = file.filename.lower().split('.')[-1]
#     image_ext = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp']
#     video_ext = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm']

#     # Parse zones
#     zone_list = []
#     if zones:
#         try:
#             zone_list = json.loads(zones)
#         except json.JSONDecodeError:
#             raise HTTPException(status_code=400, detail="Invalid zone JSON format")

#     # ---------------- IMAGE HANDLING ----------------
#     if file_extension in image_ext:
#         results = model(file_path)
#         result = results[0]
#         height, width = result.orig_img.shape[:2]

#         detections = []
#         people_coords = []

#         for box in result.boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0]) * 100
#             label = model.names[cls_id]

#             if label == "person":
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
#                 people_coords.append((cx, cy))

#             detections.append({"object": label, "confidence": round(conf, 2)})

#         # --- Frame Density ---
#         total_people = len(people_coords)
#         frame_density = round(total_people / (width * height), 8)  # small decimal

#         # --- Zone Density (if zones exist) ---
#         zone_density_list = []
#         for i, zone in enumerate(zone_list):
#             zx1, zy1 = zone["top_left"]["x"], zone["top_left"]["y"]
#             zx2, zy2 = zone["bottom_right"]["x"], zone["bottom_right"]["y"]
#             name = zone.get("name", f"Zone {i+1}")
#             area = (zx2 - zx1) * (zy2 - zy1)
#             if area <= 0:
#                 density = 0
#             else:
#                 count_in_zone = sum(zx1 < x < zx2 and zy1 < y < zy2 for (x, y) in people_coords)
#                 density = round(count_in_zone / area, 8)
#             zone_density_list.append({"zone_name": name, "zone_density": density})

#         # --- Object Summary ---
#         summary_dict = defaultdict(lambda: {"count": 0, "avg_conf": 0})
#         for det in detections:
#             obj = det["object"]
#             summary_dict[obj]["count"] += 1
#             summary_dict[obj]["avg_conf"] += det["confidence"]

#         summary = [
#             {
#                 "object": obj,
#                 "count": data["count"],
#                 "avg_confidence": round(data["avg_conf"] / data["count"], 2)
#             }
#             for obj, data in summary_dict.items()
#         ]

#         # Save annotated image
#         annotated = result.plot()
#         output_file = os.path.join(output_dir, f"annotated_{file.filename}")
#         cv2.imwrite(output_file, annotated)

#         end_time = time.time()
#         processing_time = round(end_time - start_time, 2)

#         # --- Response ---
#         response = FileResponse(output_file, media_type="image/jpeg")
#         response.headers["X-Detection-Summary"] = json.dumps(summary)
#         response.headers["X-Processing-Time"] = str(processing_time)
#         response.headers["X-Frame-Density"] = str(frame_density)
#         if zone_density_list:
#             response.headers["X-Zone-Density"] = json.dumps(zone_density_list)
#         response.headers["Access-Control-Expose-Headers"] = (
#             "X-Detection-Summary, X-Processing-Time, X-Frame-Density, X-Zone-Density"
#         )
#         return response

#     # ---------------- VIDEO HANDLING ----------------
#     elif file_extension in video_ext:
#         cap = cv2.VideoCapture(file_path)
#         fps = cap.get(cv2.CAP_PROP_FPS) or 30
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         output_file = os.path.join(output_dir, f"annotated_{file.filename}")
#         fourcc = cv2.VideoWriter_fourcc(*'avc1')
#         out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

#         summary_dict = defaultdict(lambda: {"count": 0, "avg_conf": 0})
#         zone_summary = [{"zone_name": z["name"], "total_count": 0} for z in zone_list]
#         total_people_detected = 0
#         total_frames = 0

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             total_frames += 1
#             results = model(frame)
#             result = results[0]
#             people_coords = []
#             frame_person_count = 0

#             for box in result.boxes:
#                 cls_id = int(box.cls[0])
#                 label = model.names[cls_id]
#                 conf = float(box.conf[0]) * 100

#                 if label == "person":
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
#                     people_coords.append((cx, cy))
#                     frame_person_count += 1
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

#                 summary_dict[label]["count"] += 1
#                 summary_dict[label]["avg_conf"] += conf

#             total_people_detected += frame_person_count

#             cv2.putText(frame, f"Total People: {frame_person_count}", (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

#             for i, zone in enumerate(zone_list):
#                 zx1, zy1 = zone["top_left"]["x"], zone["top_left"]["y"]
#                 zx2, zy2 = zone["bottom_right"]["x"], zone["bottom_right"]["y"]
#                 name = zone.get("name", f"Zone {i+1}")
#                 color = (255, 0, 0)
#                 cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, 2)
#                 count_in_zone = sum(zx1 < x < zx2 and zy1 < y < zy2 for (x, y) in people_coords)
#                 zone_summary[i]["total_count"] += count_in_zone

#                 label_text = f"{name}: {count_in_zone}"
#                 text_x, text_y = zx1 + 5, zy1 - 10 if zy1 - 10 > 20 else zy1 + 20
#                 cv2.putText(frame, label_text, (text_x, text_y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#             out.write(frame)

#         cap.release()
#         out.release()

#         # --- Frame Density (average across frames) ---
#         frame_density = round((total_people_detected / total_frames) / (width * height), 8)

#         # --- Zone Density ---
#         zone_density_list = []
#         for i, zone in enumerate(zone_list):
#             zx1, zy1 = zone["top_left"]["x"], zone["top_left"]["y"]
#             zx2, zy2 = zone["bottom_right"]["x"], zone["bottom_right"]["y"]
#             name = zone.get("name", f"Zone {i+1}")
#             area = (zx2 - zx1) * (zy2 - zy1)
#             total_count = zone_summary[i]["total_count"]
#             avg_count = total_count / total_frames if total_frames > 0 else 0
#             density = round(avg_count / area, 8) if area > 0 else 0
#             zone_density_list.append({"zone_name": name, "zone_density": density})

#         # --- Summary for header ---
#         summary = [
#             {
#                 "object": obj,
#                 "count": data["count"],
#                 "avg_confidence": round(data["avg_conf"] / data["count"], 2)
#             }
#             for obj, data in summary_dict.items()
#         ]

#         end_time = time.time()
#         processing_time = round(end_time - start_time, 2)

#         response = FileResponse(output_file, media_type="video/mp4")
#         response.headers["X-Processing-Time"] = str(processing_time)
#         response.headers["X-Detection-Summary"] = json.dumps(summary)
#         response.headers["X-Zone-Summary"] = json.dumps(zone_summary)
#         response.headers["X-Frame-Density"] = str(frame_density)
#         if zone_density_list:
#             response.headers["X-Zone-Density"] = json.dumps(zone_density_list)
#         response.headers["Access-Control-Expose-Headers"] = (
#             "X-Detection-Summary, X-Zone-Summary, X-Frame-Density, X-Zone-Density, X-Processing-Time"
#         )
#         return response

#     else:
#         raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")


@app.post("/new/upload")
async def detect_media(
    file: UploadFile = File(...),
    zones: Optional[str] = Form(None)
):
    start_time = time.time()
    upload_dir = "temp/uploads"
    output_dir = "temp/output"

    # Reset temp directories
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Save the uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Determine file type
    file_extension = file.filename.lower().split('.')[-1]
    image_ext = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp']
    video_ext = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm']

    # Parse zone JSON (if present)
    zone_list = []
    if zones:
        try:
            zone_list = json.loads(zones)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid zone JSON format")

    # -----------------------------------------------------------
    # IMAGE HANDLING
    # -----------------------------------------------------------
    if file_extension in image_ext:
        results = model(file_path)
        result = results[0]
        height, width = result.orig_img.shape[:2]

        people_coords = []
        detections = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0]) * 100
            label = model.names[cls_id]

            # Only consider "person" for density
            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                people_coords.append((cx, cy))

            detections.append({"object": label, "confidence": round(conf, 2)})

        total_people = len(people_coords)
        frame_density = round(total_people / (width * height), 8)

        # --- Zone Density Calculation ---
        zone_density_list = []
        for i, zone in enumerate(zone_list):
            zx1, zy1 = zone["top_left"]["x"], zone["top_left"]["y"]
            zx2, zy2 = zone["bottom_right"]["x"], zone["bottom_right"]["y"]
            name = zone.get("name", f"Zone {i+1}")
            area = max((zx2 - zx1) * (zy2 - zy1), 1)  # avoid divide by zero

            count_in_zone = sum(zx1 < x < zx2 and zy1 < y < zy2 for (x, y) in people_coords)
            density = round(count_in_zone / area, 8)
            zone_density_list.append({"zone_name": name, "zone_density": density})

        # --- Object Summary ---
        summary_dict = defaultdict(lambda: {"count": 0, "avg_conf": 0})
        for det in detections:
            obj = det["object"]
            summary_dict[obj]["count"] += 1
            summary_dict[obj]["avg_conf"] += det["confidence"]

        summary = [
            {
                "object": obj,
                "count": data["count"],
                "avg_confidence": round(data["avg_conf"] / data["count"], 2)
            }
            for obj, data in summary_dict.items()
        ]

        annotated = result.plot()
        output_file = os.path.join(output_dir, f"annotated_{file.filename}")
        cv2.imwrite(output_file, annotated)

        end_time = time.time()
        processing_time = round(end_time - start_time, 2)

        response = FileResponse(output_file, media_type="image/jpeg")
        response.headers["X-Detection-Summary"] = json.dumps(summary)
        response.headers["X-Processing-Time"] = str(processing_time)
        response.headers["X-Frame-Density"] = str(frame_density)
        if zone_density_list:
            response.headers["X-Zone-Density"] = json.dumps(zone_density_list)
        response.headers["Access-Control-Expose-Headers"] = (
            "X-Detection-Summary, X-Processing-Time, X-Frame-Density, X-Zone-Density"
        )
        return response

    # -----------------------------------------------------------
    # VIDEO HANDLING
    # -----------------------------------------------------------
    elif file_extension in video_ext:
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_file = os.path.join(output_dir, f"annotated_{file.filename}")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        summary_dict = defaultdict(lambda: {"count": 0, "avg_conf": 0})
        zone_summary = [{"zone_name": z["name"], "total_count": 0} for z in zone_list]

        total_people_detected = 0
        total_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            results = model(frame)
            result = results[0]
            people_coords = []
            frame_person_count = 0

            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0]) * 100

                if label == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    people_coords.append((cx, cy))
                    frame_person_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                summary_dict[label]["count"] += 1
                summary_dict[label]["avg_conf"] += conf

            total_people_detected += frame_person_count

            # Draw total count on frame
            cv2.putText(frame, f"People: {frame_person_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            # Draw and count per zone
            for i, zone in enumerate(zone_list):
                zx1, zy1 = zone["top_left"]["x"], zone["top_left"]["y"]
                zx2, zy2 = zone["bottom_right"]["x"], zone["bottom_right"]["y"]
                name = zone.get("name", f"Zone {i+1}")
                color = (255, 0, 0)
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, 2)
                count_in_zone = sum(zx1 < x < zx2 and zy1 < y < zy2 for (x, y) in people_coords)
                zone_summary[i]["total_count"] += count_in_zone

                label_text = f"{name}: {count_in_zone}"
                cv2.putText(frame, label_text, (zx1 + 5, zy1 - 10 if zy1 - 10 > 20 else zy1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            out.write(frame)

        cap.release()
        out.release()

        # --- Frame Density (average per frame) ---
        avg_people_per_frame = total_people_detected / total_frames if total_frames > 0 else 0
        frame_density = round(avg_people_per_frame / (width * height), 8)

        # --- Zone Density ---
        zone_density_list = []
        for i, zone in enumerate(zone_list):
            zx1, zy1 = zone["top_left"]["x"], zone["top_left"]["y"]
            zx2, zy2 = zone["bottom_right"]["x"], zone["bottom_right"]["y"]
            name = zone.get("name", f"Zone {i+1}")
            area = max((zx2 - zx1) * (zy2 - zy1), 1)
            avg_count = zone_summary[i]["total_count"] / total_frames if total_frames > 0 else 0
            density = round(avg_count / area, 8)
            zone_density_list.append({"zone_name": name, "zone_density": density})

        summary = [
            {
                "object": obj,
                "count": data["count"],
                "avg_confidence": round(data["avg_conf"] / data["count"], 2)
            }
            for obj, data in summary_dict.items()
        ]

        end_time = time.time()
        processing_time = round(end_time - start_time, 2)

        response = FileResponse(output_file, media_type="video/mp4")
        response.headers["X-Processing-Time"] = str(processing_time)
        response.headers["X-Detection-Summary"] = json.dumps(summary)
        response.headers["X-Frame-Density"] = str(frame_density)
        if zone_density_list:
            response.headers["X-Zone-Density"] = json.dumps(zone_density_list)
        response.headers["Access-Control-Expose-Headers"] = (
            "X-Detection-Summary, X-Frame-Density, X-Zone-Density, X-Processing-Time"
        )
        return response

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")






