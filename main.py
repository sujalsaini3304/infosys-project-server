from fastapi.staticfiles import StaticFiles
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
    expose_headers=["X-Detection-Results", "X-Detection-Summary"]  
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

    # 🔐 Check if the user already exists
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






