from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import json
from FileProcess import FileProcessor
from pathlib import Path
from pydantic import BaseModel

app = FastAPI()
file_processor = FileProcessor()
jobs = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path("results")
UPLOAD_DIR = Path("uploads")

# Load tool configuration
with open("toolConfig.json") as f:
    config = json.load(f)

class FileMetadata(BaseModel):
    requirementName: str
    multiple: bool
    index: int = None

class UploadResponse(BaseModel):
    upload_id: str
    file_paths: Dict[str, List[str]]
    message: str

class ProcessResponse(BaseModel):
    job_id: str
    message: str

class ProcessRequest(BaseModel):
    upload_id: str
    tool_id: str

# 1. File Upload Endpoint
@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
) -> UploadResponse:
    try:
        # Generate upload ID
        upload_id = file_processor.create_upload_id()
        upload_dir = UPLOAD_DIR / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save files and track their paths
        file_paths = {}
        for file in files:
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Organize by file extension
            ext = file_path.suffix
            if ext not in file_paths:
                file_paths[ext] = []
            file_paths[ext].append(str(file_path))

        return UploadResponse(
            upload_id=upload_id,
            file_paths=file_paths,
            message="Files uploaded successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. Process Files Endpoint
@app.post("/api/process")
async def process_files(
    request: ProcessRequest,
    background_tasks: BackgroundTasks
) -> ProcessResponse:
    try:
        tool_id = request.tool_id
        upload_id = request.upload_id

        # Validate tool exists
        tool_config = next((tool for tool in config["tools"] if tool["id"] == tool_id), None)
        if not tool_config:
            raise HTTPException(status_code=400, detail="Invalid tool ID")

        # Create job
        job_id = file_processor.create_job()
        
        # Organize uploaded files by requirement
        upload_dir = UPLOAD_DIR / upload_id
        if not upload_dir.exists():
            raise HTTPException(status_code=404, detail="Upload not found")

        # Match files to requirements
        organized_files = {}
        requirements = tool_config["requirements"]
        
        for requirement in requirements:
            req_name = requirement["name"]
            req_type = requirement["type"]
            
            matching_files = list(upload_dir.glob(f"*{req_type}"))
            
            if requirement["required"] and not matching_files:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required files for: {req_name}"
                )
                
            if not requirement["multiple"] and len(matching_files) > 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Multiple files provided for single-file requirement: {req_name}"
                )
                
            organized_files[req_name] = [
                {
                    'path': str(f),
                    'original_name': f.name
                }
                for f in matching_files
            ]

        # Start processing in background
        background_tasks.add_task(
            file_processor.process_files,
            job_id,
            tool_id,
            organized_files
        )

        return ProcessResponse(
            job_id=job_id,
            message="Processing started"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Get Results Endpoint
@app.get("/api/results/{tool_id}/{job_id}")
async def get_results(tool_id: str, job_id: str):
    try:
        # Check job status
        job_status = file_processor.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
            
        if job_status["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not complete. Current status: {job_status['status']}"
            )

        # Get results path
        results_dir = RESULTS_DIR / tool_id / job_id
        if not results_dir.exists():
            raise HTTPException(status_code=404, detail=f"Results not found")

        # Return tool-specific results
        tool_results = file_processor.get_tool_results(tool_id, job_id, results_dir)
        
        return {
            "status": "completed",
            "results": tool_results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Status check endpoint remains the same
@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    return file_processor.get_job_status(job_id)

# Serve result files
@app.get("/results/{tool_id}/{job_id}/{category}/{filename}")
async def serve_result_file(tool_id: str, job_id: str, category: str, filename: str):
    file_path = RESULTS_DIR / tool_id / job_id / category / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)