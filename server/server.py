# routes.py
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import os
import shutil
from pathlib import Path

from processor import process_gel, process_muscle, process_spg, process_stack_czi

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directories
UPLOAD_DIR = Path("data/uploads")
RESULTS_DIR = Path("data/results")

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile],
    job_id: str = Form(...),
    tool_id: str = Form(...),
    requirement_names: List[str] = Form(...)
):
    try:
        # Create job directory for uploads
        upload_path = UPLOAD_DIR / job_id
        upload_path.mkdir(parents=True, exist_ok=True)
        
        # Create directories for each requirement
        file_paths = {}
        for req_name in requirement_names:
            req_dir = upload_path / req_name
            req_dir.mkdir(exist_ok=True)
            file_paths[req_name] = []

        # Save uploaded files to appropriate directories
        for file, req_name in zip(files, requirement_names):
            file_path = upload_path / req_name / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths[req_name].append(str(file_path))

        return JSONResponse({
            "status": "success",
            "job_id": job_id,
            "upload_path": str(upload_path),
            "file_paths": file_paths
        })

    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/api/process/{job_id}")
async def process_files(job_id: str, tool_id: str):
    try:
        # Get upload path
        upload_path = UPLOAD_DIR / job_id
        if not upload_path.exists():
            return JSONResponse({
                "status": "error",
                "message": "Upload directory not found"
            }, status_code=404)

        # Create results directory
        result_path = RESULTS_DIR / job_id
        result_path.mkdir(parents=True, exist_ok=True)

        # Process files based on tool_id
        # This will be replaced with actual tool-specific processing
        tool_processor = get_tool_processor(tool_id)
        if tool_processor:
            result = await tool_processor(
                upload_path=str(upload_path),
                result_path=str(result_path),
                job_id=job_id
            )
            
            return JSONResponse({
                "status": "completed",
                "job_id": job_id,
                "result_path": str(result_path),
                "results": result
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": f"No processor found for tool {tool_id}"
            }, status_code=400)

    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    try:
        result_path = RESULTS_DIR / job_id
        if not result_path.exists():
            return JSONResponse({
                "status": "error",
                "message": "Results not found"
            }, status_code=404)

        # Here you would implement logic to read and return the results
        # This is a placeholder that would be replaced with actual result reading
        results = {
            "status": "completed",
            "job_id": job_id,
            "result_path": str(result_path),
            "files": list(result_path.glob("**/*")),  # List all files in result directory
        }

        return JSONResponse(results)

    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

# Tool processor mapping (to be implemented)
def get_tool_processor(tool_id: str):
    # This would be replaced with actual tool processors
    tool_processors = {
        "stack_czi": process_stack_czi,
        "spg": process_spg,
        "gel": process_gel,
        "muscle": process_muscle
    }
    return tool_processors.get(tool_id)