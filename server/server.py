from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from typing import List, Dict, Optional
import json
from FileProcess import FileProcessor

app = FastAPI()
file_processor = FileProcessor()

@app.post("/api/process/{tool_id}")
async def upload_files(
    files: List[UploadFile] = File(...),
    tool_id: str = None,
    background_tasks: Optional[BackgroundTasks] = None
):
    try:
        # Create new job
        job_id = file_processor.create_job()
        
        # Parse tool configuration
        tool_config = json.loads(toolConfig)
        requirements = {req["name"]: req for req in tool_config["requirements"]}
        
        # Organize files by requirement
        organized_files: Dict[str, List[Dict]] = {}
        
        # Process each uploaded file
        for file in files:
            metadata_key = f"metadata_{file.filename}"
            metadata = json.loads(toolConfig)  # Using toolConfig as it contains file metadata
            requirement_name = metadata["requirementName"]
            
            if requirement_name not in organized_files:
                organized_files[requirement_name] = []
            
            # Save file
            file_path = await file_processor.save_uploaded_file(file, job_id)
            
            organized_files[requirement_name].append({
                'path': file_path,
                'metadata': metadata,
                'original_name': file.filename
            })

        # Validate requirements
        for req_name, req_info in requirements.items():
            if req_name not in organized_files:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required files for: {req_name}"
                )
            if not req_info["multiple"] and len(organized_files[req_name]) > 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Multiple files provided for single-file requirement: {req_name}"
                )

        # Start processing in background
        background_tasks.add_task(
            file_processor.process_files,
            job_id,
            tool_id,
            organized_files
        )
        
        return {
            "jobId": job_id,
            "message": "Files uploaded successfully, processing started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    return file_processor.get_job_status(job_id)