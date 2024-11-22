from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from FileProcess import FileProcessor

app = FastAPI()
file_processor = FileProcessor()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/process/{tool_id}")
async def upload_files(
    tool_id: str,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks
):
    try:
        # Create new job
        job_id = file_processor.create_job()
        
        # Save all files
        file_paths = []
        for file in files:
            file_path = await file_processor.save_uploaded_file(file, job_id)
            file_paths.append(file_path)
        
        # Start processing in background
        background_tasks.add_task(
            file_processor.process_files,
            job_id,
            tool_id,
            file_paths
        )
        
        return {
            "jobId": job_id,
            "message": "Files uploaded successfully, processing started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    return file_processor.get_job_status(job_id)