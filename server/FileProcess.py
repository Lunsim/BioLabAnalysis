# app/core/file_processor.py
import os
import uuid
from pathlib import Path
from datetime import datetime
import asyncio
from typing import List, Dict, Any
import aiofiles
import logging
from fastapi import UploadFile, BackgroundTasks
import shutil

logger = logging.getLogger(__name__)

class FileProcessor:
    def __init__(self):
        self.BASE_DIR = Path("data")
        self.UPLOAD_DIR = self.BASE_DIR / "uploads"
        self.RESULTS_DIR = self.BASE_DIR / "results"
        self.jobs: Dict[str, Dict[str, Any]] = {}
        
        # Ensure base directories exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_job_upload_dir(self, job_id: str) -> Path:
        """Get the upload directory for a specific job."""
        job_upload_dir = self.UPLOAD_DIR / job_id
        job_upload_dir.mkdir(exist_ok=True)
        return job_upload_dir

    def _get_job_results_dir(self, job_id: str) -> Path:
        """Get the results directory for a specific job."""
        job_results_dir = self.RESULTS_DIR / job_id
        job_results_dir.mkdir(exist_ok=True)
        return job_results_dir

    async def save_uploaded_file(self, file: UploadFile, job_id: str) -> Path:
        """Save uploaded file to job-specific directory."""
        # Get job-specific upload directory
        upload_dir = self._get_job_upload_dir(job_id)
        
        # Create safe filename
        safe_filename = file.filename.replace(' ', '_')
        file_path = upload_dir / safe_filename

        try:
            async with aiofiles.open(file_path, 'wb') as out_file:
                # Read and write in chunks to handle large files
                while content := await file.read(1024 * 1024):  # 1MB chunks
                    await out_file.write(content)
            return file_path
        except Exception as e:
            logger.error(f"Error saving file {safe_filename} for job {job_id}: {str(e)}")
            # Clean up if save fails
            if file_path.exists():
                file_path.unlink()
            raise

    async def process_czi_files(self, job_id: str, file_paths: List[Path]) -> None:
        """Process CZI files and generate results."""
        results_dir = self._get_job_results_dir(job_id)
        
        try:
            self.jobs[job_id].update({
                "status": "processing",
                "progress": 0,
                "message": "Starting processing..."
            })

            total_files = len(file_paths)
            processed_files = []

            for idx, file_path in enumerate(file_paths, 1):
                try:
                    # Your CZI processing logic here
                    # For example:
                    # result_path = results_dir / f"{file_path.stem}_processed.jpg"
                    # processed_image = process_czi_image(file_path, result_path)
                    # processed_files.append(result_path)
                    
                    # Update progress
                    progress = int((idx / total_files) * 100)
                    self.jobs[job_id].update({
                        "progress": progress,
                        "message": f"Processing file {idx} of {total_files}: {file_path.name}"
                    })
                    
                    # Simulate processing time (remove in production)
                    await asyncio.sleep(2)
                
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    raise

            # Update job status to completed
            self.jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Processing completed",
                "result_dir": str(results_dir),
                "processed_files": [str(p) for p in processed_files]
            })

        except Exception as e:
            logger.error(f"Error in job {job_id}: {str(e)}")
            self.jobs[job_id].update({
                "status": "failed",
                "message": f"Processing failed: {str(e)}"
            })
        
        finally:
            # Clean up uploaded files after processing
            try:
                upload_dir = self._get_job_upload_dir(job_id)
                if upload_dir.exists():
                    shutil.rmtree(upload_dir)
            except Exception as e:
                logger.error(f"Error cleaning up upload directory for job {job_id}: {str(e)}")

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old jobs and their files."""
        current_time = datetime.utcnow()
        for job_id, job in list(self.jobs.items()):
            if (current_time - job["created_at"]).total_seconds() > max_age_hours * 3600:
                try:
                    # Clean up upload directory
                    upload_dir = self._get_job_upload_dir(job_id)
                    if upload_dir.exists():
                        shutil.rmtree(upload_dir)
                    
                    # Clean up results directory
                    results_dir = self._get_job_results_dir(job_id)
                    if results_dir.exists():
                        shutil.rmtree(results_dir)
                    
                    # Remove job from memory
                    del self.jobs[job_id]
                except Exception as e:
                    logger.error(f"Error cleaning up job {job_id}: {str(e)}")

# FastAPI routes
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
            file_processor.process_czi_files,
            job_id,
            file_paths
        )
        
        return JSONResponse({
            "jobId": job_id,
            "message": "Files uploaded successfully, processing started"
        })
        
    except Exception as e:
        # Clean up job directory if creation fails
        try:
            upload_dir = file_processor._get_job_upload_dir(job_id)
            if upload_dir.exists():
                shutil.rmtree(upload_dir)
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))