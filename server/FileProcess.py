import os
import uuid
from pathlib import Path
from datetime import datetime
import asyncio
from typing import List, Dict, Any
import aiofiles
import logging
from fastapi import UploadFile
import shutil

from stack_czi import process_stack_czi_files
from spg import process_spg_files
from gel import process_gel_files
from muscle import process_muscle_files
from cell_border import process_cell_border_files

logger = logging.getLogger(__name__)

class FileProcessor:
    def __init__(self):
        self.BASE_DIR = Path("data")
        self.UPLOAD_DIR = self.BASE_DIR / "uploads"
        self.RESULTS_DIR = self.BASE_DIR / "results"
        self.jobs: Dict[str, Dict[str, Any]] = {}
        
        # Ensure directories exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_job_upload_dir(self, job_id: str) -> Path:
        job_upload_dir = self.UPLOAD_DIR / job_id
        job_upload_dir.mkdir(exist_ok=True)
        return job_upload_dir

    def _get_job_results_dir(self, job_id: str) -> Path:
        job_results_dir = self.RESULTS_DIR / job_id
        job_results_dir.mkdir(exist_ok=True)
        return job_results_dir

    async def save_uploaded_file(self, file: UploadFile, job_id: str) -> Path:
        upload_dir = self._get_job_upload_dir(job_id)
        file_path = upload_dir / file.filename

        async with aiofiles.open(file_path, 'wb') as out_file:
            while content := await file.read(1024 * 1024):  # 1MB chunks
                await out_file.write(content)
        return file_path

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "status": "created",
            "created_at": datetime.utcnow(),
            "progress": 0,
            "message": "Job created"
        }
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        return self.jobs.get(job_id, {
            "status": "not_found",
            "message": "Job not found"
        })

    async def process_files(self, job_id: str, tool_id: str, file_paths: List[Path]) -> None:
        try:
            # Get result directory
            results_dir = self._get_job_results_dir(job_id)
            
            # Update status
            self.jobs[job_id].update({
                "status": "processing",
                "progress": 0,
                "message": "Starting processing..."
            })

            # Process based on tool_id
            if tool_id == "stack_czi":
                result_files = await process_czi_files(file_paths, results_dir, self.jobs[job_id])
            elif tool_id == "spg":
                result_files = await process_spg_files(file_paths, results_dir, self.jobs[job_id])
            elif tool_id == "gel":
                result_files = await process_gel_files(file_paths, results_dir, self.jobs[job_id])
            else:
                raise ValueError(f"Unknown tool_id: {tool_id}")

            # Update job status to completed
            self.jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Processing completed",
                "result_files": result_files
            })

        except Exception as e:
            logger.error(f"Error processing files for job {job_id}: {str(e)}")
            self.jobs[job_id].update({
                "status": "failed",
                "message": f"Processing failed: {str(e)}"
            })

        finally:
            # Clean up upload directory
            upload_dir = self._get_job_upload_dir(job_id)
            if upload_dir.exists():
                shutil.rmtree(upload_dir)