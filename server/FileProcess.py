import os
import uuid
from pathlib import Path
from datetime import datetime
import asyncio
from typing import List, Dict, Any
import shutil
import json

from stack_czi import process_stack_czi_files

class FileProcessor:
    def __init__(self):
        self.BASE_DIR = Path("data")
        self.UPLOAD_DIR = Path("uploads")
        self.RESULTS_DIR = Path("results")
        self.jobs: Dict[str, Dict[str, Any]] = {}
        
        # Ensure directories exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def create_upload_id(self) -> str:
        """Create a unique ID for file uploads"""
        return str(uuid.uuid4())

    def create_job(self) -> str:
        """Create a new processing job and return its ID"""
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
        """Get the current status of a job"""
        return self.jobs.get(job_id, {
            "status": "not_found",
            "message": "Job not found"
        })

    async def process_files(self, job_id: str, tool_id: str, organized_files: Dict[str, List[Dict]]) -> None:
        """Process files for a specific tool"""
        try:
            results_dir = self.RESULTS_DIR / tool_id / job_id
            results_dir.mkdir(parents=True, exist_ok=True)
            
            self.jobs[job_id].update({
                "status": "processing",
                "progress": 0,
                "message": "Starting processing..."
            })

            # Process based on tool_id
            if tool_id == "stack_czi":
                await self._process_stack_czi(job_id, organized_files, results_dir)
            elif tool_id == "spg":
                await self._process_spg(job_id, organized_files, results_dir)
            # Add other tool processing methods as needed
            
            self.jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "message": "Processing completed"
            })

        except Exception as e:
            self.jobs[job_id].update({
                "status": "failed",
                "message": f"Processing failed: {str(e)}"
            })
            raise

    def get_tool_results(self, tool_id: str, job_id: str, results_dir: Path) -> Dict:
        """Get processed results for a specific tool"""
        if tool_id == "spg":
            return self._get_spg_results(results_dir)
        elif tool_id == "stack_czi":
            return self._get_stack_czi_results(results_dir)
        # Add other tool result handlers
        
        raise ValueError(f"Results handling not implemented for tool: {tool_id}")

    def _get_spg_results(self, results_dir: Path) -> Dict:
        """Get results for SPG analysis"""
        results = {
            "Voronoi": [],
            "Nuclei_to_edge": []
        }
        
        # Load Voronoi results
        voronoi_dir = results_dir / "voronoi"
        if voronoi_dir.exists():
            results["Voronoi"] = [
                {
                    "id": i,
                    "url": str(f.relative_to(results_dir)),
                    "title": f.stem
                }
                for i, f in enumerate(voronoi_dir.glob("*.jpg"), 1)
            ]
            
        # Load Nuclei results
        nuclei_dir = results_dir / "nuclei"
        if nuclei_dir.exists():
            results["Nuclei_to_edge"] = [
                {
                    "id": i,
                    "url": str(f.relative_to(results_dir)),
                    "title": f.stem
                }
                for i, f in enumerate(nuclei_dir.glob("*.png"), 1)
            ]
            
        return results

    # Add other private processing methods as needed
    async def _process_stack_czi(self, job_id: str, organized_files: Dict[str, List[Dict]], results_dir: Path):
        """Process CZI stacking"""
        return process_stack_czi_files(job_id, organized_files, results_dir)

    async def _process_spg(self, job_id: str, organized_files: Dict[str, List[Dict]], results_dir: Path):
        """Process SPG analysis"""
        pass