from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from typing import List, Dict, Optional
import json
from FileProcess import FileProcessor
from pathlib import Path
from pydantic import BaseModel

app = FastAPI()
file_processor = FileProcessor()
jobs = {}

RESULTS_DIR = Path("results")

with open("toolConfig.json") as f:
    toolConfig = json.load(f)
    
class FileMetadata(BaseModel):
    requirementName: str
    multiple: bool
    index: int = None
    
# server.py
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from typing import List
import json
from FileProcess import FileProcessor
from pathlib import Path

app = FastAPI()
file_processor = FileProcessor()

# Get tool config
with open("toolConfig.json") as f:
    config = json.load(f)

@app.post("/api/process/{tool_id}")
async def upload_files(
    tool_id: str,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):   
    print("frontend fetch recieved")
    try:
        # Create new job
        job_id = file_processor.create_job()
        
        print(f"created job: {job_id}")
            
        tool_config = next((tool for tool in config["tools"] if tool["id"] == tool_id), None)
        if not tool_config:
            raise HTTPException(status_code=400, detail="Invalid tool ID")
        
        print(f"saving for {tool_id}")
        
        # Validate files against requirements
        requirements = tool_config["requirements"]
        
        # Save files and organize by requirement type
        organized_files = {}
        for file in files:
            # Find matching requirement based on file extension
            file_ext = Path(file.filename).suffix
            requirement = next((req for req in requirements if req["type"] == file_ext), None)
            
            if not requirement:
                continue  # Skip files that don't match any requirement
                
            req_name = requirement["name"]
            if req_name not in organized_files:
                organized_files[req_name] = []
                
            # Save file
            file_path = await file_processor.save_uploaded_file(file, job_id)
            organized_files[req_name].append({
                'path': file_path,
                'original_name': file.filename
            })
            
        # Validate requirements
        for req in requirements:
            if req["required"]:
                if req["name"] not in organized_files:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Missing required files for: {req['name']}"
                    )
                    
                if not req["multiple"] and len(organized_files[req["name"]]) > 1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multiple files provided for single-file requirement: {req['name']}"
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

@app.get("/api/results/{tool_id}/{job_id}")
async def get_results(tool_id: str, job_id: str):
    try:
        # Check if job exists and is complete
        job_status = file_processor.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}"
            )
            
        if job_status["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not complete. Current status: {job_status['status']}"
            )
            
        # Get tool configuration
        if tool_id not in toolConfig["tools"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tool ID: {tool_id}"
            )
            
        # Get results directory for this job
        job_results_dir = RESULTS_DIR / tool_id / job_id
        
        if not job_results_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Results not found for job: {job_id}"
            )
            
        # Read and return results based on tool type
        if tool_id == "spg":
            # Example for SPG tool
            results = {
                "Voronoi": [],
                "Nuclei_to_edge": []
            }
            
            # Load Voronoi results
            voronoi_dir = job_results_dir / "voronoi"
            if voronoi_dir.exists():
                results["Voronoi"] = [
                    {
                        "id": i,
                        "url": f"/results/{tool_id}/{job_id}/voronoi/{f.name}",
                        "title": f.stem
                    }
                    for i, f in enumerate(voronoi_dir.glob("*.jpg"), 1)
                ]
                
            # Load Nuclei results
            nuclei_dir = job_results_dir / "nuclei"
            if nuclei_dir.exists():
                results["Nuclei_to_edge"] = [
                    {
                        "id": i,
                        "url": f"/results/{tool_id}/{job_id}/nuclei/{f.name}",
                        "title": f.stem
                    }
                    for i, f in enumerate(nuclei_dir.glob("*.png"), 1)
                ]
                
            return results
            
        elif tool_id == "gel":
            # Example for Gel tool
            results_file = job_results_dir / "analysis.json"
            if results_file.exists():
                return json.loads(results_file.read_text())
                
        # Add other tool types as needed
        
        raise HTTPException(
            status_code=400,
            detail=f"Results handling not implemented for tool: {tool_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{tool_id}/{job_id}/{category}/{filename}")
async def serve_result_file(tool_id: str, job_id: str, category: str, filename: str):
    file_path = RESULTS_DIR / tool_id / job_id / category / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

def main():
    return True