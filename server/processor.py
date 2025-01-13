# processors.py
from pathlib import Path
from typing import Dict, Any
import json
import logging
import numpy as np
import cv2
from aicsimageio import AICSImage
import tifffile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def im_adjust(I, thres=[1, 99, True], autoscale=None):
    """Adjust image intensity"""
    if thres[2]:
        I_low, I_high = np.percentile(I.reshape(-1), thres[:2])
    else:
        I_low, I_high = thres[0], thres[1]
    
    I[I > I_high] = I_high
    I[I < I_low] = I_low
    
    if autoscale is not None:
        I = (I.astype(float) - I_low) / (I_high - I_low)
        if autoscale == "uint8":
            I = (I * 255).astype(np.uint8)
    return I

async def process_stack_czi(upload_path: str, result_path: str, job_id: str) -> Dict[str, Any]:
    """
    Process stack CZI files
    Args:
        upload_path: Path to uploaded files
        result_path: Path to store results
        job_id: Unique job identifier
    Returns:
        Dict containing processing results and metadata
    """
    try:
        logger.info(f"Starting CZI processing for job {job_id}")
        upload_dir = Path(upload_path)
        result_dir = Path(result_path)
        result_dir.mkdir(parents=True, exist_ok=True)

        processed_files = []
        metadata = {
            "job_id": job_id,
            "tool": "stack_czi",
            "processed_files": []
        }

        # Process all CZI files in upload directory
        for czi_file in upload_dir.glob("**/*.czi"):
            try:
                logger.info(f"Processing file: {czi_file.name}")
                
                # Load and process the image
                img = AICSImage(str(czi_file))
                
                # Process each channel
                channel_img = []
                for c in range(img.shape[1]):
                    # Get image data for channel
                    image = img.get_image_data("ZYX", C=c)
                    
                    # Maximum intensity projection
                    image = np.max(image, axis=0)
                    
                    # Rotate image
                    channel_img.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
                
                # Stack channels
                stacked_image = np.stack(channel_img, axis=-1)
                
                # Adjust image
                uint8_stacked = im_adjust(stacked_image, autoscale='uint8')
                
                # Generate output filename
                base_filename = czi_file.stem
                tiff_file_path = result_dir / f'{base_filename}_stacked.tiff'
                
                # Save as TIFF
                tifffile.imwrite(str(tiff_file_path), uint8_stacked)
                logger.info(f"Saved processed file: {tiff_file_path}")
                
                # Add to processed files list
                file_info = {
                    'original_name': czi_file.name,
                    'processed_name': tiff_file_path.name,
                    'channels': img.shape[1],
                    'path': str(tiff_file_path.relative_to(result_dir))
                }
                processed_files.append(file_info)
                metadata["processed_files"].append(file_info)

            except Exception as e:
                logger.error(f"Error processing file {czi_file.name}: {str(e)}")
                continue

        # Save metadata
        with (result_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "status": "completed",
            "files": processed_files,
            "metadata": metadata
        }

    except Exception as e:
        logger.error(f"Error in process_stack_czi: {str(e)}")
        raise

async def process_spg(upload_path: str, result_path: str, job_id: str) -> Dict[str, Any]:
    """Process SPG analysis"""
    # Implementation for SPG analysis would go here
    pass

async def process_gel(upload_path: str, result_path: str, job_id: str) -> Dict[str, Any]:
    """Process gel analysis"""
    # Implementation for gel analysis would go here
    pass

async def process_muscle(upload_path: str, result_path: str, job_id: str) -> Dict[str, Any]:
    """Process muscle analysis"""
    # Implementation for muscle analysis would go here
    pass