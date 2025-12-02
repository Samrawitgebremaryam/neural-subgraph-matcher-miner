import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from ..services.mining_service import MiningService
from ..config.settings import Config

router = APIRouter()

@router.post("/mine")
def mine(
    graph_file: UploadFile = File(...), 
    job_id: str = Form(None)
):
    # Validate file
    if not graph_file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    # Save the uploaded file
    filename = "{}.pkl".format(uuid.uuid4())
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(graph_file.file, buffer)
            
        # Run miner with job_id and parameters
        result = MiningService.run_miner(
            filepath, 
            job_id=job_id
        )

        # Construct response
        response = {
            'job_id': result['job_id'],
            'results_path': result['results_path'],
            'plots_path': result['plots_path'],
            'status': 'success'
        }
        
        return JSONResponse(content=response)

    except Exception as e:
        print("Error: {}".format(str(e)), flush=True)
        return JSONResponse(status_code=500, content={'error': str(e)})
    finally:
        # Cleanup input file
        if os.path.exists(filepath):
            os.remove(filepath)
