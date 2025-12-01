import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ..services.mining_service import MiningService
from ..config.settings import Config

router = APIRouter()

@router.post("/mine")
def mine(graph_file: UploadFile = File(...)):
    # Validate file
    if not graph_file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    # Save the uploaded file
    filename = "{}.pkl".format(uuid.uuid4())
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(graph_file.file, buffer)
            
        mining_results = MiningService.run_miner(filepath)

        # Construct response matching MinerService expectations
        response = {
            'motifs': mining_results,
            'statistics': {
                'count': len(mining_results),
                'status': 'success'
            }
        }
        
        return JSONResponse(content=response)

    except Exception as e:
        print("Error: {}".format(str(e)))
        return JSONResponse(status_code=500, content={'error': str(e)})
    finally:
        # Cleanup input file
        if os.path.exists(filepath):
            os.remove(filepath)
