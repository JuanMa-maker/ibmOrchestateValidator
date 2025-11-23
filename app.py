from fastapi import FastAPI, HTTPException,BackgroundTasks
import resend

from pydantic import BaseModel,EmailStr
import db_manager
import PlainTextDetector
from typing import Optional
import os

app = FastAPI()

resend.api_key = "re_hPyuqD2m_He6N7X3aJRoWaRWMFGHC5moU"

class ProcessRequest(BaseModel):
    record_id: int
    
class RequestModel(BaseModel):
    id: str
    type: str            # Ej: "Vacaciones"
    name: str
    email: EmailStr      # El correo del empleado
    status: str          # Ej: "APROBADA"
    
    # Campos opcionales basados en tu JSON anterior
    reason: Optional[str] = "Sin observaciones"
    days: Optional[int] = 0
    entrance: Optional[str] = "N/A"
    out: Optional[str] = "N/A"
    time: Optional[str] = None


@app.post("/api/process_record")
async def process_record_endpoint(request: ProcessRequest):
    """
    Main Endpoint called by the AI Orchestrator.
    Receives: {"record_id": 123}
    Returns: JSON with status and extracted data.
    """
    rec_id = request.record_id
    
    pdf_url = db_manager.get_pdf_url(rec_id)
    
    if not pdf_url:
        raise HTTPException(status_code=404, detail="PDF URL not found for this Record ID.")
    
    extracted_data = PlainTextDetector.pdf_checker(pdf_url)
    
    if not extracted_data:
        raise HTTPException(status_code=500, detail="Failed to download or read the PDF.")
        

    success = db_manager.update_disability_record(rec_id, extracted_data)
    
    if success:
        return {
            "status": "success",
            "message": "Record updated successfully.",
            "data": extracted_data
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to update database record.")


