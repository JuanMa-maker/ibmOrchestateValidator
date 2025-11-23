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




def run_agent_logic(data: RequestModel):
    print(f"🚀 Preparando envío por Resend a: {data.email}")

    try:
        # HTML del correo
        html_content = f"""
        <div style="font-family: Arial, sans-serif; color: #333;">
            <h2>Hola {data.name},</h2>
            <p>Te informamos sobre el estado de tu solicitud de <strong>{data.type}</strong>.</p>
            
            <div style="background-color: #f4f4f4; padding: 15px; border-radius: 5px;">
                <p><strong>Estado:</strong> <span style="color: {'green' if 'aprobada' in data.status.lower() else 'red'}; font-weight: bold;">{data.status}</span></p>
                <p><strong>Motivo/Detalles:</strong> {data.reason}</p>
                <p><strong>Días:</strong> {data.days}</p>
                <p><strong>Fecha inicio:</strong> {data.entrance}</p>
            </div>

            <p style="margin-top: 20px; font-size: 12px; color: #888;">
                Este es un mensaje automático generado por el Agente de RRHH de IBM.
            </p>
        </div>
        """

        params = {
            # IMPORTANTE: Si no tienes dominio propio verificado en Resend,
            # DEBES usar 'onboarding@resend.dev' como remitente para pruebas.
            "from": "Agente RRHH <onboarding@resend.dev>",
            "to": [data.email],
            "subject": f"Actualización de Solicitud: {data.type} - {data.status}",
            "html": html_content,
        }

        email = resend.Emails.send(params)
        print(f"✅ Correo enviado con éxito via Resend. ID: {email}")

    except Exception as e:
        print(f"❌ Error al enviar con Resend: {str(e)}")

@app.post("/ai-process-request")
async def ai_process_request(request: RequestModel, background_tasks: BackgroundTasks):
    
    # Validaciones rápidas
    if not request.email:
        raise HTTPException(status_code=400, detail="Falta el email")

    # Enviar tarea a segundo plano
    background_tasks.add_task(run_agent_logic, request)

    return {
        "status": "success", 
        "message": "Solicitud recibida. El agente enviará el correo via Resend."
    }