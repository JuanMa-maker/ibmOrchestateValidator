import os
import resend
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Resend API Key
# It reads from .env, or uses the hardcoded one if .env fails (for testing)
resend.api_key = os.environ.get("RESEND_API_KEY", "re_hPyuqD2m_He6N7X3aJRoWaRWMFGHC5moU")

def send_email_notification(data):
    """
    Logic to build the HTML and send the email via Resend.
    Receives the Pydantic model 'data' from main.py.
    """
    print(f"🚀 Preparing Resend email for: {data.email}")

    try:
        # HTML Content (Copied from your provided snippet)
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
            # IMPORTANT: Use 'onboarding@resend.dev' if you haven't verified a domain yet.
            "from": "Agente RRHH <onboarding@resend.dev>",
            "to": [data.email],
            "subject": f"Actualización de Solicitud: {data.type} - {data.status}",
            "html": html_content,
        }

        email = resend.Emails.send(params)
        print(f"✅ Email sent successfully via Resend. ID: {email}")
        return True

    except Exception as e:
        print(f"❌ Error sending via Resend: {str(e)}")
        return False