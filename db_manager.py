import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

url: str = os.environ.get("https://mdhaybgsjcpolwpjwudt.supabase.co")
key: str = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1kaGF5YmdzamNwb2x3cGp3dWR0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM3NjQ0NDcsImV4cCI6MjA3OTM0MDQ0N30.1u-FeN0Je3qbke4YtqiZJ6xXX4_jEnnP-1Cbau10Y14")

# url: str = os.environ.get("SUPABASE_URL")
# key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

def get_pdf_url(record_id: int):
    try:
        response = supabase.table("leave_requests")\
            .select("ref_pdf")\
            .eq("request_id", record_id)\
            .execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]['ref_pdf']
        return None
    except Exception as e:
        print(f"Supabase Error (Get URL): {e}")
        return None

def update_disability_record(record_id: int, data: dict):
    try:
        response = supabase.table("leave_requests")\
            .update({
                "approved_days": data.get("authorized_days"),
                "end_date": data.get("end_date")
            })\
            .eq("id", record_id)\
            .execute()
        
        return True
    except Exception as e:
        print(f"Supabase Error (Update): {e}")
        return False