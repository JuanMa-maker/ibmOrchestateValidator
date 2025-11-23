import requests
import re
import io
import pypdf as pdf_reader
from datetime import datetime, timedelta

headers_reales = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive"
}

def pdf_checker(url_file: str):
    try:
        connection = requests.get(url_file, headers=headers_reales, timeout=10)
        connection.raise_for_status()
        reader = pdf_reader.PdfReader(io.BytesIO(connection.content))
        
        page = reader.pages[0]
        content = page.extract_text()
        
        data = {}

        date_pattern = r"(INICIAL|CONTINUACION)\s+(?:[A-Z\s]+)\s+(\d+)\s+(\d{1,2}/\d{1,2}/\d{2,4})"
        date_match = re.search(date_pattern, content, re.IGNORECASE)

        if date_match:
            days_int = int(date_match.group(2))      
            start_date_str = date_match.group(3)     

            start_date_obj = datetime.strptime(start_date_str, "%d/%m/%Y")
            end_date_obj = start_date_obj + timedelta(days=days_int-1)
        
            data['end_date'] = end_date_obj.strftime("%Y-%m-%d")
            data['authorized_days'] = days_int
            
        else:
            data['authorized_days'] = 0
            data['end_date'] = None 
        
        return data

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None
