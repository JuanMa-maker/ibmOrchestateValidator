import requests
import re
import io
import pypdf as pdf_reader
import ibm_watsonx_orchestrate.agent_builder.tools as IBM_tools

# @IBM_tools()
def pdf_checker(url_file: str):
    try:
        connection = requests.get(url_file)
        connection.raise_for_status()
        reader=pdf_reader.PdfReader(io.BytesIO(connection.content))
        page=reader.pages[0]
        content=page.extract_text()
        
        data={}
        name_pattern = r"NOMBRE DEL (?:ASUGURADO|ASEGURADO):\s*(.*?)\s*CURP:" 
        name_match=re.search(name_pattern,content,re.DOTALL)

        if name_match:
            data['pacient_name']=name_match.group(1).strip()
        else:
            data['pacient_name']="Not found"
        date_pattern = r"(INICIAL|CONTINUACION)\s+(?:[A-Z\s]+)\s+(\d+)\s+(\d{1,2}/\d{1,2}/\d{2,4})"
        date_match = re.search(date_pattern, content, re.IGNORECASE)

        if date_match:
            data['authorized_days']=int(date_match.group(2))
            data['expedition_date']=date_match.group(3)
        else:
            data['authorized_days']=0
            data['expedition_date']="Not found"
        
        return{
            "status": "success",
            "data": data
        }


    except Exception as e:
        return {"status:": "error",
                "message": f"Error message: {e}"}
    
x="https://mdhaybgsjcpolwpjwudt.supabase.co/storage/v1/object/sign/prescription/622180827-incapacidad-imss-editable.pdf?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV85N2M2NGY3Zi0wNjJiLTQ3Y2ItYmVhYS0zNjA3NjE1ZTI5YzIiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJwcmVzY3JpcHRpb24vNjIyMTgwODI3LWluY2FwYWNpZGFkLWltc3MtZWRpdGFibGUucGRmIiwiaWF0IjoxNzYzODY2MDY0LCJleHAiOjE3NjQ0NzA4NjR9.lCghr7R_ecvhwBHGR0Ypb-EqX1Jursks9v9gCv5vX0U"
y="https://mdhaybgsjcpolwpjwudt.supabase.co/storage/v1/object/sign/prescription/434704698-371736047-Justificante-Incapacidad-Imss-Autoguardado.pdf?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV85N2M2NGY3Zi0wNjJiLTQ3Y2ItYmVhYS0zNjA3NjE1ZTI5YzIiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJwcmVzY3JpcHRpb24vNDM0NzA0Njk4LTM3MTczNjA0Ny1KdXN0aWZpY2FudGUtSW5jYXBhY2lkYWQtSW1zcy1BdXRvZ3VhcmRhZG8ucGRmIiwiaWF0IjoxNzYzODY3MzMzLCJleHAiOjE3NjQ0NzIxMzN9.p0mJJn2mHVpTN4E0-AlkQMMUM-XGjANfAjQEwtikUB8"
print(pdf_checker(y))