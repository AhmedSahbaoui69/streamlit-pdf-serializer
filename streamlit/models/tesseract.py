from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from langchain_ollama import ChatOllama
import re
import json

class TesseractOCR():
  def run(self, file_path):
    if file_path.lower().endswith(('.pdf')):
      first_page = convert_from_path(file_path, first_page=1, last_page=1)
      image_path = "./temp/temp_image.jpg"
      first_page[0].save(image_path, "JPEG")
    else:
      image_path = file_path

    text = pytesseract.image_to_string(Image.open(image_path))
    # Clean up the text
    text = re.sub(r'\n+', '\n', text)
    text = "\n".join(line.strip() for line in text.splitlines())
    print(text)

    yield {"progress": "Running Ollama Extraction", "output": ""}

    # Initialize ChatOllama
    llm = ChatOllama(
      model="hf.co/bartowski/NuExtract-v1.5-GGUF:Q4_K_S",
      temperature=0.3
    )

    # Define the prompt template
    system_message = """
    Please follow the template provided by the user and extract only the fields listed within the template. 
    Respect the order of the fields and do not add or remove anything. 
    If any field is unknown or cannot be determined, put "N/A". 
    Do not hallucinate or make assumptions beyond what is provided in the input text. 
    Ensure that the output is strictly in JSON format without any comments or additional text.
    """

    template = """
{
  "invoice": {
  "invoice_number": "",
  "invoice_date": "",
  "po_number": "",
  "billing_period": "",
  "due_date": "",
  "total_amount_due": {
    "amount": "",
    "currency": ""
  },
  "sold_to": {
    "name": "",
    "contact": "",
    "customer_number": "",
    "city": "",
    "phone": ""
  },
  "end_customer": {
    "name": "",
    "address": "",
    "city": "",
    "postal_code": ""
  },
  "payment_instructions": {
    "account_name": "",
    "bank_name": "",
    "bank_account_number": "",
    "swift_code": "",
    "iban_number": "",
    "email": "",
    "payment_method": ""
  }
  }
}

### Text:
    """

    # Create the prompt
    messages = [
      ("system", system_message),
      ("human", template + text),
    ]

    # Run the chain with the messages
    response = llm.invoke(messages)
    print(response)
    output = re.sub(r'^###.*$', '', response.content, flags=re.MULTILINE)
    output = re.sub(r'^.*### Template.*$', '', output, flags=re.MULTILINE)
    try:
      formatted_output = json.loads(output)
      output = json.dumps(formatted_output, indent=4)
    except json.JSONDecodeError:
      print("Invalid JSON response from LLM")

    yield {"progress": "Ollama Extraction Completed", "output": output}
