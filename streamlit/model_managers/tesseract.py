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
      temperature=0.2
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
### Template
{
  "invoice": {
    "invoice_number": "",  // String (e.g., "INV12345")
    "invoice_date": "",  // Date (ISO 8601 format, e.g., "2025-02-09")
    "purchase_order_number": "",  // String (e.g., "PO123456")
    "billing_period": "",  // String (e.g., "01-Jan-2025 to 31-Jan-2025")
    "due_date": "",  // Date (ISO 8601 format, e.g., "2025-02-20")
    "total_amount_due": {
      "amount": "",  // Number (e.g., 1234.56)
      "currency": ""  // String (e.g., "USD", "EUR")
    },
    "deliver_to": {
      "company_name": "",  // String (e.g., "ABC Corp.")
      "contact": "",  // String (e.g., "John Doe")
      "customer_number": "",  // String or Number (e.g., "CUST123")
      "city": "",  // String (e.g., "New York")
      "postal_code": "",  // String (e.g., "10001")
      "phone": ""  // String (e.g., "+1-800-555-5555")
    },
    "sold_to": {
      "company_name": "",  // String (e.g., "XYZ Ltd.")
      "contact": "",  // String (e.g., "Jane Smith")
      "customer_number": "",  // String or Number (e.g., "CUST456")
      "address": "",  // String (e.g., "123 Main St")
      "city": "",  // String (e.g., "Los Angeles")
      "postal_code": ""  // String (e.g., "90001")
    }
  }
}

### Text:
    """

    # Create the prompt
    messages = [
      ("system", system_message),
      ("human", template + "\n" + text),
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