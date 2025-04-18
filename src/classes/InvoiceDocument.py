import fitz

class InvoiceDocument:
  def __init__(self, pdf_bytes: bytes):
    self.pdf_bytes = pdf_bytes
    self.pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    self.pages = [self.pdf_doc[i] for i in range(len(self.pdf_doc))]
