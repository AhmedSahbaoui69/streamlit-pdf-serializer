from turtle import width
import streamlit as st
from classes.InvoiceDocument import InvoiceDocument
from classes.InvoiceDocumentParser import InvoiceDocumentParser
from streamlit_pdf_viewer import pdf_viewer
from dateutil.parser import parse
from io import BytesIO

st.set_page_config(page_title="Invoice Parser", layout="wide")

st.title("📄 Invoice Parser")

uploaded_file = st.file_uploader("Upload a PDF invoice", type=["pdf"])

def create_pdf_with_annotations(pdf_bytes, annotations_by_page):
  """Create a PDF with bounding box annotations"""
  import fitz

  # Open the PDF from bytes
  doc = fitz.open(stream=pdf_bytes, filetype="pdf")

  # Add annotations to each page
  for page_num, annotations in annotations_by_page.items():
    page = doc[int(page_num)]
    for annot in annotations:
      bbox = annot["bbox"]
      # Add a rectangular annotation for the bounding box
      page.add_rect_annot(bbox)  # Red rectangle

  # Save the annotated PDF to bytes
  output_bytes = BytesIO()
  doc.save(output_bytes)
  doc.close()

  return output_bytes.getvalue()

if uploaded_file:
  # Check if the uploaded file has changed
  if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file:
    # Read the file as bytes
    pdf_bytes = uploaded_file.read()

    # Build the InvoiceDocument object
    invoice_doc = InvoiceDocument(pdf_bytes)

    # Parse the document
    parser = InvoiceDocumentParser(invoice_doc)
    results = parser.parse_header()

    # Store results and file in session state
    st.session_state.last_uploaded_file = uploaded_file
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.invoice_doc = invoice_doc
    st.session_state.results = results

  # Retrieve data from session state
  pdf_bytes = st.session_state.pdf_bytes
  invoice_doc = st.session_state.invoice_doc
  results = st.session_state.results

  # Get total pages
  total_pages = len(invoice_doc.pdf_doc)

  # Create two columns
  cont1, cont2 = st.columns([2, 1])

  with cont1:
    # Page selector
    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    # Create annotations dictionary by page
    annotations_by_page = {}
    if isinstance(results, dict):
      for key, data in results.items():
        if data.get("bounding_box") and data.get("page") is not None:
          page = data["page"]
          if str(page) not in annotations_by_page:
            annotations_by_page[str(page)] = []

          annotations_by_page[str(page)].append({
            "field": key,
            "bbox": data["bounding_box"],
            "value": data["value"]
          })

    # Toggle to show/hide bounding boxes
    show_boxes = st.checkbox("Show bounding boxes", value=True)

    if show_boxes and annotations_by_page:
      # Create annotated PDF with bounding boxes
      annotated_pdf = create_pdf_with_annotations(pdf_bytes, annotations_by_page)

      # Display PDF preview of selected page with annotations
      pdf_viewer(
        annotated_pdf,  # Annotated PDF bytes
        pages_to_render=[page_num],
        width="100%"
      )
    else:
      # Display original PDF preview of selected page
      pdf_viewer(
        pdf_bytes,  # Input PDF bytes
        pages_to_render=[page_num],
        width="100%"
      )

  with cont2:
    # Display results as form fields, 2 per row
    if isinstance(results, dict):
      cols = st.columns(2)  # Create two columns for input fields
      col_index = 0  # Track which column to use

      for key, data in results.items():
        value = data["value"] if isinstance(data, dict) else data
        bbox = None if not isinstance(data, dict) else data.get("bounding_box")
        page = None if not isinstance(data, dict) else data.get("page")

        # Try to parse value as a date
        is_date = False
        try:
          # Add validation to prevent OverflowError
          # Check if value is a reasonable string length for dates
          if isinstance(value, str) and len(value) <= 30:
            parsed_date = parse(value, fuzzy=False)
            # Display as a date input field if parsing succeeds
            with cols[col_index]:
              st.date_input(label=key, value=parsed_date.date(), key=f"date_input_{key}")
            is_date = True
        except (ValueError, TypeError, OverflowError):
          # Skip and continue to text input
          pass

        if not is_date:
          with cols[col_index]:
            st.text_input(label=key, value=value, key=f"text_input_{key}")

        # Switch to the next column
        col_index = (col_index + 1) % 2
