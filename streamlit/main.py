import streamlit as st
from models.tesseract import TesseractOCR
from models.azure import AzureAI
from st_diff_viewer import diff_viewer
import os
import time
import json
from pdf2image import convert_from_path

# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("Comparison of OCR Models")

# Upload sections
col1, col2 = st.columns(2)
with col1:
  uploaded_file = st.file_uploader("Upload Invoice File *", type=["pdf", "png", "jpg", "jpeg"], key="file_uploader")
with col2:
  validation_file = st.file_uploader("Upload Validation JSON", type=["json"], key="json_uploader")

# Model selection
models = {
  "Tesseract + Nuextract": TesseractOCR(),
  "Azure AI Document Intelligence": AzureAI()
}
selected_model_names = st.multiselect("Choose OCR models to use *", list(models.keys()), key="model_selector")

# Extraction trigger
if st.button("Extract Data"):
  error_messages = []
  if not uploaded_file:
    error_messages.append("Please upload a PDF file.")
  if not selected_model_names:
    error_messages.append("Please select at least one OCR model.")

  if error_messages:
    for error in error_messages:
      st.error(error)
  else:
    # Save uploaded PDF
    os.makedirs("./temp", exist_ok=True)
    pdf_path = os.path.abspath("./temp/uploaded.pdf")
    with open(pdf_path, "wb") as f:
      f.write(uploaded_file.getbuffer())

    # Load validation JSON if provided
    validation_text = None
    if validation_file is not None:
      validation_data = json.load(validation_file)
      validation_text = json.dumps(validation_data, indent=4)

    extracted_texts = {}
    timing_info = {}
    progress_info = st.info("Starting extraction...")

    # Run OCR for each selected model
    for model_name in selected_model_names:
      progress_info.info(f"Processing with {model_name}...")
      start_time = time.time()

      for result in models[model_name].run(pdf_path):
        output = result.get("output", "")
        progress = result.get("progress", "")
        if progress:
          progress_info.info(progress)
        if output:
          extracted_texts[model_name] = output

      elapsed_time = time.time() - start_time
      minutes, seconds = divmod(elapsed_time, 60)
      timing_info[model_name] = (int(minutes), int(seconds))

    # Display results
    if extracted_texts:
      tabs = st.tabs(list(extracted_texts.keys()))
      for i, (model_name, extracted_text) in enumerate(extracted_texts.items()):
        with tabs[i]:
          col1, col2 = st.columns(2)

          # Always show PDF preview and extracted text
          with col1:
            st.subheader("PDF Preview")
            try:
              images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=1)
              if images:
                img = images[0]
                img_width, img_height = img.size
                display_height = int(500 * (img_height / img_width))
                st.image(img, width=500)
            except Exception as e:
              st.warning(f"Failed to render PDF preview: {e}")

          with col2:
            st.subheader(f"Extracted Text")
            text_height = display_height if 'display_height' in locals() else 700
            st.code(extracted_text, language="text", height=text_height)

          if validation_text:
            with col1:
              st.subheader("Extracted Data")
            with col2:
              st.subheader("Target Data")

            diff_viewer(
              extracted_text, validation_text,
              split_view=True,
              key=f"diff_{model_name}",
              line_height=0.8,
              hide_line_numbers=True
            )

          minutes, seconds = timing_info[model_name]
          st.success(f"Duration: **{minutes} min {seconds} s**")