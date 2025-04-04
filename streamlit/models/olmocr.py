# import torch
# import base64

# from io import BytesIO
# from PIL import Image
# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# from olmocr.data.renderpdf import render_pdf_to_base64png
# from olmocr.prompts import build_finetuning_prompt
# from olmocr.prompts.anchor import get_anchor_text

# class OlmOCR():
#     def __init__(self):
#         self.model_id = "allenai/olmOCR-7B-0225-preview"
#     def run(self, pdf_path):
#         model = Qwen2VLForConditionalGeneration.from_pretrained(
#             self.model_id,
#             torch_dtype=torch.bfloat16
#         ).eval()
        
#         processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
        
#         # Encode page 1 into base64 PNG
#         image_base64 = render_pdf_to_base64png(pdf_path, 1, target_longest_image_dim=1024)
        
#         # Yield initial progress message
#         yield {"progress": "Initializing OCR extraction...", "output": ""}
        
#         # Build the prompt, using document metadata
#         anchor_text = get_anchor_text(pdf_path, 1, pdf_engine="pdfreport", target_length=4000)
#         prompt = build_finetuning_prompt(anchor_text)
        
#         # Yield progress message before building the full prompt
#         yield {"progress": "Building OCR prompt...", "output": ""}
        
#         # Build the full prompt
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
#                 ],
#             }
#         ]
        
#         # Apply the chat template and processor
#         text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         main_image = Image.open(BytesIO(base64.b64decode(image_base64)))
        
#         inputs = processor(
#             text=[text],
#             images=[main_image],
#             padding=True,
#             return_tensors="pt",
#         )
#         inputs = {key: value.to(device) for (key, value) in inputs.items()}
        
#         # Yield progress message before starting the model generation
#         yield {"progress": "Generating OCR output...", "output": ""}
        
#         # Generate the output
#         output = model.generate(
#             **inputs,
#             temperature=0.8,
#             max_new_tokens=50,
#             num_return_sequences=1,
#             do_sample=True,
#         )
        
#         # Decode the output
#         prompt_length = inputs["input_ids"].shape[1]
#         new_tokens = output[:, prompt_length:]
#         text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
#         # Yield final progress message with the OCR output
#         yield {"progress": "OCR extraction complete", "output": text_output[0] if text_output else "Error"}
