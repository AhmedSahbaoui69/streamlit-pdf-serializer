import json
from urllib import response
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dateutil.parser import parse
from langid import classify
from pprint import pprint
from transformers import pipeline
import os
from difflib import SequenceMatcher

# Initialize qa model
# qa_model = pipeline("question-answering", model="timpal0l/mdeberta-v3-base-squad2")
qa_model = pipeline("text2text-generation", model="google/flan-t5-large")

class InvoiceDocumentParser:
  def __init__(self, invoice_document):
    self.invoice_document = invoice_document
    self.lang = "en"

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(project_dir, "src", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
      config = json.load(f)

    self.prompts = config["prompts"]
    self.fields = config["fields"]
    self.business_synonyms = config["business_synonyms"]
    self.result_keys = config["fields"]['en']

  def parse_header(self):
    # Step 1: Combine the text from the first and last pages
    combined_text = (
      self.invoice_document.pages[0].get_text() + "\n" +
      self.invoice_document.pages[-2].get_text() + "\n" +
      self.invoice_document.pages[-1].get_text()
    )

    # Step 2: Split the combined text into lines and clean them
    lines = [
      line.strip()
      for line in combined_text.splitlines()
      if len(line.strip()) > 1
    ]

    # Step 3: Detect the text language 
    self._detect_language(combined_text)

    # Step 4: Build context for each field
    field_contexts = self._build_field_contexts(lines)

    # Step 5: Extract field values from the field contexts
    field_values = self._extract_field_values(field_contexts)

    # Step 6: Extract all word-level bounding boxes from all pages
    all_words_with_bbox = self._extract_all_words_with_bbox()

    # Match results with bounding boxes using a more accurate approach
    final_results = {}
    for key, value in field_values.items():
      if not value:
        final_results[key] = {
          "value": "",
          "bounding_box": None,
          "page": None
        }
        continue
        
      # Find the best matching words/phrases in the document
      best_match = self._find_best_match_for_value(value, all_words_with_bbox)
      
      if best_match:
        final_results[key] = {
          "value": value,
          "bounding_box": best_match["bbox"],
          "page": best_match["page"]
        }
      else:
        final_results[key] = {
          "value": value,
          "bounding_box": None,
          "page": None
        }
    
    return final_results

  def _extract_all_words_with_bbox(self):
    """Extract all words with their bounding boxes from all pages"""
    all_words = []
    
    for page_idx, page in enumerate(self.invoice_document.pages):
      # Get all words with their bounding boxes
      # Format: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
      words = page.get_text("words")
      
      for word_info in words:
        all_words.append({
          "text": word_info[4],
          "bbox": (word_info[0], word_info[1], word_info[2], word_info[3]),
          "page": page_idx,
          "block": word_info[5],
          "line": word_info[6],
          "word_idx": word_info[7]
        })
    
    return all_words

  def _find_best_match_for_value(self, value, all_words_with_bbox):
    """Find the best matching word or sequence of words for a given value"""
    value = value.lower().strip(".,:;!?\"'()[]{}")  # Strip extra characters from the sides
    
    # Try exact match first
    for word in all_words_with_bbox:
      word_text = word["text"].lower().strip(".,:;!?\"'()[]{}")  # Strip extra characters
      if word_text == value:
        return word
    
    best_match, best_similarity = None, 0
    words = value.split()
    
    # For single words, use direct similarity comparison
    if len(words) == 1:
      for word in all_words_with_bbox:
        word_text = word["text"].lower().strip(".,:;!?\"'()[]{}")  # Strip extra characters
        sim = SequenceMatcher(None, value, word_text).ratio()
        if sim > best_similarity and sim > 0.8:
          best_similarity, best_match = sim, word
      return best_match
    
    # For phrases, group words by location and try sequential matching
    location_dict = {}
    for word in all_words_with_bbox:
      key = (word["page"], word["block"], word["line"])
      location_dict.setdefault(key, []).append(word)
    
    # Process each location group
    for word_list in location_dict.values():
      word_list.sort(key=lambda w: w["word_idx"])
      
      if len(word_list) < len(words):
        continue
      
      # Check each possible sequence of words
      for i in range(len(word_list) - len(words) + 1):
        phrase = " ".join(w["text"].strip(".,:;!?\"'()[]{}") for w in word_list[i:i+len(words)])  # Strip extra characters
        sim = SequenceMatcher(None, value, phrase.lower()).ratio()
        
        if sim > best_similarity and sim > 0.8:
          best_similarity = sim
          
          # Create combined bounding box
          slice_words = word_list[i:i+len(words)]
          x0 = min(w["bbox"][0] for w in slice_words)
          y0 = min(w["bbox"][1] for w in slice_words)
          x1 = max(w["bbox"][2] for w in slice_words)
          y1 = max(w["bbox"][3] for w in slice_words)
          
          best_match = {
            "text": phrase,
            "bbox": (x0, y0, x1, y1),
            "page": slice_words[0]["page"],
            "block": slice_words[0]["block"],
            "line": slice_words[0]["line"]
          }
    
    return best_match
    
  def _detect_language(self, text):
    self.lang = classify(text)[0]
    if self.lang not in ["en", "fr"]:
      self.lang = "en"

  def _build_field_contexts(self, lines):
    field_contexts = {}
    current_fields = self.fields[self.lang]

    for idx, field in enumerate(current_fields):
      enhanced_field = self._enhance_text(field)
      context_lines = self._create_context_slices(
        lines, self._find_similar_lines(enhanced_field, lines)
      )
      result_key = self.result_keys[idx]
      context = "\n".join(context_lines)

      field_contexts[result_key] = context

    return field_contexts

  def _enhance_text(self, text):
    synonyms = self.business_synonyms[self.lang]
    for term, syns in synonyms.items():
      if term in text.lower():
        text += " " + " ".join(syns)
    return text

  def _find_similar_lines(self, query, lines):
    if not lines:
      return {"indices": [], "scores": []}

    enhanced_lines = [line for line in lines]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    try:
      vectors = vectorizer.fit_transform(enhanced_lines + [query])
    except ValueError:
      return {"indices": [], "scores": []}

    query_vec = vectors[-1]
    line_vecs = vectors[:-1]
    scores = cosine_similarity(query_vec, line_vecs).flatten()

    top_indices = np.argsort(scores)[::-1][:5].tolist()
    top_scores = scores[top_indices].tolist()

    return {
      "indices": top_indices,
      "scores": top_scores,
    }

  def _create_context_slices(self, lines, matches, context_size=3):
    # Generate slices around matched indices
    slices = []
    for idx in matches["indices"][:context_size]:
      start = max(0, idx)
      end = min(idx + context_size, len(lines))
      slices.append((start, end))

    # Sort slices by start index
    slices.sort(key=lambda x: x[0])

    # Extract lines from merged slices
    result_lines = []
    for s, e in slices:
      for idx, line in enumerate(lines[s:e]):
        # Always include the first line
        if "please" in line.lower():
          continue
        if idx == 0:
          if len(line) > 50:
            truncated = line[:40].rsplit(" ", 1)[0]
            result_lines.append(truncated)
          else:
            result_lines.append(line)
          continue

        # Skip lines that are too short
        elif len(line) < 5:
          continue

        # Truncate line if too long
        if len(line) > 40:
          truncated = line[:40].rsplit(" ", 1)[0]
      
          # Skip lines with too many words
          if len(truncated.split()) > 3:
            continue
          else:
            result_lines.append(truncated)

        # Skip lines with too many words
        elif len(line.split()) >= 3:
          continue
        else:
          result_lines.append(line)

    print(result_lines)

    return result_lines

  def _extract_field_values(self, field_contexts):
    results = {key: "" for key in self.result_keys}
    prompts = self.prompts[self.lang]

    for idx, key in enumerate(self.result_keys):
      context = field_contexts.get(key, "")
      if not context:
        continue

      pprint(context)
      # response = qa_model(question=prompts[idx], context=context)
      response = qa_model(f"{context}\n {prompts[idx]}")
      # results[key] = self._clean_answer(response["answer"], key)
      results[key] = self._clean_answer(response[0]["generated_text"], key)

    return results

  def _clean_answer(self, text, key=""):
    clean_text = text.strip().replace(":", "").replace(";", "").replace('"', "")

    if key in ["due_date", "invoice_date"]:
      try:
        dt = parse(clean_text, dayfirst=True, fuzzy=True)
        return dt.date().isoformat()
      except (ValueError, TypeError):
        return clean_text  # fallback to raw cleaned string if parsing fails

    return clean_text
