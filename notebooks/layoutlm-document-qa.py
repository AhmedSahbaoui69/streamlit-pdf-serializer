import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from pdf2image import convert_from_bytes, convert_from_path
    import numpy as np
    from PIL import Image, ImageDraw
    import pytesseract
    from io import BytesIO
    from PyPDF2 import PdfReader
    import random
    import torch
    import pandas as pd
    from langid import classify
    import json
    import os
    import json
    import time
    import csv
    from pathlib import Path
    from transformers import (
        AutoImageProcessor,
        TableTransformerForObjectDetection,
        pipeline,
        AutoProcessor,
        DetrFeatureExtractor,
        AutoModelForObjectDetection,
        Pix2StructForConditionalGeneration,
    )
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    return (
        AutoImageProcessor,
        AutoModelForObjectDetection,
        AutoProcessor,
        BytesIO,
        DetrFeatureExtractor,
        Image,
        ImageDraw,
        Path,
        PdfReader,
        Pix2StructForConditionalGeneration,
        TableTransformerForObjectDetection,
        classify,
        convert_from_bytes,
        convert_from_path,
        csv,
        json,
        mo,
        np,
        os,
        pd,
        pipeline,
        plt,
        pytesseract,
        random,
        time,
        torch,
        warnings,
    )


@app.cell
def _(convert_from_bytes, mo, np):
    image_bytes = "x"
    image_np = np.zeros((1, 1, 3), dtype=np.uint8)


    def on_change_file(value=None):
        try:
            if file_path.value and file_path.value[0].contents:
                global image_bytes
                global image_np

                image_bytes = convert_from_bytes(
                    file_path.value[0].contents,
                    first_page=page_number.value,
                    last_page=page_number.value,
                )[0].convert("RGB")

                image_np = np.array(image_bytes)

            else:
                image_bytes = "x"

        except:
            pass


    file_path = mo.ui.file(
        kind="area", multiple=False, filetypes=[".pdf"], on_change=on_change_file
    )

    page_number = mo.ui.number(
        value=1, start=1, stop=10, step=1, on_change=on_change_file
    )
    return file_path, image_bytes, image_np, on_change_file, page_number


@app.cell
def _(file_path, image_np, mo, page_number):
    mo.vstack(
        [
            mo.md("# LayoutLM Document QA Demo"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            file_path,
                            page_number,
                        ],
                        align="center",
                        justify="space-between",
                    ),
                    mo.image(src=image_np, rounded=True, width=400, height=600),
                ],
                justify="space-around",
                align="center",
            ),
        ],
    )
    return


@app.cell
def get_valid_boxes(mo, np):
    def get_valid_boxes(img, labels, boxes, max_header_height=60):
        labels = labels.tolist()

        # Extract headers, rows, and columns
        headers = [b for idx, b in enumerate(boxes) if labels[idx] == 3]
        rows = sorted(
            [b for idx, b in enumerate(boxes) if labels[idx] == 2],
            key=lambda b: b[1],
        )
        columns = sorted(
            [b for idx, b in enumerate(boxes) if labels[idx] == 1],
            key=lambda b: b[0],
        )

        valid = []
        # Generous margins
        if columns:
            # Expand first column's xmin
            columns[0][0] = max(0, columns[0][0] - 5)

            # Expand last column's xmax
            columns[-1][2] = columns[-1][2] + 5

        # Determine first header
        first_header = min(headers, key=lambda x: x[1]) if headers else None
        header_ymin = None

        if first_header is not None:
            (xmin, ymin, xmax, ymax) = first_header
            if ymax - ymin > max_header_height:
                ymin = ymax - max_header_height
                header_ymin = ymin

                valid.append((3, (xmin, ymin, xmax, ymax + 2)))

        # Process rows in reverse order
        empty_row_ymin = None
        for xmin, ymin, xmax, ymax in rows:
            row_image = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))

            row_pixels = np.array(row_image)

            # Check if more than 98% of the pixels are white (assuming white is (255, 255, 255))
            white_pixel_percentage = (
                np.sum(np.all(row_pixels == [255, 255, 255], axis=-1))
                / row_pixels.size
            )
            print(white_pixel_percentage)
            if white_pixel_percentage > 0.98:
                mo.output.append(mo.md("# YAY"))
                empty_row_ymin = ymin
                break

            valid.append((2, (xmin, ymin, xmax, ymax)))

        # Process columns
        prev_xmax = None
        for i, (xmin, ymin, xmax, ymax) in enumerate(columns):
            # Fix overlapping Y ranges if needed
            if empty_row_ymin is not None:
                ymax = empty_row_ymin
            if header_ymin is not None:
                ymin = header_ymin
            ymax = max(ymin, ymax)

            # Fix X overlap or gaps between columns
            if prev_xmax is not None:
                # If there's vertical clipping or overlap
                prev_ymin, prev_ymax = valid[-1][1][1], valid[-1][1][3]
                if prev_ymax > ymin or ymax > prev_ymin:
                    # Columns overlap in Y, force alignment in X
                    xmin = prev_xmax

            valid.append((1, (xmin, ymin, xmax, ymax)))
            prev_xmax = xmax

        return valid
    return (get_valid_boxes,)


@app.cell
def _(ImageDraw, image_processor_TTD, model_TTD, np, torch):
    def get_image_part(image, half="full"):
        """Extract the specified part of the image."""
        width, height = image.size

        if half == "top":
            return image.crop((0, 0, width, height // 2))
        elif half == "bottom":
            return image.crop((0, height // 2.4, width, height))
        else:  # "full"
            return image


    def plot_results(pil_img, valid_boxes):
        # Convert the image to RGB (in case it's not in RGB mode)
        pil_img = pil_img.convert("RGB")

        # Create a drawing context
        draw = ImageDraw.Draw(pil_img)

        # Loop to plot bounding boxes
        for label, (xmin, ymin, xmax, ymax) in valid_boxes:
            if label == 1:
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=1)
            elif label == 3:
                draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=1)
            elif label == 2:
                pass
                draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=1)

        return pil_img


    def process_image_ttsr(image, half="bottom"):
        """Process image with Table Transformer Structure Recognition model."""
        # Get the specified part of the image
        cropped_part = get_image_part(image, half)

        image_np = np.array(cropped_part)
        inputs = image_processor_TTD(images=image_np, return_tensors="pt")

        outputs = model_TTD(**inputs)

        target_sizes = torch.tensor([cropped_part.size[::-1]])
        results_TTD = image_processor_TTD.post_process_object_detection(
            outputs, threshold=0.6, target_sizes=target_sizes
        )[0]

        boxes = results_TTD["boxes"].tolist()

        if not boxes:
            return None

        largest_box = max(boxes, key=lambda box: (box[2] - box[0]), default=None)

        # Crop the image to the largest box
        x0, y0, x1, y1 = map(int, largest_box)
        cropped_part = cropped_part.crop((x0 - 10, y0 - 20, x1 + 10, y1 + 20))

        return cropped_part


    def inference_ttsr(image, image_processor_TTSR, model_TTSR):
        """Run inference on image with TTSR model."""
        # Prepare image for the model
        encoding = image_processor_TTSR(image, return_tensors="pt")

        # Run model inference
        with torch.no_grad():
            outputs = model_TTSR(**encoding)

        # Post-process results
        target_sizes = [image.size[::-1]]
        results = image_processor_TTSR.post_process_object_detection(
            outputs, threshold=0.4, target_sizes=target_sizes
        )[0]

        return results
    return get_image_part, inference_ttsr, plot_results, process_image_ttsr


@app.cell
def _(
    AutoImageProcessor,
    DetrFeatureExtractor,
    TableTransformerForObjectDetection,
):
    # Initialize TTD Model
    image_processor_TTD = AutoImageProcessor.from_pretrained(
        "microsoft/table-transformer-detection", use_fast=True
    )
    model_TTD = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection"
    )

    # Initialize TTSR Model
    image_processor_TTSR = DetrFeatureExtractor()
    model_TTSR = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition-v1.1-all"
    )
    return image_processor_TTD, image_processor_TTSR, model_TTD, model_TTSR


@app.cell
def _(
    get_valid_boxes,
    image_bytes,
    image_processor_TTSR,
    inference_ttsr,
    mo,
    model_TTSR,
    np,
    plot_results,
    process_image_ttsr,
):
    # Process bottom half of image with TTSR
    cropped_image = process_image_ttsr(image_bytes, half="bottom")

    cropped_image = cropped_image.convert("L").convert("RGB")

    # Run inference
    structure = inference_ttsr(cropped_image, image_processor_TTSR, model_TTSR)
    columns = get_valid_boxes(
        cropped_image,
        structure["labels"],
        structure["boxes"],
    )
    plotted_image = plot_results(cropped_image, columns)

    mo.output.append(mo.image(np.array(plotted_image)))
    return columns, cropped_image, plotted_image, structure


@app.cell
def _(columns, cropped_image, pd, pytesseract):
    # Filter boxes with label == 1
    text_columns = [box for label, box in columns if label == 1]

    # OCR each column and split by line
    column_texts = []
    for box in text_columns:
        xmin, ymin, xmax, ymax = [int(v) for v in box]
        cropped_column = cropped_image.crop((xmin, ymin, xmax, ymax))
        text = pytesseract.image_to_string(cropped_column, config="--psm 6")
        lines = text.strip().split('\n')  # Split text into rows
        column_texts.append(lines)

    # Normalize column lengths by padding with empty strings
    max_rows = max(len(col) for col in column_texts)
    for i in range(len(column_texts)):
        while len(column_texts[i]) < max_rows:
            column_texts[i].append("")

    # Create a DataFrame
    df = pd.DataFrame(
        {f"col_{i + 1}": column_texts[i] for i in range(len(column_texts))}
    )
    df
    return (
        box,
        column_texts,
        cropped_column,
        df,
        i,
        lines,
        max_rows,
        text,
        text_columns,
        xmax,
        xmin,
        ymax,
        ymin,
    )


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
