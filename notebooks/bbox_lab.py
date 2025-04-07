import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pytesseract
    from PIL import Image, ImageDraw, ImageFont
    import pytesseract
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from ultralytics import YOLO
    from pdf2image import convert_from_bytes
    import base64
    from io import BytesIO
    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    import torch
    import re
    import warnings
    return (
        AutoImageProcessor,
        BytesIO,
        Image,
        ImageDraw,
        ImageFont,
        TableTransformerForObjectDetection,
        YOLO,
        base64,
        convert_from_bytes,
        hf_hub_download,
        mo,
        np,
        patches,
        pd,
        plt,
        pytesseract,
        re,
        time,
        torch,
        warnings,
    )


@app.cell
def _(mo):
    # Marimo File Picker
    file_path = mo.ui.file(kind="area", filetypes=[".pdf"])
    page_number = mo.ui.number(start=1, stop=10, step=1)
    mo.vstack([file_path, page_number], align="center")
    return file_path, page_number


@app.cell
def _(convert_from_bytes, file_path, np, page_number):
    # Convert Selected Page To Bytes
    page = convert_from_bytes(
        file_path.value[0].contents,
        first_page=page_number.value,
        last_page=page_number.value,
    )[0]

    # Transform into np.array
    image_np = np.array(page)
    return image_np, page


@app.cell
def _(AutoImageProcessor, TableTransformerForObjectDetection, YOLO, warnings):
    # Load Yolo Model
    model_YOLO = YOLO("notebooks/models/best.pt")

    # Load Microsoft TTD Model
    warnings.filterwarnings("ignore", category=UserWarning)
    image_processor_TTD = AutoImageProcessor.from_pretrained(
        "microsoft/table-transformer-detection", use_fast=True
    )
    model_TTD = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection"
    )
    return image_processor_TTD, model_TTD, model_YOLO


@app.cell
def _(image_np, image_processor_TTD, model_TTD, model_YOLO, page, time, torch):
    # Run YOLO Detection
    start_YOLO = time.time()  # Start timing YOLO
    results_YOLO = model_YOLO(image_np, conf=0.35, max_det=500, agnostic_nms=True)
    end_YOLO = time.time()  # End timing YOLO
    time_YOLO = end_YOLO - start_YOLO

    # Run TTD Detection
    start_TTD = time.time()  # Start timing TTD
    inputs = image_processor_TTD(images=page, return_tensors="pt")
    outputs = model_TTD(**inputs)
    target_sizes = torch.tensor([page.size[::-1]])
    results_TTD = image_processor_TTD.post_process_object_detection(
        outputs, threshold=0.7, target_sizes=target_sizes
    )[0]
    end_TTD = time.time()  # End timing TTD
    time_TTD = end_TTD - start_TTD
    return (
        end_TTD,
        end_YOLO,
        inputs,
        outputs,
        results_TTD,
        results_YOLO,
        start_TTD,
        start_YOLO,
        target_sizes,
        time_TTD,
        time_YOLO,
    )


@app.cell(hide_code=True)
def _(results_TTD, results_YOLO):
    # Extract Detected Bounding Boxes
    boxes_YOLO = [
        {"box": box, "score": score}
        for result in results_YOLO
        for box, score in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist()
        )
    ]
    boxes_TTD = [
        {"box": box, "score": score}
        for box, score in zip(
            results_TTD["boxes"].tolist(), results_TTD["scores"].tolist()
        )
    ]
    return boxes_TTD, boxes_YOLO


@app.cell
def _(BytesIO, Image, ImageDraw, ImageFont, base64, np):
    def plot_boxes(image, detections, color="red", padding=0):
        img = image.copy()
        censored_img = image.copy()
        draw = ImageDraw.Draw(img)
        cropped_images = []

        for det in detections:
            coords = [
                max(0, int(round(c))) + (-padding if i < 2 else padding)
                for i, c in enumerate(det["box"])
            ]
            x_min, y_min, x_max, y_max = coords
            x_max = min(x_max, img.width)
            y_max = min(y_max, img.height)

            # Draw rectangle and label
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=5)
            label = f"{det['score']:.2f}"
            text_bbox = draw.textbbox((0, 0), label, font=ImageFont.load_default())
            text_w, text_h = (
                text_bbox[2] - text_bbox[0],
                text_bbox[3] - text_bbox[1],
            )
            draw.rectangle(
                [x_min, y_min - text_h - 6, x_min + text_w + 6, y_min], fill=color
            )
            draw.text((x_min + 3, y_min - text_h - 3), label, fill="white")

            # Crop and store cropped images
            cropped_images.append(np.array(img.crop((x_min, y_min, x_max, y_max))))

            # Draw on censored image
            ImageDraw.Draw(censored_img).rectangle(
                [x_min, y_min, x_max, y_max], fill="white"
            )

        return np.array(img), np.array(censored_img), cropped_images


    def img_to_base64(img_array):
        img = Image.fromarray(img_array.astype("uint8"))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_to_base64, plot_boxes


@app.cell
def _(boxes_TTD, boxes_YOLO, img_to_base64, page, plot_boxes):
    im_array_YOLO, censored_im_array_YOLO, cropped_im_arrays_YOLO = plot_boxes(
        page, boxes_YOLO, color="blue", padding=10
    )
    im_array_TTD, censored_im_array_TTD, cropped_im_arrays_TTD = plot_boxes(
        page, boxes_TTD, padding=20
    )

    # Process models
    models = {
        "YOLO": (im_array_YOLO, censored_im_array_YOLO, cropped_im_arrays_YOLO),
        "TTD": (im_array_TTD, censored_im_array_TTD, cropped_im_arrays_TTD),
    }

    base64_images = {
        model: {
            "full_image": [img_to_base64(im)],
            "censored_image": [img_to_base64(censored)],
            "cropped_images": [img_to_base64(crop) for crop in cropped],
        }
        for model, (im, censored, cropped) in models.items()
    }
    return (
        base64_images,
        censored_im_array_TTD,
        censored_im_array_YOLO,
        cropped_im_arrays_TTD,
        cropped_im_arrays_YOLO,
        im_array_TTD,
        im_array_YOLO,
        models,
    )


@app.cell
def _(base64_images, mo, time_TTD, time_YOLO):
    mo.hstack(
        [
            mo.vstack(
                [
                    # YOLO full image
                    mo.md(
                        f'# YOLO\n{time_YOLO}\n<img src="data:image/png;base64,{base64_images["YOLO"]["full_image"][0]}", width="500"/>'
                    ).center(),
                    # YOLO cropped images
                    *[
                        mo.md(
                            f'<img src="data:image/png;base64,{cropped_base64}", width="500"/>'
                        ).center()
                        for cropped_base64 in base64_images["YOLO"][
                            "cropped_images"
                        ]
                    ],
                    # YOLO censored image
                    mo.md(
                        f'<img src="data:image/png;base64,{base64_images["YOLO"]["censored_image"][0]}", width="500"/>'
                    ).center(),
                ]
            ),
            mo.vstack(
                [
                    # TTD full image
                    mo.md(
                        f'# TTD\n{time_TTD}\n<img src="data:image/png;base64,{base64_images["TTD"]["full_image"][0]}", width="500"/>'
                    ).center(),
                    # TTD cropped images
                    *[
                        mo.md(
                            f'<img src="data:image/png;base64,{cropped_base64}", width="500"/>'
                        ).center()
                        for cropped_base64 in base64_images["TTD"][
                            "cropped_images"
                        ]
                    ],
                    # TTD censored image
                    mo.md(
                        f'<img src="data:image/png;base64,{base64_images["TTD"]["censored_image"][0]}", width="500"/>'
                    ).center(),
                ]
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
