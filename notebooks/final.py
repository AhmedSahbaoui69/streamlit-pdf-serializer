import marimo

__generated_with = "0.12.8"
app = marimo.App(width="full", app_title="DeBERTa v3 Demo")


@app.cell(hide_code=True)
def _():
    from warnings import filterwarnings
    from os import environ

    filterwarnings("ignore", category=UserWarning)
    filterwarnings("ignore", category=RuntimeWarning)
    filterwarnings("ignore", category=FutureWarning)
    environ["TOKENIZERS_PARALLELISM"] = "false"
    return environ, filterwarnings


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    from pdf2image import convert_from_bytes

    # Placeholder values
    image_bytes, image_np = "x", np.zeros((1, 1, 3), dtype=np.uint8)


    def on_change_file(value=None):
        """Trigger Function to update displayed image"""
        try:
            if file_path.value and file_path.value[0].contents:
                global image_bytes, image_np

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
    return (
        convert_from_bytes,
        file_path,
        image_bytes,
        image_np,
        mo,
        np,
        on_change_file,
        page_number,
    )


@app.cell
def _(file_path, image_np, mo, page_number):
    mo.vstack(
        [
            mo.md("# mDeBERTa v3 Demo"),
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
def _(mo):
    button = mo.ui.button(
        label="Extract Fields",
        kind="success",
        value=False,
        on_click=lambda value: not value,
    )
    button.center()
    return (button,)


@app.cell
def _(__file__):
    import sys
    import os

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    )
    from classes.InvoiceDocument import InvoiceDocument
    from classes.InvoiceDocumentParser import InvoiceDocumentParser
    return InvoiceDocument, InvoiceDocumentParser, os, sys


@app.cell
def _(InvoiceDocument, InvoiceDocumentParser, button, file_path, mo):
    import timeit

    with mo.status.spinner(title="Working on it..."):
        if button.value:
            # Time start
            start = timeit.default_timer()

            test_doc = InvoiceDocument(pdf_bytes=file_path.value[0].contents)
            test_doc_parser = InvoiceDocumentParser(invoice_document=test_doc)
            results = test_doc_parser.parse_header()

            # Time end
            end = timeit.default_timer()

            elapsed_time = end - start
            mo.output.append(
                mo.hstack(
                    [
                        results,
                        mo.stat(
                            value=f"{elapsed_time:.2f} s",
                            label="Total Time",
                            direction="increase",
                            caption=" ",
                        ),
                    ]
                )
            )
    return elapsed_time, end, results, start, test_doc, test_doc_parser, timeit


@app.cell
def _(file_path, mo, np, results):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import fitz  # PyMuPDF
    from io import BytesIO
    from PIL import Image

    # Load PDF from bytes (replace with your actual PDF byte data)
    # For example: with open("file.pdf", "rb") as f: pdf_bytes = f.read()
    doc = fitz.open(stream=file_path.value[0].contents, filetype="pdf")

    # Get the first page and render to an image
    page = doc[0]
    # Scale factor (e.g., 2 for 2x resolution)
    scale = 2.0
    matrix = fitz.Matrix(scale, scale)

    # Render with higher resolution
    pix = page.get_pixmap(matrix=matrix)

    # Convert to PIL Image
    img = Image.open(BytesIO(pix.tobytes("png")))
    width, height = img.size

    # Plot image and draw red rectangles
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(img)

    # Adjust bounding boxes to match the scale
    for field, info in results.items():
        try:
            x0, y0, x1, y1 = info["bounding_box"]
            rect = patches.Rectangle(
                (x0 * scale, y0 * scale),
                (x1 - x0) * scale,
                (y1 - y0) * scale,
                linewidth=1,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
        except:
            pass

    ax.axis("off")
    plt.tight_layout()

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    buf.seek(0)

    # Load image from buffer
    img_with_boxes = Image.open(buf)

    mo.image(np.array(img_with_boxes))
    return (
        BytesIO,
        Image,
        ax,
        buf,
        doc,
        field,
        fig,
        fitz,
        height,
        img,
        img_with_boxes,
        info,
        matrix,
        page,
        patches,
        pix,
        plt,
        rect,
        scale,
        width,
        x0,
        x1,
        y0,
        y1,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
