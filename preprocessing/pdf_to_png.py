import fitz
from pathlib import Path

def convert_pdf_to_png(pdf_path, output_dir, dpi=300):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    with fitz.open(pdf_path) as doc:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_index in range(len(doc)):
            page = doc[page_index]
            pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            page_num = page_index + 1
            out_filename = f"{pdf_path.stem}_page_{page_num:03d}.png"
            out_path = output_dir / out_filename

            pixmap.save(str(out_path))
            saved_files.append(out_path)

    return saved_files

