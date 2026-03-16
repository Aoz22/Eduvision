import fitz  
from pathlib import Path


def get_pages_per_student(model_answer_path):
    model_answer_path = Path(model_answer_path)

    with fitz.open(model_answer_path) as doc:
        return len(doc)


def split_pdf(big_pdf_path, pages_per_student, output_dir):
    big_pdf_path = Path(big_pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    flagged = []

    with fitz.open(big_pdf_path) as doc:
        total_pages = len(doc)
        num_students = total_pages // pages_per_student

        for i in range(num_students):
            start_page = i * pages_per_student
            end_page = start_page + pages_per_student

            student_name = f"student_{i + 1:03d}"
            student_pdf = fitz.open()

            for page_index in range(start_page, end_page):
                student_pdf.insert_pdf(doc, from_page=page_index, to_page=page_index)

            out_path = output_dir / f"{student_name}.pdf"
            student_pdf.save(str(out_path))
            student_pdf.close()
            saved_files.append(out_path)

        # Handle leftover pages
        # TODO: replace with YOLO-based boundary detection after YOLO training is complete because this code cannot handle the problem of missing pages in the middle
        remaining_pages = total_pages % pages_per_student

        if remaining_pages != 0:
            student_name = f"student_{num_students + 1:03d}"
            student_pdf = fitz.open()

            for page_index in range(num_students * pages_per_student, total_pages):
                student_pdf.insert_pdf(doc, from_page=page_index, to_page=page_index)

            flagged.append({
                "student": student_name,
                "expected": pages_per_student,
                "got": remaining_pages
            })

            out_path = output_dir / f"{student_name}.pdf"
            student_pdf.save(str(out_path))
            student_pdf.close()
            saved_files.append(out_path)

    return saved_files, flagged