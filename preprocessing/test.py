# from pdf_splitter import get_pages_per_student, split_pdf
# from pdf_to_png import convert_pdf_to_png
# from pathlib import Path

# model_answer_path = "C:/Projects/python/Eduvision/preprocessing/tamp1.pdf"
# big_pdf_path = "C:\Projects\python\Eduvision\preprocessing\ilovepdf_merged (1).pdf"
# splits_dir = "data/splits"
# images_dir = "data/images"

# # Step 1
# pages_per_student = get_pages_per_student(model_answer_path)
# print(f"Pages per student: {pages_per_student}")

# # Step 2
# saved_files, flagged = split_pdf(big_pdf_path, pages_per_student, splits_dir)
# print(f"Split complete: {len(saved_files)} students, {len(flagged)} flagged")
# for f in flagged:
#     print(f"⚠ {f['student']} — expected {f['expected']} pages, got {f['got']}")
# # Step 3
# for student_pdf in saved_files:
#     pngs = convert_pdf_to_png(student_pdf, Path(images_dir) / student_pdf.stem)
#     print(f"{student_pdf.name} → {len(pngs)} PNG(s)")


import cv2
from enhance import deskew, apply_clahe, denoise, remove_background

# load image
image = cv2.imread(r"C:\Projects\python\Eduvision\data\images\student_005\student_005_page_002.png")

# apply enhance functions in order
deskewed = deskew(image)
clahe_applied = apply_clahe(deskewed)
denoised = denoise(clahe_applied)
cleaned = remove_background(denoised)

# save result
cv2.imwrite("output.png", cleaned)
print("Done — check output.png")