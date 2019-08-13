import os
from src.pdf_to_image import convert_pdf_to_image
from src.find_table import extract_table

if __name__ == '__main__':
    input_dir = '../resources/pdf'
    output_dir = '../resources/pdf-image'
    tables_dir = '../resources/tables'
    files = os.listdir(input_dir)
    for file in files:
        image, num_pages = convert_pdf_to_image(input_dir, file, output_dir)
        extract_table(file.replace(".pdf", ""), tables_dir, num_pages, image)
