import os
import json
from src.pdf_to_image import convert_pdf_to_image
from src.find_table import extract_tables

if __name__ == '__main__':
    input_dir = '../resources/pdf'
    # output_dir = '../resources/pdf-image'
    tables_dir = '../resources/tables'
    files = os.listdir(input_dir)
    processed = []
    no_tables = []
    for file in files:
        image, num_pages = convert_pdf_to_image(input_dir, file)
        if num_pages != 0:
            tables = extract_tables(num_pages, image)
            if len(tables) > 0:
                output_folder = os.path.join(tables_dir, file.replace(".pdf", ""))
                os.makedirs(output_folder, exist_ok=True)
                for i, table in enumerate(tables):
                    with open(os.path.join(output_folder, "%s.json" % i), "w") as f:
                        json.dump(table, f)
                print("processed %s" % file)
                processed.append(file)
            else:
                no_tables.append(file)
