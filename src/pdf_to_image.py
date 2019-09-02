import os
from pdf2image import convert_from_path
import numpy as np
from PIL import Image


def convert_pdf_to_image(file_dir, filename, output_dir=None):
    try:
        images_from_path = convert_from_path(os.path.join(file_dir, filename))
        combined = np.vstack(list(map(np.asarray, images_from_path)))
        if output_dir is not None:
            image_filename = filename.replace(".pdf", ".jpg")
            Image.fromarray(combined).save(os.path.join(output_dir, image_filename), 'JPEG')
        return combined, len(images_from_path)
    except ValueError:
        print("Error converting pdf to image for %s!!!" % filename)
        return [], 0

