import os
import cv2

from src.extract_text import extract_text


def extract_table(image_dir, file, tables_dir, num_pages=2):
    output_dir = os.path.join(tables_dir, file.replace(".jpg", ""))
    os.makedirs(output_dir, exist_ok=True)
    src = cv2.imread(os.path.join(image_dir, file))
    grayed = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    scale = 30
    vertical, horizontal = get_lines(grayed, scale, num_pages * scale)

    mask = vertical + horizontal

    joints = cv2.bitwise_and(horizontal, vertical)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables_found = 0
    for i in range(len(contours)-1, -1, -1):
        area = cv2.contourArea(contours[i])
        if area < 100:
            continue
        contour_poly = cv2.approxPolyDP(contours[i], 3, True)
        bound_rect = cv2.boundingRect(contour_poly)
        x, y, w, h = bound_rect
        roi = joints[y:y+h, x:x+w]
        joint_contours, _ = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(joint_contours) <= 4 or h >= 0.75*(vertical.shape[0]/num_pages):
            continue
        tables_found += 1
        table = extract_text(grayed[y:y + h, x:x + w], joint_contours)
        table.to_csv(os.path.join(output_dir, "%s.csv" % tables_found), index=False, sep="|")


def get_lines(grayed, horizontal_scale, vertical_scale):
    thresholded = cv2.adaptiveThreshold(~grayed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    horizontal = thresholded.copy()
    vertical = thresholded.copy()

    # horizontal lines
    horizontal_size = horizontal.shape[1] // horizontal_scale
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    cv2.erode(horizontal, horizontal_structure, horizontal, iterations=5)
    cv2.dilate(horizontal, horizontal_structure, horizontal, iterations=5)

    # vertical line
    vertical_size = vertical.shape[0] // vertical_scale
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    cv2.erode(vertical, vertical_structure, vertical, iterations=5)
    cv2.dilate(vertical, vertical_structure, vertical, iterations=5)

    return vertical, horizontal

