from statistics import mode

import cv2
import numpy as np
import pytesseract


def extract_tables(num_pages, src):
    grayed = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    scale = 60
    vertical, horizontal = get_lines(grayed, scale, num_pages * scale)

    mask = vertical + horizontal

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return process_contours(contours, grayed, num_pages, mask)


def process_contours(contours, grayed, num_pages, joints):
    tables = []
    image_area = (grayed.shape[0] / num_pages) * grayed.shape[1]
    for contour in contours[::-1]:
        area = cv2.contourArea(contour)
        if area < 100 or area > 0.8 * image_area:
            continue
        contour_poly = cv2.approxPolyDP(contour, 3, True)
        x, y, w, h = cv2.boundingRect(contour_poly)
        roi = joints[y:y + h, x:x + w]
        joint_contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(joint_contours) < 4:
            continue

        table = grayed[y:y + h, x:x + w]
        table_con = joint_contours[0]
        table_x = table_con[:, 0, 0]
        table_max_x = np.max(table_x)
        rows = []
        cells = []
        prev_top_left_bottom = None
        for con in joint_contours[:0:-1]:
            all_x = con[:, 0, 0]
            all_y = con[:, 0, 1]
            cell_min_x, cell_min_y = np.min(all_x), np.min(all_y)
            cell_max_x, cell_max_y = np.max(all_x), np.max(all_y)

            text = check_if_left_out(prev_top_left_bottom, cell_max_x, table_max_x, table)
            if text is not None:
                cells.append(text)
                rows.append(cells)
                cells = []

            cell = table[cell_min_y:cell_max_y, cell_min_x:cell_max_x]
            text = extract_text_from_cell(cell)
            cells.append(text)
            prev_top_left_bottom = (cell_min_y, cell_max_x, cell_max_y)
            if table_max_x == cell_max_x:
                rows.append(cells)
                cells = []

        text = check_if_left_out(prev_top_left_bottom, -1, table_max_x, table)
        if text is not None:
            cells.append(text)
            rows.append(cells)

        if len(rows) > 0:
            common_len = mode(map(lambda r: len(r), rows))
            extracted_table = list(filter(lambda r: len(r) == common_len, rows))
            tables.append(extracted_table)

    return tables


def check_if_left_out(prev_top_left_bottom, cell_max_x, table_max_x, table):
    if prev_top_left_bottom is not None and (cell_max_x < prev_top_left_bottom[1] != table_max_x):
        left_out_min_y = prev_top_left_bottom[0]
        left_out_min_x = prev_top_left_bottom[1]
        left_out_max_y = prev_top_left_bottom[2]
        left_out_cell = table[left_out_min_y:left_out_max_y, left_out_min_x:table_max_x]
        text = extract_text_from_cell(left_out_cell)
        return text
    return None


def extract_text_from_cell(image):
    try:
        (origH, origW) = image.shape[:2]
        (newW, newH) = (int(origW * 1.2), int(origH * 1.2))
        image = cv2.resize(image, (newW, newH))
        image = cv2.threshold(image, 124, 255, cv2.THRESH_TRUNC)[1]
        return pytesseract.image_to_string(image, config='--oem 1 --tessdata-dir "../resources/tessdata/" --psm 6')
    except cv2.error:
        return ""


def get_lines(grayed, horizontal_scale, vertical_scale):
    thresholded = cv2.adaptiveThreshold(~grayed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    horizontal = thresholded.copy()
    vertical = thresholded.copy()

    # horizontal lines
    horizontal_size = horizontal.shape[1] // horizontal_scale
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    cv2.erode(horizontal, horizontal_structure, horizontal, iterations=5)
    cv2.dilate(horizontal, horizontal_structure, horizontal, iterations=5)
    horizontal = np.roll(horizontal, -5)

    # vertical line
    vertical_size = vertical.shape[0] // vertical_scale
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    cv2.erode(vertical, vertical_structure, vertical, iterations=5)
    cv2.dilate(vertical, vertical_structure, vertical, iterations=5)
    vertical = np.roll(vertical, -5, axis=0)

    return vertical, horizontal
