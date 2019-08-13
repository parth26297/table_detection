from collections import defaultdict
from functools import reduce
import cv2
import pandas as pd
import pytesseract


def get_dict(d, c):
    k = c[0][0]
    d[k[1]].append(k)
    return d


def extract_text(table, contours):
    flattened = list(reduce(get_dict, contours, defaultdict(list)).values())

    total_length = len(flattened)
    row_values = []
    for i in range(0, total_length):
        row_length = len(flattened[i])
        cell_values = []
        for j in range(0, row_length):
            current_contour = flattened[i][j]
            left_x = flattened[i][j+1][0] if j+1 < row_length and flattened[i][j][0] != 0 else 0
            top_y = flattened[i+1][j][1] if i + 1 < total_length and flattened[i][j][1] != 0 else 0
            right_x = current_contour[0]
            bottom_y = current_contour[1]
            if left_x != right_x and top_y != bottom_y:
                cell = table[top_y:bottom_y, left_x:right_x]
                image = cell.copy()
                (origH, origW) = image.shape[:2]
                (newW, newH) = (int(origW*1.2), int(origH*1.2))
                image = cv2.resize(image, (newW, newH))
                image = cv2.threshold(image, 124, 255, cv2.THRESH_TRUNC)[1]
                text = pytesseract.image_to_string(image, config='--oem 1 --tessdata-dir "../resources/tessdata/" --psm 6')
                cell_values.append(text)
        if len(cell_values) > 0:
            row_values.append(cell_values[::-1])

    table = row_values[::-1]
    return pd.DataFrame(data=table[1:], columns=table[0])

