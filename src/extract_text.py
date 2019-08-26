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
    bottom_right = flattened[0][0]
    top_left = flattened[len(flattened)-1][len(flattened[len(flattened)-1])-1]
    table = table[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    total_length = len(flattened)
    row_values = []
    for i in range(0, total_length):
        row_length = len(flattened[i])
        cell_values = []
        for j in range(0, row_length):
            current_contour = flattened[i][j]
            left_x = flattened[i][j+1][0] if j+1 < row_length and flattened[i][j][0] != top_left[0] else top_left[0]
            top_y = flattened[i+1][j][1] if i + 1 < total_length and j < len(flattened[i+1]) and flattened[i][j][1] != top_left[1] else top_left[1]
            right_x = current_contour[0]
            bottom_y = current_contour[1]

            if right_x-left_x > 20 and bottom_y-top_y > 10:
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

    from statistics import mode
    if len(row_values) > 0:
        common_len = mode(map(lambda x: len(x), row_values))
        table = list(filter(lambda x: len(x) == common_len, row_values[::-1]))
        return pd.DataFrame(data=table[1:], columns=table[0])
    else:
        return None

