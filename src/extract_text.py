from PIL import Image
import pytesseract
import pandas as pd


def extract_text(table, contours):
    flattened = []
    j = -1
    y = None
    for i in range(0, len(contours)):
        coordinates = contours[i][0][0]
        temp_y = coordinates[1]
        if y is not None and temp_y == y:
            flattened[j].append(coordinates)
        else:
            j = j+1
            y = temp_y
            flattened.append([coordinates])

    total_length = len(flattened)
    cells = []
    row_values = []
    for i in range(0, total_length):
        row_length = len(flattened[i])
        cell_values = []
        for j in range(0, row_length):
            current_contour = flattened[i][j]
            left_contour = flattened[i][j+1] if j+1 < row_length and flattened[i][j][0] != 0 else [0, flattened[i][j][1]]
            top_contour = flattened[i+1][j] if i+1 < total_length and flattened[i][j][1] != 0 else [flattened[i][j][0], 0]
            top_left_contour = [left_contour[0], top_contour[1]]
            x1 = top_left_contour[0]
            y1 = top_left_contour[1]
            x2 = current_contour[0]
            y2 = current_contour[1]
            if x1 != x2 and y1 != y2:
                cell = table[y1:y2, x1:x2]
                cells.append(Image.fromarray(cell, 'L'))
                text = pytesseract.image_to_string(cell, config='--oem 1 --psm 6 -l eng')
                cell_values.append(text)
        if len(cell_values) > 0:
            row_values.append(cell_values[::-1])

    table = row_values[::-1]
    return pd.DataFrame(data=table[1:], columns=table[0])

