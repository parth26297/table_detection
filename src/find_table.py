from statistics import mode, StatisticsError

import cv2
import numpy as np
import pytesseract


def extract_tables(num_pages, src):
    grayed = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    scale = 60
    vertical, horizontal = get_lines(grayed, scale, num_pages * scale)

    mask = vertical + horizontal

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return check_and_parse_tables(contours, grayed, num_pages, mask)


def check_and_parse_tables(contours, grayed, num_pages, joints):
    tables = []
    image_area = (grayed.shape[0] / num_pages) * grayed.shape[1]
    for contour in contours[::-1]:
        area = cv2.contourArea(contour)
        if area < 100 or area > 0.8 * image_area:
            continue
        contour_poly = cv2.approxPolyDP(contour, 3, True)
        c = 5
        x, y, w, h = cv2.boundingRect(contour_poly)
        left, right, top, bottom = x-c, x+w+2*c, y-c, y+h+2*c
        roi = joints[top:bottom, left:right]
        external_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not (2 <= len(external_contours) < 6) or (w*h) < 80000:
            continue

        table = grayed[top:bottom, left:right]
        t_vertical, t_horizontal = get_lines(table, 30, 20)
        t_mask = align_lines(t_vertical, t_horizontal)
        if t_mask is not None:
            joint_contours, _ = cv2.findContours(t_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            table_data = parse_table(table, joint_contours)
            if len(table) > 0:
                tables.append(table_data)
    return tables


def parse_table(table, joint_contours):
    table_con = joint_contours[0]
    table_x = table_con[:, 0, 0]
    table_max_x = np.max(table_x)
    rows = []
    cells = []
    for con in joint_contours[:0:-1]:
        cv2.drawContours(table, np.array([con]), -1, (0, 0, 0), 8)
        all_x = con[:, 0, 0]
        all_y = con[:, 0, 1]
        cell_min_x, cell_min_y = np.min(all_x), np.min(all_y)
        cell_max_x, cell_max_y = np.max(all_x), np.max(all_y)

        cell = table[cell_min_y:cell_max_y, cell_min_x:cell_max_x]
        text = extract_text_from_cell(cell)
        cells.append(text)
        if abs(table_max_x - cell_max_x) < 10:
            rows.append(cells)
            cells = []

    if len(rows) > 0:
        try:
            common_len = mode(map(lambda r: len(r), rows))
            return list(filter(lambda r: len(r) == common_len, rows))
        except StatisticsError:
            print("Error with mode!!")
            return rows
    return []


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

    # vertical lines
    vertical_size = vertical.shape[0] // vertical_scale
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    cv2.erode(vertical, vertical_structure, vertical, iterations=5)
    cv2.dilate(vertical, vertical_structure, vertical, iterations=5)

    return vertical, horizontal


def align_lines(vertical, horizontal):
    vertical_lines = cv2.HoughLinesP(vertical, 1, np.pi/180, 100)
    horizontal_lines = cv2.HoughLinesP(horizontal, 1, np.pi/180, 100)
    verticals = {}
    horizontals = {}

    for vl in vertical_lines:
        y1 = vl[0][1]
        y2 = vl[0][3]
        key = "%s%s" % (y1, y2)
        if key not in verticals:
            verticals[key] = []
        verticals[key].append(vl[0])

    for hl in horizontal_lines:
        x1 = hl[0][0]
        x2 = hl[0][2]
        key = "%s%s" % (x1, x2)
        if key not in horizontals:
            horizontals[key] = []
        horizontals[key].append(hl[0])

    vl = max(verticals.values(), key=lambda x: len(x))
    hl = max(horizontals.values(), key=lambda x: len(x))
    if len(vl) < 3 or len(hl) < 4:
        return None

    vertical_lines = sorted(vl, key=lambda x: x[0])
    horizontal_lines = sorted(hl, key=lambda x: x[1])

    [v_min_x, v_min_y] = vertical_lines[0][2:]
    [h_min_x, h_min_y] = horizontal_lines[0][:2]
    if abs(v_min_x - h_min_x) < 10 and abs(v_min_y - h_min_y) < 10:
        h_roll = v_min_x - h_min_x
        v_roll = h_min_y - v_min_y
        hr = np.roll(horizontal, h_roll)
        vr = np.roll(vertical, v_roll, axis=0)
        return hr+vr
    return None
