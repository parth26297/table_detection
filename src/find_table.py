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
    return process_contours(contours, grayed, num_pages, mask)


def process_contours(contours, grayed, num_pages, joints):
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
        if not (2 <= len(external_contours) < 4) or (w*h) < 80000:
            continue

        table = grayed[top:bottom, left:right]
        t_vertical, t_horizontal = get_lines(table, 30, 20)
        # t_mask = t_horizontal + t_vertical
        get_possible_tables(t_vertical, t_horizontal)
        # cv2.imshow("table", t_mask)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # joint_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # table_data = process_table(table, joint_contours)
        # if table_data is not None and len(table) > 0:
        #     tables.append(table_data)
    return tables


def process_table(table, joint_contours):
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
        try:
            common_len = mode(map(lambda r: len(r), rows))
            return list(filter(lambda r: len(r) == common_len, rows))
        except StatisticsError:
            print("Error with mode!!")
    return None


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


def get_possible_tables(vertical, horizontal, grayed=None, num_pages=None):
    # mask = vertical + horizontal
    # cv2.imwrite("temp.jpg", mask)
    vertical_lines = cv2.HoughLinesP(vertical, 1, np.pi/180, 100)
    horizontal_lines = cv2.HoughLinesP(horizontal, 1, np.pi/180, 100)

    vertical_lines = sorted(vertical_lines, key=lambda x: x[0][0])
    horizontal_lines = sorted(horizontal_lines, key=lambda x: -x[0][1])
    [v_min_x, v_max_y] = vertical_lines[0][0][:2]
    [v_max_x, v_min_y] = vertical_lines[len(vertical_lines)-1][0][2:]
    if len(vertical_lines) < 3 and len(horizontal_lines) < 4:
        return None

    [h_min_x, h_max_y] = horizontal_lines[0][0][:2]
    [h_max_x, h_min_y] = horizontal_lines[len(horizontal_lines) - 1][0][2:]
    if abs(v_min_x - h_min_x) < 10 and abs(v_min_y - h_min_y) < 10:
        h_roll = v_min_x - h_min_x
        v_roll = h_min_y - v_min_y
        hr = np.roll(horizontal, h_roll)
        vr = np.roll(vertical, v_roll, axis=0)
        cv2.imshow("new mask", hr+vr)
        cv2.waitKey()
        cv2.destroyAllWindows()
        # left = min(v_min_x, h_min_x)
        # top = min(v_min_y, h_min_y)
        # right = max(v_max_x, h_max_x)
        # bottom = max(v_max_y, h_max_y) if abs(v_max_y-h_max_y) < 10 else min(v_max_y, h_max_y)

        # mini_hr = horizontal[top:bottom + 1, left:right + 1]
        # mini_vr = vertical[top:bottom + 1, left:right + 1]
        # mini_hr = np.roll(mini_hr, h_roll)
        # mini_vr = np.roll(mini_vr,  v_roll, axis=0)
        # roi_mask = mini_vr + mini_hr
        # roi_table = grayed[top:bottom, left:right]
        # table = test(roi_mask, roi_table)
        # if table is not None:
        #     tables.append(table)
