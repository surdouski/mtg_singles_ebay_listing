import cv2
import imagehash as ih
import numpy as np
import pandas as pd
from PIL import Image
import decimal
import json
import ast

#  from mtg_card_detector.mtg_card_detector import main, _invalid_path, command_line_args


def order_points(pts):
    """Initialize a list of coordinates that will be ordered such that the first entry in the list is the top-left,
        the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left.

    Parameters
    ----------
    pts : np.array

    Returns
    -------
    : ordered list of 4 points
    """

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # the top-left point will have the smallest sum, whereas
    rect[2] = pts[np.argmax(s)]  # the bottom-right point will have the largest sum

    diff = np.diff(pts, axis=1)     # now, compute the difference between the points, the
    rect[1] = pts[np.argmin(diff)]  # top-right point will have the smallest difference,
    rect[3] = pts[np.argmax(diff)]  # whereas the bottom-left will have the largest difference
    return rect


def four_point_transform(image, pts):
    """Transform a quadrilateral section of an image into a rectangular area.

    Parameters
    ----------
    image : Image
        source image
    pts : np.array

    Returns
    -------
    Image
        Transformed rectangular image
    """

    spacing_around_card = 20
    double = 2
    double_spacing_around_card = double * spacing_around_card

    rect = order_points(pts)
    max_height, max_width = get_edges(double_spacing_around_card, rect)

    transformed_image = warp_image(image, max_height, max_width, rect, spacing_around_card)
    if _image_is_horizontal(max_width, max_height):
        transformed_image = rotate_image(max_height, max_width, transformed_image)
    return transformed_image


def rotate_image(max_height, max_width, transformed_image):
    center = (max_height / 2, max_height / 2)
    rotated_applied_transformation_matrix = cv2.getRotationMatrix2D(center, 270, 1.0)
    transformed_image = cv2.warpAffine(transformed_image, rotated_applied_transformation_matrix, (max_height, max_width))
    return transformed_image


def _image_is_horizontal(max_width, max_height):
    return max_width > max_height


def warp_image(image, max_height, max_width, rect, spacing_around_card):
    transformation_array = np.array([
        [0 + spacing_around_card, 0 + spacing_around_card],
        [max_width - spacing_around_card, 0 + spacing_around_card],
        [max_width - spacing_around_card, max_height - spacing_around_card],
        [0 + spacing_around_card, max_height - spacing_around_card]
    ],
        dtype="float32"
    )
    applied_transformation_matrix = cv2.getPerspectiveTransform(rect, transformation_array)
    warped_matrix = cv2.warpPerspective(image, applied_transformation_matrix, (max_width, max_height))
    return warped_matrix


def get_edges(double_spacing_around_card, rect):
    (tl, tr, br, bl) = rect
    max_width = max(int(get_edge(bl, br)), int(get_edge(tl, tr)))
    max_width += double_spacing_around_card
    max_height = max(int(get_edge(br, tr)), int(get_edge(bl, tl)))
    max_height += double_spacing_around_card
    return max_height, max_width


def get_edge(bl, br):
    return np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))


def find_card(img, thresh_c=5, kernel_size=(3, 3), size_thresh=10000):
    """Find contours of all cards in the image

    Parameters
    ----------
    img : Image
        source image
    thresh_c : int
        value of the constant C for adaptive thresholding
    kernel_size : tuple(int, int)
        dimension of the kernel used for dilation and erosion
    size_thresh : int
        threshold for size (in pixel) of the contour to be a candidate

    Returns
    -------
    list
    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, thresh_c)

    kernel = np.ones(kernel_size, np.uint8)
    img_dilate = cv2.dilate(img_thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return _fetch_candidate_contours(contours, hierarchy, size_thresh)


def _fetch_candidate_contours(contours, hierarchy, size_thresh):
    """
    TODO: CLEANUP
    """

    contours_rect = []
    if not contours:
        return contours_rect

    stack = [
        (0, hierarchy[0][0]),
    ]

    while len(stack) > 0:
        i_cnt, h = stack.pop()
        i_next, i_prev, i_child, i_parent = h
        if i_next != -1:
            stack.append((i_next, hierarchy[0][i_next]))
        cnt = contours[i_cnt]
        size = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if size >= size_thresh and len(approx) == 4:
            contours_rect.append(approx)
        else:
            if i_child != -1:
                stack.append((i_child, hierarchy[0][i_child]))
    return contours_rect


def detect_frame(image, card_pool, hash_size=32, size_thresh=10000):
    """Identify all cards in the input frame, display or save the frame if needed

    Parameters
    ----------
    image : Image
        input frame
    card_pool : pd.DataFrame
        pandas dataframe of all card's information
    hash_size : int
        param for pHash algorithm (default is 32)
    size_thresh : int
        threshold for size (in pixel) of the contour to be a
        candidate (default is 10000)
    out_path : str
        path to save the result (default is None)
    display : bool
        flag for displaying the result (default is True)
    debug : bool
        flag for debug mode (default is False)

    Returns
    -------
    list
        list of detected card's name/set and resulting image
    """

    image_result = image.copy()  # For displaying and saving
    detected_cards = []
    contours = find_card(image_result, size_thresh=size_thresh)
    for n, contour in enumerate(contours):
        rectangle_points = get_rectangle_points_from_contour(contour)

        card_image = four_point_transform(image, rectangle_points)
        card_image_object = Image.fromarray(card_image.astype('uint8'), 'RGB')

        card = fetch_best_fit_card(card_image_object, card_pool, hash_size)

        card_name = card['name']
        card_set = card['set']
        card_price_usd = extract_card_price(card)
        card_label = f"{card_name}, {card_set}, {card_price_usd}"
        detected_cards.append((card_name, card_set))

        _write_then_save_image(card_label, contour, image_result, rectangle_points)
        _write_then_save_card_image(card_image, card_name, card_label, n)
    return detected_cards, image_result


def extract_card_price(card):
    prices = card.get('prices')
    if not prices:
        return 'price unknown'
    if isinstance(prices, str):
        _p = prices.replace("\'", "\"").replace("None", "null")     # this is needed due to unusual
        prices = json.loads(_p)                                     # response types from api
    usd_price = decimal.Decimal(prices['usd'])
    return usd_price


def _write_then_save_card_image(card_image, card_name, card_label, n):
    cv2.putText(card_image, card_label, (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (143, 0, 255), 2)
    cv2.imwrite(f'images/output/{card_name}_{n}.jpg', card_image)


def _write_then_save_image(card_label, contour, image_result, rectangle_points):
    cv2.drawContours(image_result, [contour], -1, (0, 255, 0), 2)
    cv2.putText(
        image_result,
        card_label,
        _minumum_width_by_minimum_height(rectangle_points),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (143, 0, 255),
        2
    )


def _minumum_width_by_minimum_height(rectangle_points):
    return (minimum_width(rectangle_points), minimum_height(rectangle_points))


def get_rectangle_points_from_contour(contour):
    rectangle_points = np.float32([p[0] for p in contour])
    return rectangle_points


def minimum_height(rectangle_points):
    return min(rectangle_points[0][1], rectangle_points[1][1])


def minimum_width(rectangle_points):
    return min(rectangle_points[0][0], rectangle_points[1][0])


def fetch_best_fit_card(card_image, card_pool, hash_size):
    card_hash = create_and_flatten_perceptual_hash_from_card_image(card_image, hash_size)
    card_pool['hash_diff'] = card_pool['card_hash_%d' % hash_size]
    card_pool['hash_diff'] = card_pool['hash_diff'].apply(lambda x: np.count_nonzero(x != card_hash))
    return card_pool[card_pool['hash_diff'] == min(card_pool['hash_diff'])].iloc[0]


def create_and_flatten_perceptual_hash_from_card_image(card_image, hash_size):
    return ih.phash(card_image, hash_size=hash_size).hash.flatten()


def _transform_from_rectangular_card_to_image(transformed_image):
    return Image.fromarray(transformed_image.astype('uint8'), 'RGB')
