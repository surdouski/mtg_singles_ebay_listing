import cv2
import numpy as np
import pandas as pd
import imagehash as ih
import os
import sys
import math
import random
from PIL import Image
from .. import fetch_data
from .. import transform_data

card_width = 315
card_height = 440

df = fetch_data.load_all_cards_text('%s/csv/rsv.csv' % transform_data.data_dir)
df['art_hash'] = np.NaN
for _, card_info in card_pool.iterrows():
    img_name = '%s/card_img/png/%s/%s_%s.png' % (data_dir, card_info['set'], card_info['collector_number'],
                                                 fetch_data.get_valid_filename(card_info['name']))
    card_img = cv2.imread(img_name)
    if card_img is None:
        fetch_data.fetch_card_image(card_info, out_dir='%s/card_img/png/%s' % (data_dir, card_info['set']))
        card_img = cv2.imread(img_name)
    if card_img is None:
        print('WARNING: card %s is not found!' % img_name)
    img_art = Image.fromarray(card_img[121:580, 63:685])
    card_info['art_hash'] = ih.phash(img_card, hash_size=32, highfreq_factor=4)

print(df['art_hash'])



# Disclaimer: majority of the basic framework in this file is modified from the following tutorial:
# https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/


# www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # If the image is horizontally long, rotate it by 90
    if maxWidth > maxHeight:
        center = (maxHeight / 2, maxHeight / 2)
        M_rot = cv2.getRotationMatrix2D(center, 270, 1.0)
        warped = cv2.warpAffine(warped, M_rot, (maxHeight, maxWidth))

    # return the warped image
    return warped


# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence using non-maxima suppression
def post_process(frame, outs, thresh_conf, thresh_nms):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]


    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > thresh_conf:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = [ind[0] for ind in cv2.dnn.NMSBoxes(boxes, confidences, thresh_conf, thresh_nms)]
    
    ret = [[class_ids[i], confidences[i], boxes[i]] for i in indices]
    return ret


# Draw the predicted bounding box
def draw_pred(frame, class_id, classes, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (class_id < len(classes))
        label = '%s:%s' % (classes[class_id], label)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


def remove_glare(img):
    """
    Inspired from:
    http://www.amphident.de/en/blog/preprocessing-for-automatic-pattern-identification-in-wildlife-removing-glare.html
    The idea is to find area that has low saturation but high value, which is what a glare usually look like.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(img_hsv)
    non_sat = (s < 32) * 255  # Find all pixels that are not very saturated

    # Slightly decrease the area of the non-satuared pixels by a erosion operation.
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    non_sat = cv2.erode(non_sat.astype(np.uint8), disk)

    # Set all brightness values, where the pixels are still saturated to 0.
    v[non_sat == 0] = 0
    # filter out very bright pixels.
    glare = (v > 200) * 255

    # Slightly increase the area for each pixel
    glare = cv2.dilate(glare.astype(np.uint8), disk)
    glare_reduced = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 200
    glare = cv2.cvtColor(glare, cv2.COLOR_GRAY2BGR)
    corrected = np.where(glare, glare_reduced, img)
    return corrected


def find_card(img, thresh_c=5, kernel_size=(3, 3), size_ratio=0.15):
    # Typical pre-processing - grayscale, blurring, thresholding
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, thresh_c)

    # Dilute the image, then erode them to remove minor noises
    kernel = np.ones(kernel_size, np.uint8)
    img_dilate = cv2.dilate(img_thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)

    # Find the contour
    #img_contour = img_erode.copy()
    _, cnts, hier = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        print('no contours')
        return []
    #img_contour = cv2.cvtColor(img_contour, cv2.COLOR_GRAY2BGR)
    #img_contour = cv2.drawContours(img_contour, cnts, -1, (0, 255, 0), 1)
    #cv2.imshow('test', img_contour)

    # For each contours detected, check if they are large enough and are rectangle
    cnts_rect = []
    ind_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)
    for i in range(len(cnts)):
        size = cv2.contourArea(cnts[ind_sort[i]])
        peri = cv2.arcLength(cnts[ind_sort[i]], True)
        approx = cv2.approxPolyDP(cnts[ind_sort[i]], 0.04 * peri, True)
        if size > img.shape[0] * img.shape[1] * size_ratio and len(approx) == 4:
            cnts_rect.append(approx)

    return cnts_rect

    '''
    #card_dim = [630, 880]
    #for cnt in cnts_rect:
    #    pts = np.float32([p[0] for p in cnt])
    #    img_warp = four_point_transform(img, pts)
        
        # Check which side is longer
        len_1 = math.sqrt((cnt[0][0][0] - cnt[1][0][0]) ** 2 + (cnt[0][0][1] - cnt[1][0][1]) ** 2)
        len_2 = math.sqrt((cnt[0][0][0] - cnt[-1][0][0]) ** 2 + (cnt[0][0][1] - cnt[-1][0][1]) ** 2)
        #print(len_1, len_2)

        orig_corner = np.array([p[0] for p in cnt], dtype=np.float32)
        if len_1 > len_2:
            new_corner = np.array([[0, 0], [0, card_dim[1]], [card_dim[0], card_dim[1]], [card_dim[0], 0]], dtype=np.float32)
        else:
            new_corner = np.array([[0, 0], [card_dim[0], 0], [card_dim[0], card_dim[1]], [0, card_dim[1]]],
                                  dtype=np.float32)

        M = cv2.getPerspectiveTransform(orig_corner, new_corner)
        img_warp = cv2.warpPerspective(img, M, (card_dim[0], card_dim[1]))
        
        #cv2.imshow('warp', img_warp)
        #cv2.waitKey(0)
    #img_contour = cv2.drawContours(img_contour, cnts_rect, -1, (0, 255, 0), 3)
    #img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
    #img_erode = cv2.cvtColor(img_erode, cv2.COLOR_GRAY2BGR)
    #img_dilate = cv2.cvtColor(img_dilate, cv2.COLOR_GRAY2BGR)
    #return img_thresh, img_erode, img_contour
    '''

def detect_frame(net, classes, img, thresh_conf=0.5, thresh_nms=0.4, in_dim=(416, 416), display=True, out_path=None):
    img_copy = img.copy()
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(img, 1 / 255, in_dim, [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))

    # Remove the bounding boxes with low confidence
    obj_list = post_process(img, outs, thresh_conf, thresh_nms)
    for obj in obj_list:
        class_id, confidence, box = obj
        left, top, width, height = box
        draw_pred(img, class_id, classes, confidence, left, top, left + width, top + height)

    # Put efficiency information. The function getPerfProfile returns the
    # overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(img, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    if out_path is not None:
        cv2.imwrite(out_path, img.astype(np.uint8))
    if display:
        #no_glare = remove_glare(img_copy)
        #img_concat = np.concatenate((img, no_glare), axis=1)
        cv2.imshow('result', img)
        '''
        for i in range(len(obj_list)):
            class_id, confidence, box = obj_list[i]
            left, top, width, height = box
            img_snip = img_copy[max(0, top):min(img.shape[0], top + height),
                                max(0, left):min(img.shape[1], left + width)]
            img_thresh, img_dilate, img_canny, img_hough = find_card(img_snip)
            img_concat = np.concatenate((img_snip, img_thresh, img_dilate, img_canny, img_hough), axis=1)
            cv2.imshow('feature#%d' % i, img_concat)
        '''
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return obj_list


def detect_video(net, classes, capture, thresh_conf=0.5, thresh_nms=0.4, in_dim=(416, 416), display=True, out_path=None):
    if out_path is not None:
        vid_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                     (round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    max_num_obj = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            # End of video
            print("End of video. Press any key to exit")
            cv2.waitKey(0)
            break
        img = frame.copy()
        obj_list = detect_frame(net, classes, frame, thresh_conf=thresh_conf, thresh_nms=thresh_nms, in_dim=in_dim,
                                display=False, out_path=None)
        #cnts_rect = find_card(img)
        max_num_obj = max(max_num_obj, len(obj_list))
        if display:
            img_result = frame.copy()
            #img_result = cv2.drawContours(img_result, cnts_rect, -1, (0, 255, 0), 2)
            #for i in range(len(cnts_rect)):
            #    pts = np.float32([p[0] for p in cnts_rect[i]])
            #    img_warp = four_point_transform(img, pts)
            #    cv2.imshow('card#%d' % i, img_warp)
            #for i in range(len(cnts_rect), max_num_obj):
            #    cv2.imshow('card#%d' % i, np.zeros((1, 1), dtype=np.uint8))
            #no_glare = remove_glare(img)
            #img_thresh, img_erode, img_contour = find_card(no_glare)
            #img_concat = np.concatenate((no_glare, img_contour), axis=1)

            for i in range(len(obj_list)):
                class_id, confidence, box = obj_list[i]
                left, top, width, height = box
                offset_ratio = 0.1
                x1 = max(0, int(left - offset_ratio * width))
                x2 = min(img.shape[1], int(left + (1 + offset_ratio) * width))
                y1 = max(0, int(top - offset_ratio * height))
                y2 = min(img.shape[0], int(top + (1 + offset_ratio) * height))
                img_snip = img[y1:y2, x1:x2]
                cnts = find_card(img_snip)
                if len(cnts) > 0:
                    cnt = cnts[-1]
                    pts = np.float32([p[0] for p in cnt])
                    img_warp = four_point_transform(img_snip, pts)
                    img_warp = cv2.resize(img_warp, (card_width, card_height))
                    img_card = img_warp[47:249, 22:294]
                    img_card = Image.fromarray(img_card.astype('uint8'), 'RGB')
                    card_hash = ih.phash(img_card, hash_size=32, highfreq_factor=4)
                    print(card_hash - rift_hash)
                    #img_thresh, img_dilate, img_contour = find_card(img_snip)
                    #img_concat = np.concatenate((img_snip, img_contour), axis=1)
                    cv2.rectangle(img_warp, (22, 47), (294, 249), (0, 255, 0), 2)

                    cv2.imshow('card#%d' % i, img_warp)
                else:
                    cv2.imshow('card#%d' % i, np.zeros((1, 1), dtype=np.uint8))
            for i in range(len(obj_list), max_num_obj):
                cv2.imshow('card#%d' % i, np.zeros((1, 1), dtype=np.uint8))
            cv2.imshow('result', img_result)
            #if len(obj_list) > 0:
            #    cv2.waitKey(0)


        if out_path is not None:
            vid_writer.write(frame.astype(np.uint8))
        cv2.waitKey(1)

    if out_path is not None:
        vid_writer.release()
    cv2.destroyAllWindows()


def main():
    # Specify paths for all necessary files
    test_path = os.path.abspath('../data/test4.mp4')
    #weight_path = 'backup/tiny_yolo_10_39500.weights'
    #cfg_path = 'cfg/tiny_yolo_10.cfg'
    #class_path = "data/obj_10.names"
    weight_path = 'weights/second_general/tiny_yolo_final.weights'
    cfg_path = 'cfg/tiny_yolo.cfg'
    class_path = 'data/obj.names'
    out_dir = 'out'
    if not os.path.isfile(test_path):
        print('The test file %s doesn\'t exist!' % os.path.abspath(test_path))
        return
    if not os.path.isfile(weight_path):
        print('The weight file %s doesn\'t exist!' % os.path.abspath(test_path))
        return
    if not os.path.isfile(cfg_path):
        print('The config file %s doesn\'t exist!' % os.path.abspath(test_path))
        return
    if not os.path.isfile(class_path):
        print('The class file %s doesn\'t exist!' % os.path.abspath(test_path))
        return

    thresh_conf = 0.01
    thresh_nms = 0.8

    # Setup
    # Read class names from text file
    with open(class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    # Load up the neural net using the config and weights
    net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Save the detection result if out_dir is provided
    if out_dir is None or out_dir == '':
        out_path = None
    else:
        out_path = out_dir + '/' + os.path.split(test_path)[1]
    # Check if test file is image or video
    test_ext = test_path[test_path.find('.') + 1:]

    if test_ext in ['jpg', 'jpeg', 'bmp', 'png', 'tiff']:
        img = cv2.imread(test_path)
        detect_frame(net, classes, img, out_path=out_path, thresh_conf=thresh_conf, thresh_nms=thresh_nms)
    else:
        capture = cv2.VideoCapture(0)
        detect_video(net, classes, capture, out_path=out_path, thresh_conf=thresh_conf, thresh_nms=thresh_nms)
        capture.release()
    pass


if __name__ == '__main__':
    main()