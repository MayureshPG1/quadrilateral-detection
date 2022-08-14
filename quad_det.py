from enum import auto
import numpy as np
import cv2

def color_quantization(img, k):
# Defining input data for clustering
  data = np.float32(img).reshape((-1, 3))
# Defining criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
# Applying cv2.kmeans function
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result

def get_edges_med(img,filtersize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_1 = cv2.medianBlur(gray, filtersize)
    edges_med7 = cv2.adaptiveThreshold(gray_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    edges_med7 = ~edges_med7
    return edges_med7

def get_edges_quant_med7(img):
    res = color_quantization(img,10)
    gray7 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray_7 = cv2.medianBlur(gray7, 7)
    edges_quant_med7 = cv2.adaptiveThreshold(gray_7, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    edges_quant_med7 = ~edges_quant_med7
    return edges_quant_med7

def get_edges_med3_multi(img):
    gray_med = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for _ in range(10): 
        gray_med = cv2.medianBlur(gray_med, 3)

    edges_med3_multi = cv2.adaptiveThreshold(gray_med, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    edges_med3_multi = ~edges_med3_multi

    edges_canny = auto_canny(gray_med)
    edges_skel = get_skeleton(edges_med3_multi)

    cv2.imshow('gray_med',gray_med)
    cv2.imshow('edges_med3_multi',edges_med3_multi)
    cv2.imshow('edges_canny',edges_canny)
    cv2.imshow('edges_skel',edges_skel)
    cv2.waitKey(0)
    return edges_med3_multi

def get_edges_bilateral(img):
    color = cv2.bilateralFilter(img, d=9, sigmaColor=200,sigmaSpace=200)
    gray2 = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    edges_bilateral = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    edges_bilateral = ~edges_bilateral
    return edges_bilateral

def get_edges_bilateral_multi(img):
    img_color = img.copy()
    for _ in range(25): 
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7) 
    gray3 = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    edges_multi_bilateral = cv2.adaptiveThreshold(gray3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    edges_multi_bilateral = ~edges_multi_bilateral
    return edges_multi_bilateral

def auto_canny(image, sigma=0.33, detailedEdges = False):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	if detailedEdges:
		edged = cv2.Canny(image, lower, upper,apertureSize=5,L2gradient=True)
	else:
		edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

import docdetect
def houghRectDetect(edgesImg):
    # print('detecting lines')
    lines = docdetect.detect_lines(edgesImg,hough_thr=10,group_similar_thr=10)
    print('detecting lines', lines)
    print('detecting intersections')
    intersections = docdetect.find_intersections(lines, edgesImg)
    print('detecting quads')
    return docdetect.find_quadrilaterals(intersections)

def generalized_Hough(img, ip_edges, template_edges):
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # template = cv2.imread("template.png")
    height, width = template_edges.shape[:2]

    # edges = cv2.Canny(template, 200, 250)
    edges = template_edges
    ght = cv2.createGeneralizedHoughGuil()
    ght.setTemplate(edges)

    ght.setMinDist(100)
    ght.setMinAngle(0)
    ght.setMaxAngle(360)
    ght.setAngleStep(1)
    ght.setLevels(360)
    ght.setMinScale(0.6)
    ght.setMaxScale(1.1)
    ght.setScaleStep(0.05)
    ght.setAngleThresh(100)
    ght.setScaleThresh(100)
    ght.setPosThresh(100)
    ght.setAngleEpsilon(1)
    ght.setLevels(360)
    ght.setXi(90)

    positions = ght.detect(ip_edges)[0][0]

    for position in positions:
        center_col = int(position[0])
        center_row = int(position[1])
        scale = position[2]
        angle = int(position[3])

        found_height = int(height * scale)
        found_width = int(width * scale)

        rectangle = ((center_col, center_row),
                     (found_width, found_height),
                     angle)

        box = cv2.boxPoints(rectangle)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        for i in range(-2, 3):
            for j in range(-2, 3):
                img[center_row + i, center_col + j] = 0, 0, 255

    cv2.imshow("results.png", img)
    cv2.waitKey(0)

def get_skeleton(img):
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    skel = np.zeros(img.shape,np.uint8)
    size = np.size(img)

    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
    
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True

    return skel

def detect_quadrilaterals(folder_name,filename):

    img = cv2.imread(os.path.join(folder_name,filename))
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_med = gray
    # for _ in range(10): 
    #     gray_med = cv2.medianBlur(gray_med, 3)

    for _ in range(25): 
        img_color = cv2.bilateralFilter(img, 9, 9, 7) 

    gray_bilateral = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # color_bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=200,sigmaSpace=200)
    # gray_bilateral200 = cv2.cvtColor(color_bilateral, cv2.COLOR_BGR2GRAY)

    # gray_bilateral_thresh = cv2.adaptiveThreshold(gray_bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    # gray_bilateral_thresh = ~gray_bilateral_thresh

    # gray_med7 = cv2.medianBlur(gray, 7)

    # edges_med_canny = auto_canny(gray_med)
    edges_bilateral_canny = auto_canny(gray_bilateral)
    # edges_gray_med7_canny = auto_canny(gray_med7)
    # edges_gray_bilateral200_canny = auto_canny(gray_bilateral200)
    # edges_skel = get_skeleton(edges_med3_multi)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_bilateral_canny_closed = cv2.morphologyEx(edges_bilateral_canny, cv2.MORPH_CLOSE, kernel)
    
    # cnt_img = img.copy()
    contours, _ = cv2.findContours(edges_bilateral_canny_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours(edges_bilateral_canny_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    draw1 = img.copy()
    total_area = img.shape[0]*img.shape[1] 

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea( cnt )
        if len(approx) != 4:# or area<total_area*0.05 or area>total_area*0.9 :
            continue
        hull = cv2.convexHull(approx,returnPoints = False)
        defects = cv2.convexityDefects(approx,hull)
        if defects is None:
            cv2.drawContours(draw1, [approx], 0, (0,255,0), 3)

    cv2.imwrite('op_' + filename, draw1)

    # cv2.imshow('gray_bilateral',gray_bilateral)
    # # cv2.imshow('edges_med3_multi',edges_med3_multi)
    # # cv2.imshow('edges_gray_med7_canny',edges_gray_med7_canny)
    # # cv2.imshow('edges_gray_bilateral200_canny',edges_gray_bilateral200_canny)
    # # cv2.imshow('edges_med_canny',edges_med_canny)
    # cv2.imshow('edges_bilateral_canny',edges_bilateral_canny)
    # cv2.imshow('edges_bilateral_canny_closed',edges_bilateral_canny_closed)
    # # cv2.imshow('gray_bilateral_thresh',gray_bilateral_thresh)
    # # cv2.imshow('edges_skel',edges_skel)
    # cv2.waitKey(0)

import docdetect

def docdetect_video(video_path):
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    total_frame_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    total_frames = int(cv2.VideoCapture.get(video, total_frame_id))

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            rects = docdetect.process(frame)
            frame1 = docdetect.draw(rects, frame)
            out.write(frame1)
            cur_frame_id = int(cv2.CAP_PROP_POS_FRAMES)
            cur_frame = int(cv2.VideoCapture.get(video, cur_frame_id))
            print(f'Running on: {cur_frame}/{total_frames}')
        else:
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()


import os
if __name__ == '__main__':
    # Use doc detect library to detect quadrilaterls on a input video
    docdetect_video('data\\box.mp4')

    # Use custom implementation to detect quadrilaterals in an image
    folder_name = 'data'
    for filename in os.listdir(folder_name):
        detect_quadrilaterals(folder_name,filename)


## Test/experimental code below for reference
    # template = cv2.imread('template.jpg')
    # template_edges1 = get_edges_med3_multi(template)
    # template_edges = auto_canny(template_edges1)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # closed = cv2.morphologyEx(template_edges1, cv2.MORPH_ERODE, kernel)
    # closed1 = cv2.morphologyEx(edges_multi_bilateral, cv2.MORPH_CLOSE, kernel)

    # template_skel = get_skeleton(template_edges1)

    # edges_med3_multi = get_edges_med3_multi(img)
    # edges_multi_bilateral = get_edges_bilateral_multi(img)

    # rects = houghRectDetect(edges_med3_multi)
    # cv2.imshow('rects.jpg', rects)
    # cv2.waitKey(0)

    # generalized_Hough(img, edges_med3_multi, template_skel)

    # cv2.imshow('template_edges1.jpg', template_edges1)
    # cv2.imshow('template_skel.jpg', template_skel)
    # cv2.waitKey(0)
    # closed = edges_med3_multi
    # closed1 = edges_multi_bilateral
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # closed = cv2.morphologyEx(edges_med3_multi, cv2.MORPH_CLOSE, kernel)
    # closed1 = cv2.morphologyEx(edges_multi_bilateral, cv2.MORPH_CLOSE, kernel)
    # opened = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    # opened1 = cv2.morphologyEx(closed1, cv2.MORPH_CLOSE, kernel)

    # cnt_img = img.copy()
    # cnt_img1 = img.copy()
    # contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(cnt_img,contours,-1,(0,0,255),-1)
    # cv2.imshow('cnt_img',cnt_img)
    # contours1, _ = cv2.findContours(closed1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(cnt_img1,contours,-1,(0,0,255),-1)
    # cv2.imshow('cnt_img1',cnt_img1)
    # cv2.waitKey(0)
    # ta = img.shape[0]*img.shape[1] 

    # draw1 = img.copy()
    # for cnt in contours:
    #     peri = cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    #     area = cv2.contourArea( cnt )
    #     if len(approx) != 4 or area<ta*0.05 or area>ta*0.9 :
    #         continue
    #     hull = cv2.convexHull(approx,returnPoints = False)
    #     defects = cv2.convexityDefects(approx,hull)
    #     if defects is None:
    #         cv2.drawContours(draw1, [approx], 0, (0,255,0), 3)

    # cv2.imshow('draw1.jpg', draw1)

    # draw2 = img.copy()
    # for cnt in contours1:
    #     peri = cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    #     area = cv2.contourArea( cnt )
    #     if len(approx) != 4 or area<ta*0.05 or area>ta*0.9 :
    #         continue
    #     hull = cv2.convexHull(approx,returnPoints = False)
    #     defects = cv2.convexityDefects(approx,hull)
    #     if defects is None:
    #         cv2.drawContours(draw2, [approx], 0, (0,255,0), 3)

    # cv2.imshow('draw2.jpg', draw2)
    # cv2.waitKey(0)

    # minLineLength = 5
    # maxLineGap = 10

    # line = img.copy()
    # lines = cv2.HoughLinesP(closed,1,np.pi/5,50,minLineLength,maxLineGap)
    # for x in range(0, len(lines)):
    #     for x1,y1,x2,y2 in lines[x]:
    #         cv2.line(line,(x1,y1),(x2,y2),(0,255,0),2)

    # cv2.imshow('houghlines.jpg',line)

    # line1 = img.copy()
    # lines = cv2.HoughLinesP(closed1,1,np.pi/5,50,minLineLength,maxLineGap)
    # for x in range(0, len(lines)):
    #     for x1,y1,x2,y2 in lines[x]:
    #         cv2.line(line1,(x1,y1),(x2,y2),(0,255,0),2)

    # cv2.imshow('houghlines1.jpg',line1)

    # cv2.imshow('orig',img)
    # cv2.imshow('edges_med3_multi',edges_med3_multi)
    # cv2.imshow('edges_multi_bilateral',edges_multi_bilateral)
    # cv2.imshow('closed',closed)
    # cv2.imshow('closed1',closed1)
    # # cv2.imshow('opened',opened)
    # # cv2.imshow('opened1',opened1)
    # # cv2.imshow('edges_med',get_edges_med(img,7))
    # # cv2.imshow('edges_bilateral',get_edges_bilateral(img))
    # # cv2.imshow('edges_quant_med7',get_edges_quant_med7(img))
    # cv2.waitKey(0)