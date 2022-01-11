import cv2
import numpy as np
import math
from skimage import io
from skimage.filters import  sobel_v,threshold_otsu,sobel
from scipy import ndimage
import pytesseract

class LicenceDetection:
    harris_corner = True
    increase_number = 20
    debug = False

    @staticmethod
    def character_segmentation(image):
        lpr = image.copy()
        thrs = threshold_otsu(lpr)
        lpr = lpr
        lpr[lpr <= thrs] = 0
        lpr[lpr > thrs] = 255
        contours, hier = cv2.findContours(lpr, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        characters = []
        available_text = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        tess_config = f"-c tessedit_char_whitelist={available_text} --psm 10"
        detected_lpr_text = ''
        for contour in contours:
        # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            if w * h < 800 and w*h > 200:

                # Getting ROI
                roi = lpr[y:y+h, x:x+w]
                roi = np.pad(roi,1, constant_values= 255)
                roi = cv2.erode(roi,None)
                text = pytesseract.image_to_string(roi, config=tess_config, lang="eng").split()
                text = pytesseract.image_to_string(roi, config=tess_config, lang="eng").split()
                if len(text) == 1 and text[0] in available_text:
                    characters.append(roi)
                    detected_lpr_text += text[0]
        
        #concatenate segmented characters
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in characters])[0]
        characters_comb = []
        for char in characters:
            characters_comb.append(cv2.resize(char,min_shape, interpolation = cv2.INTER_CUBIC))

        segmented_char = cv2.hconcat(characters_comb)
        tess_config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6"
        text = pytesseract.image_to_string(lpr, config=tess_config, lang="eng")
        print('Full Licence Photo', text)
        print('Segmented Character' ,detected_lpr_text)
        return lpr, segmented_char, detected_lpr_text


    @staticmethod
    def license_detect(image, gray_img):
        v_edges = LicenceDetection.detect_vertical_edges(image)
        weighted_edges = LicenceDetection.get_weighted_edges(v_edges)
        initial_roi_regions = LicenceDetection.initial_roi_region(weighted_edges,image)
        roi_region = LicenceDetection.get_best_region(initial_roi_regions, weighted_edges, image)
        if roi_region[0] < LicenceDetection.increase_number:
            roi_region[0] = 0
        else:
            roi_region[0] -= LicenceDetection.increase_number
        if image.shape[0] - roi_region[1] < LicenceDetection.increase_number:
            roi_region[1] = image.shape[0]
        else:
            roi_region[1] += LicenceDetection.increase_number

        lpr_detected = LicenceDetection.extract_license(gray_img[roi_region[0]: roi_region[1], :])
        if lpr_detected is None:
            return None, None, None
        lpr, segmented_char, ocr_output = LicenceDetection.character_segmentation(lpr_detected)
        return lpr_detected, lpr, segmented_char, ocr_output
    
    @staticmethod
    def extract_license(image):
        
        image = image.astype('uint8')
        gray = cv2.bilateralFilter(image, 11, 17, 17)

        # Detect edges and binarize image
        edged = sobel(gray)
        th = threshold_otsu(np.abs(edged))
        edged = edged > th
        edged = edged.astype('uint8')
        if LicenceDetection.debug:
            io.imshow(edged)
            io.show()

        # Find contours based on Edges
        cnts  = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        # Top 30 Contours
        img2 = image.copy()
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:50]

        for cnt in cnts:
            rect = cv2.minAreaRect(cnt)
            box = np.int0(cv2.boxPoints(rect))
            _, start, _, end = box
            w = np.abs(end[0] - start[0])
            h = np.abs(end[1] - start[1])
            if w > h  and w > 150  :
                x_start = min(start[0], end[0])
                x_end = max(start[0], end[0])
                lpr_detected = image[:, x_start:x_end]
                x_sorted = box[np.argsort(box[:,0])]
                start_edges = x_sorted[0:2]
                end_edges = x_sorted[2:]
                start =start_edges[np.argsort(start_edges[:,1])][0]
                end = end_edges[np.argsort(end_edges[:,1])][0]
            
                angle = np.rad2deg(np.arctan2(
                    end[1] - start[1], end[0] - start[0]))
                lpr_detected = ndimage.rotate(lpr_detected, angle, cval=255)
                return lpr_detected
                 




    @staticmethod
    def detect_vertical_edges(image):
        v_edges = np.abs(sobel_v(image))
        return v_edges

    @staticmethod
    def get_weighted_edges(v_edges):
        imgMean = np.mean(v_edges.reshape(-1))
        thresh = imgMean + 3.5 * imgMean

        prevX = 0
        prevY = 0
        dist = 0
        weighted_edges = np.copy(v_edges)
        for x in range(v_edges.shape[1]):
            for y in range(v_edges.shape[0]):
                if v_edges[y, x] > thresh:
                    if dist == 0:
                        prevX = x
                        prevY = y
                        dist = 1
                        weighted_edges[y,x] = 1
                    else:
                        dist = math.sqrt((prevX-x)**2 + (prevY-y)**2)
                        if dist < 15:
                            weighted_edges[y,x] = 1
                        else:
                            weighted_edges[y,x] = 0.5
                else:
                    weighted_edges[y,x] = 0
        return weighted_edges

    @staticmethod
    def initial_roi_region(weighted_edges, gray_img):
        
        # Threshold rows depending on edges variance
        row_var = np.var(weighted_edges, axis=1)
        thresh = max(row_var)/3
        roi_img = np.zeros(weighted_edges.shape)
        roi_img[row_var>thresh, :] = gray_img[row_var>thresh, :]

        # Get ROI regions and then filter them
        roi_sum = np.sum(roi_img, axis=1)
        roi_start = 0
        roi_end = 0
        roi_regions = []

        inRegion = False
        for i in range(len(roi_sum)):
            if roi_sum[i] != 0 and inRegion == False:
                if len(roi_regions) != 0 and i-roi_regions[-1][1] < 10:
                    roi_start,_ = roi_regions.pop()
                else:
                    roi_start = i
                inRegion = True
            if roi_sum[i] == 0 and inRegion == True:
                roi_end = i-1
                inRegion = False
                
                if roi_end - roi_start >15:
                    roi_regions.append([roi_start, roi_end])

        if LicenceDetection.debug:
            print(roi_regions)
        if len(roi_regions) == 0 or roi_regions[-1][0] != roi_start:
            roi_regions.append([roi_start,roi_end])

        filtered_regions = []
        for region in roi_regions:
            if region[1] - region[0] > 10 and region[1] - region[0] < gray_img.shape[0]/3:
                filtered_regions.append(region)
        
        return filtered_regions

    @staticmethod
    def get_best_region(roi_regions,weighted_edges,img):

        if len(roi_regions) == 0 : return [0,img.shape[0]]
        if len(roi_regions) == 1 : return roi_regions[0];
        
        best_region = 0
        best_weight = 0
        
        if LicenceDetection.harris_corner:

            for i in range(len(roi_regions)):
                region_image = img[roi_regions[i][0]:roi_regions[i][1],:]
                if LicenceDetection.debug:
                    io.imshow(region_image)
                    io.show()
                gray = np.float32(region_image)
                dst = cv2.cornerHarris(gray,4,7,0.2)
                dst = cv2.dilate(dst,None)
                test_img = np.zeros(region_image.shape)
                test_img[dst>0.25*dst.max()]=255
           
                region_weight = 0
                start = test_img.shape[1]//4
                end = test_img.shape[1] - start
                for k in range(test_img.shape[0]):
                    prev_edge = 0
                    for j in range(0,test_img.shape[1]):
                        if test_img[k][j] == 255:
                            current_weight = 1/50
                            if j > start and j < end:
                                current_weight = 50
                            if prev_edge == 0:
                                region_weight += 1 * current_weight * k
                                prev_edge = j
                            else:
                                dist = math.sqrt((prev_edge-j)**2)
                                prev_edge = j
                                region_weight += 1/np.exp(dist) * current_weight * k
                if LicenceDetection.debug:
                    print('normal weight ',region_weight)
                    print('height ',region_weight/(roi_regions[i][1]-roi_regions[i][0]))
                region_weight /= (roi_regions[i][1]-roi_regions[i][0])
                if best_weight < region_weight:
                    best_weight = region_weight
                    best_region = i
        else:
            
            #Edge Power calculation
            for i in range(len(roi_regions)):
                region_image = weighted_edges[roi_regions[i][0]:roi_regions[i][1],:]
                region_weight = 0
                for k in range(region_image.shape[0]):
                    prev_edge = 0
                    for j in range(50,region_image.shape[1]-200):
                        if region_image[k][j] != 0:
                            if prev_edge == 0:
                                region_weight += 1
                                prev_edge = j
                            else:
                                dist = math.sqrt((prev_edge-j)**2) 
                                prev_edge = j
                                region_weight += 1/np.exp(dist)
                
                if best_weight < region_weight:
                    best_weight = region_weight
                    best_region = i
        

        return roi_regions[best_region]

