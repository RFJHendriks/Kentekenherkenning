# Loading the required python modules
from random import randrange

import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
import cv2  # this is opencv module
import glob
import os
import imutils
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
path_for_license_plates = os.getcwd() + '/nummerplaat/**/*.jpg'


# # specify path to the license plate images folder as shown below
# img = cv2.imread('01LVFZ.jpg')
# # license_plate, _ = os.path.splitext(img)
# plt.imshow(img)
# plt.show()
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# plt.imshow(img, cmap='Greys_r')
# plt.show()
# (thresh, blackAndWhiteImage) = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
# plt.imshow(blackAndWhiteImage, cmap='Greys_r')
# plt.show()
#
# path = 'F:/OpenCV/Nummerplaat/'
#

# print(path_for_license_plates)
#
# cv2.imwrite(os.path.join(path, 'test.jpg'), blackAndWhiteImage)
# cv2.waitKey(0)


# test = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
# print(test)

list_license_plates = []
predicted_license_plates = []
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#
counter = 10000
counter_false = 0
for path_to_license_plate in glob.glob(path_for_license_plates, recursive = True):

    license_plate_file = path_to_license_plate.split("/")[-1]
    license_plate_file = license_plate_file.split("\\")[-1]
    license_plate, _ = os.path.splitext(license_plate_file)

#     '''
#     Here we append the actual license plate to a list
#     '''
    list_license_plates.append(license_plate)
#
#     '''
#     Read each license plate image file using openCV
#     '''
    img = cv2.imread(path_to_license_plate)
    img = imutils.resize(img, height= 750)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #HSV waardes om geel te filteren
    lower = np.array([16,100,100],dtype='uint8')
    upper = np.array([25,255,255],dtype='uint8')
    mask = cv2.inRange(hsv, lower, upper)
    # cv2.imshow("color", img)
    # cv2.imshow("mask", mask)
    mask = cv2.GaussianBlur(mask, (5,5),5)

    edged = cv2.Canny(mask, 30, 200)
    # cv2.imshow("edges",edged)

    contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse = True)[:3]

    # cv2.imshow(license_plate, mask)
    # cv2.waitKey(0)
    masked = mask.copy()
    masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018*peri, True)
        # print(license_plate + ' contours length: ' + str(len(approx)))
        if len(approx) == 4:
            masked = cv2.drawContours(masked, c, -1, (0,255,0),3)
            screenCnt = approx
            break

    # cv2.imshow(license_plate, masked)
    # cv2.waitKey(0)

    # ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18))
    #
    # dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
    # contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # im2 = img.copy()

    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    gray = cv2.GaussianBlur(gray, (5,5),0)

    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    edged = cv2.Canny(gray, 30, 200)

    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse= True)[:25]
    # screenCnt= None

    # for c in contours:
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.018*peri, True)
    #     area = cv2.contourArea(c)
    #     print("area="+ str(area) + " image size="+ str(img.size))
    #     if len(approx) == 4:
    #         (x, y, w, h) = cv2.boundingRect(approx)
    #         ratio = w / float(h)
    #         height, _, _ = img.shape
    #         img2 = cv2.drawContours(img, c, -1, (0,255,0),5)
    #         cv2.imshow('blaat', img2)
    #         cv2.waitKey(0)
    #         img2 = cv2.drawContours(img, c,-1,(0,255,0),5)
    #         cv2.imshow('blaat', img2)
    #         cv2.waitKey(0)
    #         screenCnt = approx
    #         break
    #
    # if screenCnt is None:
    #     detected = 0
    #     print("No contour detected")
    # else: detected = 1
    #
    # if detected == 1:
    #     cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
    #
    #     mask = np.zeros(gray.shape,np.uint8)
    #     new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    #     new_image = cv2.bitwise_and(img,img,mask=mask)
    #
    #     (x, y) = np.where(mask ==255)
    #     (topx, topy) = (np.min(x), np.min(y))
    #     (bottomx, bottomy) = (np.max(x), np.max(y))
    #     Cropped = img[topx:bottomx+1, topy:bottomy+1]
    #     CroppedGray = gray[topx:bottomx+1, topy:bottomy+1]
    #     cropped = cv2.resize(CroppedGray, None, fx = 2, fy =2, interpolation = cv2.INTER_CUBIC)
    #     cropped = cv2.GaussianBlur(cropped, (5,5), 0)
    #
    #     text = pytesseract.image_to_string(Cropped, config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 11')
    #     textgray = pytesseract.image_to_string(CroppedGray, config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 11')
    #     textblurred = pytesseract.image_to_string(cropped, config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 11')
    #     text = "".join(text.split()).replace(":","").replace("-", "")
    #     print('werkelijke license plate: ' + license_plate + ' gelezen kleur: ' + text + ' gelezen grijs: ' + textgray + 'blurred: ' + textblurred )
    #
    #
    #
    pt_A = [screenCnt[0][0][0], screenCnt[0][0][1]]
    pt_B = [screenCnt[1][0][0], screenCnt[1][0][1]]
    pt_C = [screenCnt[2][0][0], screenCnt[2][0][1]]
    pt_D = [screenCnt[3][0][0], screenCnt[3][0][1]]
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))
    # print("width: " + str(maxWidth) + "height: " + str(maxHeight))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                     [0, maxHeight - 1],
                     [maxWidth - 1, maxHeight - 1],
                     [maxWidth - 1, 0]])
    # print(input_pts)
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(img,M,(maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    if maxWidth < maxHeight:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    # if len(screenCnt) == 5:
    #     out = cv2.rotate(out, cv2.ROTATE_180)

    path = 'F:/OpenCV/Nummerplaat/'
    imagename = license_plate + '.jpg'
    cv2.imwrite(os.path.join(path + imagename), out)

    (h, w) = out.shape[:2]
    width = 500
    r = width / float(w)
    dim = (width, int(h*r))
    testAfbeelding = cv2.resize(out, dim)

    testAfbeelding = cv2.cvtColor(testAfbeelding,cv2.COLOR_BGR2GRAY)
    testAfbeelding = cv2.GaussianBlur(testAfbeelding, (7,7), 0)
    _,testOTSU = cv2.threshold(testAfbeelding,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    def sort_contours(cnts, reverse = False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts,boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
        return cnts

    #gewenste breedte, hoogte voor karakters
    digit_w, digit_h = 30, 60
    crop_characters = []

    contours, _ = cv2.findContours(testOTSU, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in sort_contours(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        ratio = h/w
        if 1<=ratio<=5.5: #controleren of vorm overeenkomt met wat verwacht mag worden van karakter
            if h/testOTSU.shape[0]>0.5: #controleren of het groot genoeg is om een karakter te zijn
                curr_num = testOTSU[y:y+h,x:x+w] #karakter uitsnijden
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                crop_characters.append(curr_num)
    if len(crop_characters) == 6:
        for i in range(len(crop_characters)):
            if i<=5:
                titel = license_plate[i]
                path = 'F:/OpenCV/Nummerplaat/Karakters/'
                img_name = license_plate[i] + "_" + str(counter) + ".jpg"
                testtext = path + img_name
                cv2.imwrite(os.path.join(path + img_name), crop_characters[i])
                counter += 1
                text_out = pytesseract.image_to_string(crop_characters[i], config='-c tessedit_char_whitelist=ABDEFGHJKLMNOPRSTUVWXZ0123456789 --psm 11')
                print("letter: " + titel + " voorspelling: " + text_out)
                cv2.imshow("test", crop_characters[i])
                cv2.waitKey(0)
    else:
        print(license_plate)
        counter_false += 1






    # for c in sort_contours(contours):
    #     print(c)
    #     (x,y,w,h) = cv2.boundingRect(c)
    #     ratio = w/h
    #     print(w, " height: ", h)
    #     if 1<=ratio<=3.5: #alleen als ratio klopt met wat we verwachten van een karakter
    #         print("hoi")
    #         if h/testOTSU.shape[0]>0.5: #alleen als karakter minstens 50% van de hoogte van het kenteken heeft
    #             cv2.rectangle(test_roi, (x,y), (x+w, y+h), (0,255,0), 2)
    #             curr_num = testOTSU[y:y+h,x:x+w]
    #             curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
    #             _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #             crop_characters.append(curr_num)
    #
    # print("Detect {} letters...".format(len(crop_characters)))

    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1)
    new_image = cv2.bitwise_and(img,img,mask=mask)



    (x, y) = np.where(mask ==255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = new_image[topx:bottomx+1, topy:bottomy+1]

    # Cropped = cv2.getPerspectiveTransform(Cropped, wanted)

    # CroppedGray = gray[topx:bottomx+1, topy:bottomy+1]



    CroppedBlur = cv2.resize(out.copy(), None, fx = 3, fy =3, interpolation = cv2.INTER_CUBIC)
    # CroppedBlur = cv2.GaussianBlur(CroppedBlur, (9,9),0)
    CroppedBlur = cv2.cvtColor(CroppedBlur, cv2.COLOR_BGR2HSV)
    edgescolor = cv2.Canny(out.copy(),100,200)

    lower = np.array([16,120,125],dtype='uint8')
    upper = np.array([25,255,255],dtype='uint8')
    CroppedBlur = cv2.inRange(CroppedBlur, lower, upper)
    edgesGray = cv2.Canny(CroppedBlur.copy(),100,200)
    _,th2 = cv2.threshold(CroppedBlur, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # CroppedBlur = cv2.adaptiveThreshold(CroppedBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    text =          pytesseract.image_to_string(out, config='-c tessedit_char_whitelist=ABDEFGHJKLMNOPRSTUVWXZ0123456789 --psm 13')
    textBlur =      pytesseract.image_to_string(out, config='-c tessedit_char_whitelist=ABDEFGHJKLMNOPRSTUVWXZ0123456789 --psm 11')
    textOtsu = pytesseract.image_to_string(testOTSU, config='-c tessedit_char_whitelist=ABDEFGHJKLMNOPRSTUVWXZ0123456789 --psm 11')
    text = "".join(text.split()).replace(":","").replace("-", "")
    textBlur = "".join(textBlur.split()).replace(":","").replace("-", "")

    # aantalKarakters = format(len(crop_characters))
    # if len(crop_characters)!=6:
    #     print(license_plate)
    #     cv2.imshow('test', testOTSU)
    #     cv2.waitKey(0)

    # print('werkelijke license plate: ' + license_plate + ' gelezen kleur: ' + text + ' blurred: ' +textBlur + ' otsu: ' + textOtsu + 'karakters: ' + aantalKarakters)
    # cv2.imshow("blaat", CroppedBlur)
    # cv2.imshow("blaat2", edgesGray)
    # cv2.waitKey(0)
    # transform_mat = cv2.getPerspectiveTransform(Cropped, [[0,200], [0,0],[0,800],[200,800]])


        # cv2.imshow("test", out)
        # cv2.waitKey(0)
    # if len(screenCnt) == 5:
    #     pt_A = [screenCnt[0][0][0], screenCnt[0][0][1]]
    #     pt_B = [screenCnt[1][0][0], screenCnt[1][0][1]]
    #     pt_C = [screenCnt[2][0][0], screenCnt[2][0][1]]
    #     pt_D = [screenCnt[3][0][0], screenCnt[3][0][1]]
    #     pt_E = [screenCnt[4][0][0], screenCnt[4][0][1]]
    #     input_pts = np.float32([pt_A, pt_B, pt_C, pt_D, pt_E])
    #     print(input_pts)



#
#     '''
#     We then pass each license plate image file
#     to the Tesseract OCR engine using the Python library
#     wrapper for it. We get back predicted_result for
#     license plate. We append the predicted_result in a
#     list and compare it with the original the license plate
#     '''
#     custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng --psm 6'
    # print(pytesseract.image_to_string(RGBimg, lang ='eng', config =custom_config))

#
#     filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
#     predicted_license_plates.append(filter_predicted_result)
#
# print("Actual License Plate", "\t", "Predicted License Plate", "\t", "Accuracy")
# print("--------------------", "\t", "-----------------------", "\t", "--------")
#
# def calculate_predicted_accuracy(actual_list, predicted_list):
#     for actual_plate, predict_plate in zip(actual_list, predicted_list):
#         accuracy = "0 %"
#         num_matches = 0
#         if actual_plate == predict_plate:
#             accuracy = "100 %"
#         else:
#             if len(actual_plate) == len(predict_plate):
#                 for a, p in zip(actual_plate, predict_plate):
#                     if a == p:
#                         num_matches += 1
#                 accuracy = str(round((num_matches / len(actual_plate)), 2) * 100)
#                 accuracy += "%"
#         print("     ", actual_plate, "\t\t\t", predict_plate, "\t\t  ", accuracy)
#
#
# calculate_predicted_accuracy(list_license_plates, predicted_license_plates)
aantalnummerplaten = len(list_license_plates)

print("aantal: " + str(aantalnummerplaten) + " foutief: " + str(counter_false))
