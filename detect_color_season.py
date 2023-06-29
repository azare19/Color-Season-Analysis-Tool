import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt

detector = MTCNN()

def plot_histogram(img, title, mask=None):
   # split the image into blue, green and red channels
   channels = cv2.split(img)
   colors = ("b", "g", "r")
   plt.title(title)
   plt.xlabel("Bins")
   plt.ylabel("# of Pixels")
   # loop over the image channels
   for (channel, color) in zip(channels, colors):
      # compute the histogram for the current channel and plot it
      hist = cv2.calcHist([channel], [0], mask, [256], [1, 256])
      plt.plot(hist, color=color)
      plt.xlim([0, 256])

def generate_ranges(images):

    skin_hues = []
    skin_sats = []
    skin_vals = []
    
    hair_hues = []
    hair_sats = []
    hair_vals = []

    eye_hues = []
    eye_sats = []
    eye_vals = []

    for image in images:
        skin_hue, skin_sat, skin_val = skin_detect(image)
        hair_hue, hair_sat, hair_val = hair_detect(image)
        eye_hue, eye_sat, eye_val = eye_detect(image)

        skin_hues.append(skin_hue)
        skin_sats.append(skin_sat)
        skin_vals.append(skin_val)

        hair_hues.append(hair_hue)
        hair_sats.append(hair_sat)
        hair_vals.append(hair_val)

        eye_hues.append(eye_hue)
        eye_sats.append(eye_sat)
        eye_vals.append(eye_val)

    sh_std = np.std(skin_hues)
    sh_avg = np.average(skin_hues)
    
    ss_std = np.std(skin_sats)
    ss_avg = np.average(skin_sats)

    sv_std = np.std(skin_vals)
    sv_avg = np.average(skin_vals)

    hh_std = np.std(hair_hues)
    hh_avg = np.average(hair_hues)
    
    hs_std = np.std(hair_sats)
    hs_avg = np.average(hair_sats)

    hv_std = np.std(hair_vals)
    hv_avg = np.average(hair_vals)

    eh_std = np.std(eye_hues)
    eh_avg = np.average(eye_hues)
    
    es_std = np.std(eye_sats)
    es_avg = np.average(eye_sats)

    ev_std = np.std(eye_vals)
    ev_avg = np.average(eye_vals)

    skin_data = {"hue":(sh_avg,sh_std), "sat":(ss_avg,ss_std), "val":(sv_avg,sv_std)}
    hair_data = {"hue":(hh_avg,hh_std), "sat":(hs_avg,hs_std), "val":(hv_avg,hv_std)}
    eye_data = {"hue":(eh_avg,eh_std), "sat":(es_avg,es_std), "val":(ev_avg,ev_std)}

    return skin_data, hair_data, eye_data



def skin_detect(img):

    #reduce the image size for easier processing
    scale_percent = 25
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    face1 = faces[0]

    x,y,w,h = face1

    face_region = img[y+20:y+h-20,x+20:x+w-20]


    #blur image to generalize shapes
    blur_factor = (10, 10)
    blurred_img = cv2.blur(face_region, blur_factor) 

    #converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCrCb)
    #skin color range for YcrCb color space 
    # Cr - 135-180, Cb - 85-135
    YCrCb_mask = cv2.inRange(img_YCrCb, (80, 135, 100), (255,200,150)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)

    cv2.imwrite("1_HSV.jpg", HSV_result)
    cv2.imwrite("2_YCbCr.jpg", YCrCb_result)
    cv2.imwrite("3_global_result.jpg", global_result)

    masked = cv2.bitwise_and(face_region, face_region, mask=global_mask)
    masked_HSV = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    cv2.imwrite('Skin.jpg', masked)

    '''plot_histogram(masked_HSV, 'hist')
    plt.show()'''

    warm = 0
    cool = 0

    skin_amt = 0 

    saturation = 0
    lightness = 0

    h,w = global_mask.shape
    for i in range(h):
        for j in range(w):
            if global_mask[i,j] != 0:
                skin_amt+=1

                pixel = masked[i,j]
                blue = pixel[0]
                green = pixel[1]
                red = pixel[2]

                pixel_HSV = masked_HSV[i,j]
                sat = pixel_HSV[1]
                val = pixel_HSV[2]

                if red >= green and red >= blue:
                    #dom color is red
                    if blue > np.floor(red/1.5):
                        cool += 1
                    else:
                        warm += 1
                saturation += sat
                lightness+= val
    coolness = cool/(warm+cool)
    saturation = saturation/skin_amt
    lightness = lightness/skin_amt

    return coolness, saturation, lightness


def hair_detect(img):
    #reduce the image size for easier processing
    scale_percent = 25
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    face1 = faces[0]

    x,y,w,h = face1

    face_region = img[y-20:y+h+20,x+20:x+w+20]
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for i in range(y-20,y+h+20):
        for j in range(x+20,x+w-20):
            imgHSV[i][j] = (37,26,26)


    imgMask = cv2.inRange(imgHSV, (32, 0, 0), (100, 255,255))
    imgMask = cv2.morphologyEx(imgMask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    imgMask = cv2.bitwise_not(imgMask)

    masked = cv2.bitwise_and(img, img, mask=imgMask)
    masked_HSV = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

    '''plot_histogram(masked_HSV, 'hist')
    plt.show()'''

    cv2.imwrite("Hair.jpg",masked)

    '''cv2.imshow('masked', masked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    warm = 0
    cool = 0

    warmred = 0
    coolred = 0

    hair_amt = 0 

    saturation = 0
    lightness = 0

    other_colors = 0

    warmblues = np.zeros((300,300,3), dtype="uint8")

    h,w = imgMask.shape
    for i in range(h):
        for j in range(w):
            if imgMask[i,j] != 0:
                hair_amt+=1

                pixel = masked[i,j]
                blue = pixel[0]
                green = pixel[1]
                red = pixel[2]

                pixel_HSV = masked_HSV[i,j]
                hue = pixel_HSV[0]
                sat = pixel_HSV[1]
                val = pixel_HSV[2]

                if red >= green and red >= blue:
                    #dom color is red
                    if blue > np.floor(red/1.5):
                        cool += 1
                        coolred+=1
                    else:
                        warm += 1
                        warmred += 1
                    saturation += sat
                    lightness+= val
                    
                    '''print("count", val)
                    warmblues[:] = (blue, green, red)
                    cv2.imshow('warmblues', warmblues)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()'''
                else:
                    other_colors += 1
                    saturation += sat
                    lightness+= val
                    '''print(val)
                    warmblues[:] = (blue, green, red)
                    cv2.imshow('warmblues', warmblues)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()'''

                '''saturation += sat
                
                lightness+= val'''
    
    print(str(hair_amt))
    print(str(warm+cool))

    coolness = cool/(warm+cool)
    saturation = saturation/(warm+cool)
    lightness = lightness/(warm+cool)
    #print(warm+cool)
    #print("non-red colors: ", other_colors)

    return coolness, saturation, lightness

def eye_color(pixel):
    eye_colors = {
        "green": ((30,53,127),(82,255,216)),
        "blue": ((83,53,127),(120,255,216)),
        "brown": ((1,51,51),(20,255,153)),
        "brown-black": ((0,25,12),(20,102,63)),
        "brown-gray": ((10,7,76),(32,153,153)),
        "green-gray":((30,5,24),(82,52,165)),
        "blue-gray":((83,5,24),(120,52,216))

    }

    for color in eye_colors:
        if pixel[0] >= eye_colors[color][0][0] and pixel[0] <= eye_colors[color][1][0]:
            if pixel[1] >= eye_colors[color][0][1] and pixel[1] <= eye_colors[color][1][1]:
                if pixel[2] >= eye_colors[color][0][2] and pixel[2] <= eye_colors[color][1][2]:
                    return color
    
    return "other"

def eye_detect(img):
    scale_percent = 15
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[0:2]
    imgMask = np.zeros(img.shape[:2], dtype="uint8")

    warmblues = np.zeros((300,300,3), dtype="uint8")
    
    result = detector.detect_faces(img)
    if result == []:
        print('Warning: Can not detect any face in the input image!')
        return

    bounding_box = result[0]['box']
    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']

    eye_distance = np.linalg.norm(np.array(left_eye)-np.array(right_eye))
    eye_radius = eye_distance/15 # approximate
   
    cv2.circle(imgMask, left_eye, int(eye_radius), (255,255,255), -1)
    cv2.circle(imgMask, right_eye, int(eye_radius), (255,255,255), -1)


    cv2.rectangle(img,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (255,155,255),
              2)

    cv2.circle(img, left_eye, int(eye_radius), (0, 155, 255), 1)
    cv2.circle(img, right_eye, int(eye_radius), (0, 155, 255), 1)

    masked = cv2.bitwise_and(img, img, mask=imgMask)
    masked_HSV = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

    cv2.imwrite('eyes.jpg', masked)
    plot_histogram(masked, 'hist')
    plt.show()



    eye_color_counts = {
        "green": 0,
        "blue": 0,
        "brown": 0,
        "brown-black": 0,
        "brown-gray": 0,
        "green-gray": 0,
        "blue-gray": 0,
        "other": 0
    }

    warm = 0
    cool = 0

    eye_amt = 0 

    saturation = 0
    lightness = 0

    for y in range(0, h):
        for x in range(0, w):
            if imgMask[y, x] != 0:
                eye_amt+=1
                pixel = img[y,x]

                pixel_HSV = imgHSV[y,x]
                hue = pixel_HSV[0]
                sat = pixel_HSV[1]
                val = pixel_HSV[2]

                eye_color_counts[eye_color(pixel_HSV)] += 1

                '''print(hue,sat,val)
                print(eye_color(pixel_HSV))
                warmblues[:] = pixel
                cv2.imshow('warmblues', warmblues)
                cv2.waitKey(0)
                cv2.destroyAllWindows()'''

                saturation += sat
                lightness+= val
    
    cool = eye_color_counts["green"] + eye_color_counts["blue"] + eye_color_counts["brown-gray"] + eye_color_counts["green-gray"] + eye_color_counts["blue-gray"]
    warm = eye_color_counts["brown"] + eye_color_counts["brown-black"]

    #print(str(eye_color_counts))
    tot = warm + cool
    coolness = cool/tot
    saturation = saturation/eye_amt
    lightness = lightness/eye_amt
    '''print("cool ratio: ", coolness)
    print("total: ", eye_amt)

    print("saturation: ", saturation)
    print("lightness: ", lightness)'''

    
    
    masked = cv2.bitwise_and(img, img, mask=imgMask)
    '''cv2.imshow('mask', masked)

    cv2.imshow('EYE-COLOR-DETECTION', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    return coolness, saturation, lightness

def color_season(img_str, skin, hair, eye):
    img = cv2.imread(img_str)

    hue, sat, val = skin_detect(img)
    sat = sat/255
    val = val/255
    print("skin: ", hue, sat, val)

    if hue <= .2:
        #more than one std away from average hue, very warm
        s_coolness = 4
    elif hue <= .4:
        #near neutral still warm
        s_coolness = 3
    elif hue <= .6:
        #neutral
        s_coolness = 2
    elif hue <= .8:
        #neutral, leaning cool
        s_coolness = 1
    else:
        #very cool
        s_coolness = 0

    if sat <= .05:
        #not saturated
        s_sat = 0
    elif sat <= .1:
        s_sat = 1
    elif sat <= .2:
        s_sat = 2
    elif sat <= .3:
        s_sat = 3
    else:
        s_sat = 4

    if val <= .2:
        s_val = 4
    elif val <= .4:
        s_val = 3
    elif val <= .6:
        s_val = 2
    elif val <= .8:
        s_val = 1
    else:
        s_val = 0

    
    hue, sat, val = eye_detect(img)
    sat = sat/255
    val = val/255
    print("eye: ", hue, sat, val)

    if hue <= .2:
        #more than one std away from average hue, very warm
        e_coolness = 4
    elif hue <= .4:
        #near neutral still warm
        e_coolness = 3
    elif hue <= .6:
        #neutral
        e_coolness = 2
    elif hue <= .8:
        #neutral, leaning cool
        e_coolness = 1
    else:
        #very cool
        e_coolness = 0

    if sat <= .2:
        #not saturated
        e_sat = 0
    elif sat <= .4:
        e_sat = 1
    elif sat <= .6:
        e_sat = 2
    elif sat <= .8:
        e_sat = 3
    else:
        e_sat = 4

    if val <= .19:
        e_val = 3
    elif val <= .6:
        e_val = 2
    elif val <= .8:
        e_val = 1
    else:
        e_val = 0

    hue, sat, val = hair_detect(img)
    sat = sat/255
    val = val/255
    print("hair: ", hue, sat, val)

    if hue <= .4:
        #more than one std away from average hue, very warm
        h_coolness = 3
    else:
        #very cool
        h_coolness = 0
    '''elif hue <= .4:
        #near neutral still warm
        h_coolness = 3
    elif hue <= .6:
        #neutral
        h_coolness = 2
    elif hue <= .8:
        #neutral, leaning cool
        h_coolness = 1'''
    

    if sat <= .2:
        #not saturated
        h_sat = 0
    elif sat <= .4:
        h_sat = 1
    elif sat <= .6:
        h_sat = 2
    elif sat <= .8:
        h_sat = 3
    else:
        h_sat = 4

    if val <= .24:
        #dark
        h_val = 4
    elif val <= .3:
        h_val = 3
    elif val <= .5:
        h_val = 2
    elif val <= .7:
        h_val = 1
    else:
        h_val = 0

    print(s_coolness,s_sat, s_val)
    print(h_coolness,h_sat, h_val)
    print(e_coolness,e_sat, e_val)

    avg_coolness = (s_coolness + h_coolness + e_coolness) / 3
    avg_sat = (s_sat+h_sat+e_sat)/3
    avg_val = (s_val+h_val+e_val)/3

    print(avg_coolness, avg_sat, avg_val)

    coolness_fac = abs(avg_coolness - 2)
    sat_fac = abs(avg_sat - 2)
    val_fac = abs(avg_val - 2)

    seasons = {
        "True Spring": ((2, 2, 0), (4, 4, 4)),
        "True Autumn": ((2, 0, 0), (4, 2, 4)),
        "True Summer": ((0, 0, 0), (2, 2, 4)),
        "True Winter": ((0, 2, 0), (2, 4, 4)),

        "Bright Spring": ((2, 2.5, 0), (4, 4, 4)),
        "Light Spring": ((2, 0, 0), (4, 4, 1.5)),

        "Light Summer": ((0, 0, 0), (2, 4, 1.5)),
        "Soft Summer": ((0, 0, 0), (2, 1.5, 4)),

        "Soft Autumn": ((2, 0, 0), (4, 1.5, 4)),
        "Deep Autumn": ((2, 0, 2.5), (4, 4, 4)),

        "Deep Winter": ((0, 0, 2.3), (2, 4, 4)),
        "Bright Winter": ((0, 2.3, 0), (2, 4, 4))
    }

    season = "other"

    for s in seasons:
        if avg_coolness >= seasons[s][0][0] and avg_coolness <= seasons[s][1][0]:
            if avg_sat >= seasons[s][0][1] and avg_sat <= seasons[s][1][1]:
                if avg_val >= seasons[s][0][2] and avg_val <= seasons[s][1][2]:
                    season = s

        

    '''if coolness_fac >= sat_fac and coolness_fac >= val_fac:
        #most prominent feature is hue
        if avg_coolness < 2:
            #prominently cool
            if sat_fac >= val_fac:
                if avg_sat < 2:
                    #prominently muted
                    season = "True Summer"
                else:
                    #prominently bright
                    season = "True Winter"
            else:
                if avg_val < 2:
                    #prominently light
                    season = "True Summer"
                else:
                    #prominently deep
                    season = "True Winter"
        else:
            #prominently warm
            if sat_fac >= val_fac:
                if avg_sat < 2:
                    #prominently muted
                    season = "True Autumn"
                else:
                    #prominently bright
                    season = "True Spring"
            else:
                if avg_val < 2:
                    #prominently light
                    season = "True Spring"
                else:
                    #prominently deep
                    season = "True Autumn"
    elif sat_fac >= coolness_fac and sat_fac >= val_fac:
        #most prominent feature is saturation
        if avg_sat < 2:
            #prominently soft 
            if val_fac >= coolness_fac:
                if avg_val < 2:
                    #predominantly light
                    season = "Soft Summer"
                else:
                    #predominantly dark
                    season = "Soft Autumn"
            else:
                if avg_coolness < 2:
                    #predominantly cool
                    season = "Soft Summer"
                else:
                    #predominantly warm
                    season = "Soft Autumn"
        else:
            #predominantly bright
            if val_fac >= coolness_fac:
                if avg_val < 2:
                    #predominantly light
                    season = "Bright Spring"
                else:
                    #predominantly dark
                    season = "Bright Winter"
            else:
                if avg_coolness < 2:
                    #predominantly cool
                    season = "Bright Winter"
                else:
                    #predominantly warm
                    season = "Bright Spring"
    else:
        #most prominent feature is value
        if avg_val < 2:
            #predominantly light
            if coolness_fac >= sat_fac:
                if avg_coolness < 2:
                    #predominantly cool
                    season = "Light Summer"
                else:
                    #predominantly warm
                    season = "Light Spring"
            else:
                if avg_sat < 2:
                    #predominantly soft
                    season = "Light Summer"
                else:
                    #predominantly bright
                    season = "Light Spring"
        else:
            #predominantly dark
            if coolness_fac >= sat_fac:
                if avg_coolness < 2:
                    #predominantly cool
                    season = "Deep Winter"
                else:
                    #predominantly warm
                    season = "Deep Autumn"
            else:
                if avg_sat < 2:
                    #predominantly soft
                    season = "Deep Autumn"
                else:
                    season = "Deep Winter"'''


    return season, avg_coolness, avg_sat, avg_val





'''images = [
    "andrew.jpg",
    "bright_winter_1.jpg",
    "cole.jpg",
    "deep_autumn_1.jpg",
    "deep_winter_1.jpg",
    "IMG_7127.jpg",
    "IMG_7134.jpg",
    "jackson.jpg",
    "kat.jpg",
    "logan.jpg",
    "max.jpg",
    "melina.jpg",
    "soft_autumn_1.jpg",
    "true_autumn_1.jpg",
    "true_spring_1.jpg",
    "true_summer_1.jpg"
]

imgs = []

for image in images:
    img = cv2.imread(image)
    imgs.append(img)

skin, hair, eye = generate_ranges(imgs)'''

skin, hair, eye = ({'hue': (0.4218063557405385, 0.18708928108280468), 
'sat': (87.51329903845851, 8.673653870875297), 
'val': (193.80337315128074, 7.8162702687947725)}, 

{'hue': (0.29108052932157163, 0.31190584775640595), 
'sat': (107.3509823694851, 26.301180978347055), 
'val': (101.21118099148342, 47.50332759323774)},
 
{'hue': (0.45022591154145464, 0.35520934996819603), 
'sat': (67.77817536692963, 33.272686324245264), 
'val': (64.81932282284895, 17.765264951021784)})

'''season = color_season("cole2.jpg", skin, hair, eye)

print(season)'''


