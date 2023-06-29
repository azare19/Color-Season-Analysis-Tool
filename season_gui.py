import PySimpleGUI as sg
import cv2
import skin_detect as analyze
from matplotlib import pyplot as plt

images = [
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

'''for image in images:
    img = cv2.imread(image)
    imgs.append(img)

skin, hair, eye = analyze.generate_ranges(imgs)'''

sg.theme("DarkTeal2")
layout = [[sg.T("")], [sg.Text("Choose a file: "), sg.Input(key="-IN2-" ,change_submits=True), sg.FileBrowse(key="-IN-")],[sg.Button("Submit")]]

###Building Window
window = sg.Window('My File Browser', layout, size=(600,150))
    
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event=="Exit":
        break
    elif event == "Submit":
        season, coolness, sat, lightness = analyze.color_season(values["-IN-"], skin=1, hair=1, eye=1)
        x = [0,1,2,3,4]
        plt.plot(x,[0.5,0.5,0.5,0.5,0.5])
        plt.annotate('cool',(0,1.5))
        plt.annotate('warm',(4,1.5))

        plt.plot(x,[1,1,1,1,1])
        plt.annotate('light',(0,1))
        plt.annotate('dark',(4,1))

        plt.plot(x,[1.5,1.5,1.5,1.5,1.5])
        plt.annotate('soft',(0,0.5))
        plt.annotate('bright',(4,0.5))

        plt.plot(coolness,1.5,'ro')
        plt.plot(sat,0.5,'ro') 
        plt.plot(lightness,1,'ro') 

        print(season)

        result = season + ".png"
        result = cv2.imread(result)
        cv2.imshow('result', result)
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
        
        