import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cv2 import destroyAllWindows 
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np 
from playsound import playsound
import webbrowser
import speech_recognition as sr
from time import sleep
import pyttsx3 as p
import datetime
import os
import pyautogui as pg

anu=p.init()
anu.setProperty('language', 'tr')
rate=anu.getProperty("rate")
anu.setProperty('rate', 130)
print(rate)



def print_utf8_text(image, xy, text, color):  # utf-8 karakterleri
    fontName = 'FreeSerifBoldItalic.ttf'  # 'FreeSansBold.ttf' 
    font = ImageFont.truetype(fontName, 24)  # font seçimi
    img_pil = Image.fromarray(image)  # imajı pillow moduna dönüştür
    draw = ImageDraw.Draw(img_pil)  # imajı hazırla
    draw.text((xy[0],xy[1]), text, font=font,
              fill=(color[0], color[1], color[2], 0))  # b,g,r,a
    image = np.array(img_pil)  # imajı cv2 moduna çevir (numpy.array())
    return image
 

#################################################################

def speak(audio):
    anu.say(audio)
    anu.runAndWait()
def Time():
    time =datetime.datetime.now().strftime("%I:%M:%S")
    speak(time)
def date():
    year =int(datetime.datetime.now().year)
    month =int(datetime.datetime.now().month)
    date =int(datetime.datetime.now().day)
    speak(date)    
    speak(month)
    speak(year)
def wishme():
    speak("current time is,")
    Time()
    speak("current date is,")
    date()
    speak("yeah,what do you want?")

def record():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print("listening...")
        r.pause_threshold =1
        audio=r.listen(source)
        query=""
    try:
        print("Recongniznig...")
        query=r.recognize_google (audio, language='tr-TR')    
        print(query)
    except Exception as e:
        print(e)

        return 'None'
    return query

def response(voice):
    if "ses kontrol" in voice:
        cap = cv2.VideoCapture(0)

        mpHands = mp.solutions.hands 
        hands = mpHands.Hands()
        mpDraw = mp.solutions.drawing_utils
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))

        volMin,volMax = volume.GetVolumeRange()[:2]

        while True:
            success,img = cap.read()
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            height, width, channel = img.shape
            lmList = []
            if results.multi_hand_landmarks:
                for handlandmark in results.multi_hand_landmarks:
                    
                    for id,lm in enumerate(handlandmark.landmark):
                        h,w,_ = img.shape
                        cx,cy = int(lm.x*w),int(lm.y*h)
                        lmList.append([id,cx,cy]) 
                    mpDraw.draw_landmarks(img,handlandmark,mpHands.HAND_CONNECTIONS)
    
            if lmList != []:
                x1,y1 = lmList[4][1],lmList[4][2]
                x2,y2 = lmList[8][1],lmList[8][2]

                cv2.circle(img,(x1,y1),4,(255,0,0),cv2.FILLED)
                cv2.circle(img,(x2,y2),4,(255,0,0),cv2.FILLED)
                cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)

                length = hypot(x2-x1,y2-y1)

                vol = np.interp(length,[15,220],[volMin,volMax])
                print(vol,length)
                volume.SetMasterVolumeLevel(vol, None)

                # Hand range 15 - 220
                # Volume range -63.5 - 0.0
        
            cv2.imshow('Image',img)
            if cv2.waitKey(1)& 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    if "Kapıyı kitle" in voice:
        speak("iyi senden")
    if "Kapıyı aç" in voice:
        speak("iyi senden")
    if "nasılsın" in voice:
        print("iyi senden")
        speak("iyi senden")
    if "s*****" in voice:
        print("iyi senden")
        speak("asıl sen siktir ")    
    if "Ekran görüntüsü al" in voice:
        sH=pg.screenshot()
        fN='sH.jpg' 
        sH.save(fN)
    if "ortam görüntüsü al" in voice:
        count=0
        while True:
            ret, img = kamera.read()
            img = cv2.flip(img, 1)
            cv2.imwrite("Fotoğraflar/görüntü"+ str(count) + ".jpg",img)
            count+=1
            if count >= 1:
                break
        kamera.release()
        cv2.destroyAllWindows()
    if "saat kaç" in voice:
        Time()
    if "arama yap" in voice:
        search=record()
        url='https://google.com/search?q='+search
        webbrowser.get().open(url)
        print(search +' icin bulduklarım')
        speak(search +' icin bulduklarım')
  
    if "YouTube araması yap" in voice:
        search=record()
        url='https://www.youtube.com/results?search_query='+search
        webbrowser.get().open(url)
        print(search +' icin bulduklarım')
        speak(search +' icin bulduklarım')

    if "uykuya dal" in voice:
        speak("görüşmek üzere")
        exit()
    if "Işığı aç" in voice:
       
         sleep(1)
    if "ışığı kapat" in voice:
        
         sleep(0)  
    if "favori müzik parçasını çal" in voice:
        url='https://www.youtube.com/watch?v=rXdas-dN8-o'
        webbrowser.get().open(url)        
    if "radyo aç" in voice:
        url='https://www.youtube.com/watch?v=5J-w9AHKHsc'
        webbrowser.get().open(url)       
        
             
#################################################################







recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Cascade/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
# id sayacını başlat
id = 0
names = ['0', 'Patron','Ali']
 
# Canlı video yakalamayı başlat
kamera = cv2.VideoCapture(0)
kamera.set(3, 1800)  # video genişliğini belirle
kamera.set(4, 800)  # video yüksekliğini belirle
# minimum pencere boyutunu belirle
minW = 0.1 * kamera.get(3)  # genişlik
minH = 0.1 * kamera.get(4)  # yükseklik
while True:
    ret, img = kamera.read()
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    yuzler = faceCascade.detectMultiScale(
        gri,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in yuzler:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, uyum = recognizer.predict(gri[y:y + h, x:x + w])
 
        if (uyum < 50):
            id = names[id]
            uyum = f"Uyum=  {round(uyum,0)}%"
            if(id=='Patron'):
             kamera.release()
             cv2.destroyAllWindows()
             while True:
                voice=record()
                print(voice)
                response(voice)
            
        else:
            id = "bilinmiyor"
            uyum = f"Uyum=  {round(uyum,0)}%"
 
        color = (255,255,255)
        img=print_utf8_text(img,(x + 5, y - 25),str(id),color) # Türkçe karakterler
        # cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(uyum), (x + 5, y + h + 25), font, 1, (255, 255, 0), 1)
 
    cv2.imshow('kamera', img)
    k = cv2.waitKey(10) & 0xff  # Çıkış için Esc veya q tuşu
    if k == 27 or k==ord('q'):
        break
# Belleği temizle
print("\n [INFO] Programdan çıkıyor ve ortalığı temizliyorum")
kamera.release()
cv2.destroyAllWindows()