import cv2
import time
import imutils
import tkinter as tk
from tkinter import messagebox
import threading
import winsound

cam = cv2.VideoCapture(1)
time.sleep(1)

firstFrame = None
area = 500
warning_shown = False
last_warning_time = 0
sound_playing = False

def play_beep_sound():
    duration = 1000  # milliseconds
    frequency = 440  # Hz
    while sound_playing:
        winsound.Beep(frequency, duration)

def show_popup_message():
    global warning_shown, last_warning_time, sound_playing
    current_time = time.time()
    if not warning_shown and current_time - last_warning_time > 10:
        warning_shown = True
        sound_playing = True
        threading.Thread(target=play_beep_sound).start()
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning("Warning", "Moving Object Detected!")
        root.destroy()
        warning_shown = False
        last_warning_time = current_time
        sound_playing = False

while True:
    _, img = cam.read()
    text = "Normal"
    img = imutils.resize(img, width=500)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)
    cv2.imshow("videoStream", img)

    if firstFrame is None:
        firstFrame = gaussianImg
        continue

    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg.copy(), None, iterations=2)

    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object Detected"
        show_popup_message()

    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("cameraFeed", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Ensure that the sound thread is stopped before exiting
sound_playing = False

cam.release()
cv2.destroyAllWindows()
