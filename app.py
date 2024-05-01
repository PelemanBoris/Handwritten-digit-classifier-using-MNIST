from tkinter import *
from keras.models import load_model
from PIL import ImageGrab
import cv2
import numpy as np

CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 700

model = load_model("Model.h5")

last_x, last_y = 0, 0

root = Tk()
root.resizable(False, False)
root.title("Handwritten digit recognizer")

cv = Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
cv.grid(row=0, column=0)

def draw_line(event):
    global last_x, last_y
    x, y = event.x, event.y
    cv.create_line(
        x,
        y,
        x + 1,
        y + 1,
        width=7,
        fill="black",
        capstyle=ROUND,
        smooth=TRUE,
        splinesteps=12,
    )
    last_x, last_y = x, y

def activate_event(event):
    global last_x, last_y
    cv.bind("<B1-Motion>", draw_line)
    last_x, last_y = event.x, event.y

def clear_canvas():
    cv.delete("all")

def recognize_digit():
    x = root.winfo_rootx() + cv.winfo_x()
    y = root.winfo_rooty() + cv.winfo_y()
    x1 = x + CANVAS_WIDTH
    y1 = y + CANVAS_HEIGHT

    ImageGrab.grab(bbox=(x, y, x1, y1)).save("image.png")

    image = cv2.imread("image.png", cv2.IMREAD_COLOR)
    if image is None:
        print("Failed to read image")
        return

    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        digit = th[y : y + h, x : x + w]
        resized_digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(
            resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0
        )
        digit = padded_digit.reshape(1, 28, 28, 1)
        digit = digit / 255.0
        pred = model.predict([digit])[0]
        final_pred = np.argmax(pred)
        data = str(final_pred) + " " + str(int(max(pred) * 100)) + "%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, font_scale, color, thickness)

    cv2.imshow("image", image)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

cv.bind("<Button-1>", activate_event)

btn_clear = Button(root, text="Clear", command=clear_canvas)
btn_clear.grid(row=1, column=0, sticky="w")

btn_recognize = Button(root, text="Recognize", command=recognize_digit)
btn_recognize.grid(row=1, column=0, sticky="e")

root.mainloop()
