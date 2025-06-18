import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rembg import remove

# Step 1: Load image and detect face
img = cv2.imread("input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    raise Exception("No face detected!")

(x, y, w, h) = faces[0]
face_img = img[y:y+h+40, x:x+w]  # crop a bit extra for neck

# Step 2: Remove background
face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
face_no_bg = remove(face_pil)

# Step 3: Convert to binary face style
face_np = np.array(face_no_bg)
gray = cv2.cvtColor(face_np, cv2.COLOR_RGBA2GRAY)

scale = 0.15
small = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
h_orig, w_orig = gray.shape

binary_image = Image.new("RGB", (w_orig, h_orig), "black")
draw = ImageDraw.Draw(binary_image)

# Load any available monospace font
try:
    font = ImageFont.truetype("C:/Windows/Fonts/cour.ttf", size=10)
except:
    font = ImageFont.load_default()

for y in range(small.shape[0]):
    for x in range(small.shape[1]):
        brightness = small[y, x]
        char = '1' if brightness > 127 else '0'
        real_x = int(x / scale)
        real_y = int(y / scale)
        draw.text((real_x, real_y), char, font=font, fill=(0, 255, 0))

binary_image.save("hacker_face.png")
binary_image.show()
