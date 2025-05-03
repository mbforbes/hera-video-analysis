import code

import cv2
import easyocr
from imgcat import imgcat

reader = easyocr.Reader(["en"])
# print(reader.readtext("eocrtest.png"))
# res2 = reader.readtext("eocrtest2.png")

# res3 = reader.readtext("eocrtest3.png") # having trouble with this one

image = cv2.imread("eocrtest3.png")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# gray = cv2.GaussianBlur(gray, (3, 3), 0)
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh = gray
# thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
imgcat(thresh)
res3 = reader.readtext(thresh)
print(res3)
code.interact(local=dict(globals(), **locals()))
