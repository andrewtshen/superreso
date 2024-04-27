import numpy as np
from PIL import Image, ImageFilter
from ISR.models import RDN
import cv2

sample_image = cv2.blur(cv2.imread('sample_monkey.jpg'), (15,15)) # For now, we will use the blurred raspberry image as our sample
cv2.imwrite('blurred_monkey.jpg', sample_image)

sample_img = Image.open('blurred_monkey.jpg')
lr_img = np.array(sample_img)

rdn = RDN(weights='psnr-small')
sr_img = rdn.predict(lr_img)
sr_img = Image.fromarray(sr_img)
w, h = sr_img.size
sr_img = sr_img.resize((w//2, h//2))
sr_img.save('clear_monkey.jpg')

