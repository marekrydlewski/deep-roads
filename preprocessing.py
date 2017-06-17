import cv2
import numpy as np

def test_color(color):
    h, s, v = color
    if h >= 60 and h <= 160 and s >= 50 and v >= 25:
        return 255
    return 0

def neighborhood(mask, on_thr, off_thr):
    new_mask = np.zeros(mask.shape)
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            count = 0
            for x in range(-1,1):
                for y in range(-1,1):
                    if row + x >= 0 and col + y >= 0 and row + x < mask.shape[0] and col + y < mask.shape[1]:
                        if mask[row+x, col+y] == 255:
                            count += 1
            if count > on_thr:
                new_mask[row, col] = 255
            elif count < off_thr:
                new_mask[row, col] = 0
            else:
                new_mask[row, col] = mask[row, col]
    return new_mask



picture = cv2.imread("10078675_15.tiff")
hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)

size_x, size_y, size_z = hsv.shape

mask = np.zeros([size_x, size_y, 1])


for x in range(size_x):
    for y in range(size_y):
        mask[x,y] = test_color(hsv[x,y])


mask = neighborhood(mask, 2, 1)
mask = neighborhood(mask, 3, 2)
mask = neighborhood(mask, 4, 3)
cv2.imwrite("test.tiff", mask)

print("dupa")