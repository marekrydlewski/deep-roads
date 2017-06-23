# import neural
from scipy import ndimage, misc
import os


def load_img(name):
    img = ndimage.imread(name)
    return misc.imresize(img, size=(600, 600))


def slice_image(img):
    windowsize_r = 20
    windowsize_c = 20
    windows = []
    for r in range(0, img.shape[0] - windowsize_r, windowsize_r):
        for c in range(0, img.shape[1] - windowsize_c, windowsize_c):
            windows.append(img[r:r + windowsize_r, c:c + windowsize_c])
    return windows


def learn_directory():
    filenames_map = next(os.walk("test/map/"))[2]
    filenames_sat = next(os.walk("test/sat/"))[2]

    for map, sat in zip(filenames_map, filenames_sat):
        img_map = load_img("test/map/" + map)
        img_sat = load_img("test/sat/" + sat)
        # to do

if __name__ == "__main__":
    img_map = load_img("test/map/10378780_15.tif")
    img_sat = load_img("test/sat/10378780_15.tiff")
    slices_map = slice_image(img_map)
    slices_sat = slice_image(img_sat)

