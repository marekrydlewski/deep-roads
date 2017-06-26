import neural
from scipy import ndimage, misc
import os
import numpy as np
from neural import *

__threshold = 120
__is_road = np.array([1, 0])
__is_not_road = np.array([0, 1])
__windowsize_r = 20
__windowsize_c = 20


def load_img(name):
    img = ndimage.imread(name)
    return misc.imresize(img, size=(600, 600))


def slice_image(img):
    windows = []
    for r in range(0, img.shape[0] - __windowsize_r, __windowsize_r):
        for c in range(0, img.shape[1] - __windowsize_c, __windowsize_c):
            windows.append(img[r:r + __windowsize_r, c:c + __windowsize_c])
    return windows


def get_data_from_images(img_map, img_sat):
    slices_map = slice_image(img_map)
    slices_sat = slice_image(img_sat)
    x_train = np.zeros((len(slices_map), 20, 20, 3))
    y_train = np.zeros((len(slices_sat), 2))

    for i, (slice_map, slice_sat) in enumerate(zip(slices_map, slices_sat)):
        if np.sum(slice_map >__threshold) >= 1:
            y_train[i] = __is_road
            x_train[i] = slice_sat
        else:
            y_train[i] = __is_not_road
            x_train[i] = slice_sat
    return (x_train, y_train)


def learn_base_directory():
    filenames_map = next(os.walk("test/map/"))[2]
    filenames_sat = next(os.walk("test/sat/"))[2]
    model = get_base_network()

    licznik = 0
    for _ in range(3):
        for map, sat in zip(filenames_map, filenames_sat):
            img_map = load_img("test/map/" + map)
            img_sat = load_img("test/sat/" + sat)
            # print(map)
            x, y = get_data_from_images(img_map, img_sat)
            model.fit(x, y, epochs=1)
            # print(map)
            licznik += 1
            if licznik == 20:
                licznik = 0
                neural.save_base_network(model)
    neural.save_base_network(model)


def learn_directory_specialized():
    pass

if __name__ == "__main__":
    #img_map = load_img("pre/10078660_15.tif")
    #img_sat = load_img("post/10078660_15.tiff")
    #x, y = get_data_from_images(img_map, img_sat)
    learn_base_directory()
    print("Test")
