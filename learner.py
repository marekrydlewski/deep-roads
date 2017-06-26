import neural
from scipy import ndimage, misc
import os
import numpy as np

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


def learn_directory():
    filenames_map = next(os.walk("test/map/"))[2]
    filenames_sat = next(os.walk("test/sat/"))[2]
    model = neural.get_base_network()
    for map, sat in zip(filenames_map, filenames_sat):
        img_map = load_img("test/map/" + map)
        img_sat = load_img("test/sat/" + sat)
        print("Test")

        x, y = get_data_from_images(img_map, img_sat)
        print("Test2")
        model.fit(x,y)
        print("Test3")


if __name__ == "__main__":
    img_map = load_img("test/map/10378780_15.tif")
    img_sat = load_img("test/sat/10378780_15.tiff")
    x, y = get_data_from_images(img_map, img_sat)
    learn_directory()
    print("Test")
