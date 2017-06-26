import neural
from scipy import ndimage, misc
import os
import numpy as np
import random

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


def get_random_base_data_from_images(img_map, img_sat):
    x_roads = []
    x_no_roads = []
    for i in range(1000):
        x = random.randint(11, 589)
        y = random.randint(11, 589)
        sat_data = img_sat[x - 9: x + 10, y - 9: y + 10]
        map_data = img_map[x - 9: x + 10, y - 9: y + 10]
        if np.sum(map_data > __threshold) >= 1:
            x_roads.append(sat_data)
        else:
            x_no_roads.append(sat_data)
    min_len = len(x_no_roads) if (len(x_roads) >= len(x_no_roads)) else len(x_roads)

    x_roads_np = np.array(x_roads[0:min_len])
    x_no_roads_np = np.array(x_no_roads[0:min_len])
    x_train = np.concatenate((x_roads_np, x_no_roads_np))

    y_train = np.zeros((2 * min_len, 2))
    y_train[0: min_len] = __is_not_road
    y_train[min_len: 2*min_len] = __is_road

    return (x_train, y_train)


def get_specialized_data_from_images(img_map, img_sat):
    x_roads = []
    x_no_roads = []
    for i in range(1000):
        x = random.randint(11, 589)
        y = random.randint(11, 589)
        sat_data = img_sat[x - 9: x + 10, y - 9: y + 10]
        if check_road(x, y, img_map):
            x_roads.append(sat_data)
        else:
            x_no_roads.append(sat_data)
    min_len = len(x_no_roads) if (len(x_roads) >= len(x_no_roads)) else len(x_roads)

    x_roads_np = np.array(x_roads[0:min_len])
    x_no_roads_np = np.array(x_no_roads[0:min_len])
    x_train = np.concatenate((x_roads_np, x_no_roads_np))

    y_train = np.zeros((2 * min_len, 2))
    y_train[0: min_len] = __is_not_road
    y_train[min_len: 2*min_len] = __is_road

    return (x_train, y_train)


def check_road(x, y, img_mat):
    if img_mat[x][y] >= __threshold or \
                    img_mat[x][y + 1] >= __threshold or \
                    img_mat[x + 1][y] >= __threshold or \
                    img_mat[x + 1][y + 1] >= __threshold:
        return True
    else:
        return False


def learn_directory_base():
    filenames_map = next(os.walk("train/map/"))[2]
    filenames_sat = next(os.walk("train/sat/"))[2]
    model = neural.get_base_network()

    licznik = 0
    for _ in range(3):
        for map, sat in zip(filenames_map, filenames_sat):
            img_map = load_img("train/map/" + map)
            img_sat = load_img("train/sat/" + sat)
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
    filenames_map = next(os.walk("train/map/"))[2]
    filenames_sat = next(os.walk("train/sat/"))[2]
    model = neural.get_specialized_network()
    licznik = 0
    for _ in range(3):
        for map, sat in zip(filenames_map, filenames_sat):
            img_map = load_img("train/map/" + map)
            img_sat = load_img("train/sat/" + sat)
            # print(map)
            x, y = get_specialized_data_from_images(img_map, img_sat)
            model.fit(x, y, epochs=1)
            # print(map)
            licznik += 1
            if licznik == 20:
                licznik = 0
                neural.save_specialized_network(model)
    neural.save_specialized_network(model)


if __name__ == "__main__":
    img_map = load_img("train/map/10078660_15.tif")
    img_sat = load_img("train/sat/10078660_15.tiff")
    #x, y = get_data_from_images(img_map, img_sat)
    x, y = get_random_base_data_from_images(img_map, img_sat)
    learn_directory_base()
    print("Test")
