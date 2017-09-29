import neural
from scipy import ndimage, misc
import os
import numpy as np
import random
from sklearn.utils import shuffle


def load_img(name):
    img = ndimage.imread(name)
    return misc.imresize(img, size=(600, 600))


def slice_image(img, window):
    windows = []
    for r in range(0, img.shape[0], window):
        for c in range(0, img.shape[1], window):
            windows.append(img[r:r + window, c:c + window])
    return windows


def slice_image_with_axis(img, window):
    windows = []
    for r in range(0, img.shape[0], window):
        for c in range(0, img.shape[1], window):
            windows.append((img[r:r + window, c:c + window], r, c))
    return windows


def slice_image_with_axis_sep(img, window):
    windows = []
    rl = []
    ll = []
    for r in range(0, img.shape[0], window):
        for c in range(0, img.shape[1], window):
            windows.append(img[r:r + window, c:c + window])
            rl.append(r)
            ll.append(c)

    windows = np.array(windows)
    rl = np.array(rl)
    ll = np.array(ll)
    return windows, rl, ll


def get_surroundings(img, x, y):
    return img[x - 9: x + 11, y - 9: y + 11]


def get_surroundings_with_pad(img, x, y):
    return img[x - 9 + 11: x + 11 + 11, y - 9 + 11: y + 11 + 11]


def get_surroundings_with_small_pad(img, x, y):
    return img[x - 6 + 11: x + 8 + 11, y - 6 + 11: y + 8 + 11]


def get_data_from_images(img_map, img_sat):
    slices_map = slice_image(img_map, neural.WINDOW)
    slices_sat = slice_image(img_sat, neural.WINDOW)
    x_train = np.zeros((len(slices_map), 20, 20, 3))
    y_train = np.zeros((len(slices_sat), 2))

    for i, (slice_map, slice_sat) in enumerate(zip(slices_map, slices_sat)):
        if np.sum(slice_map >neural.THRESHOLD) >= 1:
            y_train[i] = neural.IS_ROAD
            x_train[i] = slice_sat
        else:
            y_train[i] = neural.IS_NOT_ROAD
            x_train[i] = slice_sat

    return shuffle(x_train, y_train)


def get_random_base_data_from_images(img_map, img_sat):
    x_roads = []
    x_no_roads = []
    for i in range(1000):
        x = random.randint(10, 589)
        y = random.randint(10, 589)
        sat_data = get_surroundings(img_sat, x, y)
        map_data = get_surroundings(img_map, x, y)
        if np.sum(map_data > neural.THRESHOLD) >= 1:
            x_roads.append(sat_data)
        else:
            x_no_roads.append(sat_data)
    min_len = len(x_no_roads) if (len(x_roads) >= len(x_no_roads)) else len(x_roads)

    x_roads_np = np.array(x_roads[0:min_len])
    x_no_roads_np = np.array(x_no_roads[0:min_len])
    x_train = np.concatenate((x_roads_np, x_no_roads_np))

    y_train = np.zeros((2 * min_len, 2))
    y_train[0: min_len] = neural.IS_ROAD
    y_train[min_len: 2*min_len] = neural.IS_NOT_ROAD

    return shuffle(x_train, y_train)


def get_specialized_data_from_images(img_map, img_sat):
    x_roads = []
    x_no_roads = []

    npad = ((11, 11), (11, 11), (0, 0))
    img_sat_pad = np.pad(img_sat, pad_width=npad, mode="symmetric")
    for i in range(500):
        x = random.randint(10, 589)
        y = random.randint(10, 589)
        map_data = get_surroundings(img_map, x, y)
        sat_data = get_surroundings(img_sat, x, y)


        ###############
        # if check_road(x, y, img_map):
        #     x_roads.append(sat_data)
        # else:
        #     x_no_roads.append(sat_data)

        if np.sum(map_data > neural.THRESHOLD) >= 1:
            for xx in range(x - 9, x + 11, 2):
                for yy in range(y - 9, y + 11, 2):
                    # lustrooooooooooooooooo
                    if check_road(xx, yy, img_map):
                        x_roads.append(get_surroundings_with_pad(img_sat_pad, xx, yy))
                    else:
                        x_no_roads.append(get_surroundings_with_pad(img_sat_pad, xx, yy))
    min_len = len(x_no_roads) if (len(x_roads) >= len(x_no_roads)) else len(x_roads)

    if min_len == 0:
        print("ojoj")
        if len(x_no_roads) >= 1:
            x_train = np.array([x_no_roads[0]])
            y_train = np.array([neural.IS_NOT_ROAD])
            return x_train, y_train
        else:
            x_train = np.array([x_roads[0]])
            y_train = np.array([neural.IS_ROAD])
            return x_train, y_train

    x_roads_np = np.array(x_roads[0:min_len])
    x_no_roads_np = np.array(x_no_roads[0:min_len])
    x_train = np.concatenate((x_roads_np, x_no_roads_np))

    y_train = np.zeros((2 * min_len, 2))
    y_train[0: min_len] = neural.IS_ROAD
    y_train[min_len: 2*min_len] = neural.IS_NOT_ROAD

    return shuffle(x_train, y_train)


def get_specialized_small_data_from_images(img_map, img_sat):
    x_roads = []
    x_no_roads = []

    npad = ((11, 11), (11, 11), (0, 0))
    img_sat_pad = np.pad(img_sat, pad_width=npad, mode="symmetric")
    for i in range(50):
        x = random.randint(10, 589)
        y = random.randint(10, 589)
        map_data = get_surroundings(img_map, x, y)
        sat_data = get_surroundings(img_sat, x, y)

        if np.sum(map_data > neural.THRESHOLD) >= 1:
            for xx in range(x - 9, x + 11):
                for yy in range(y - 9, y + 11):
                    if check_road_one_px(xx, yy, img_map):
                        x_roads.append(get_surroundings_with_small_pad(img_sat_pad, xx, yy))
                    else:
                        x_no_roads.append(get_surroundings_with_small_pad(img_sat_pad, xx, yy))
    min_len = len(x_no_roads) if (len(x_roads) >= len(x_no_roads)) else len(x_roads)

    x_roads_np = np.array(x_roads[0:min_len])
    x_no_roads_np = np.array(x_no_roads[0:min_len])
    x_train = np.concatenate((x_roads_np, x_no_roads_np))

    y_train = np.zeros((2 * min_len, 2))
    y_train[0: min_len] = neural.IS_ROAD
    y_train[min_len: 2*min_len] = neural.IS_NOT_ROAD

    return shuffle(x_train, y_train)


def check_road(x, y, img_mat):
    if img_mat[x][y] >= neural.THRESHOLD or \
                    img_mat[x][y + 1] >= neural.THRESHOLD or \
                    img_mat[x + 1][y] >= neural.THRESHOLD or \
                    img_mat[x + 1][y + 1] >= neural.THRESHOLD:
        return True
    else:
        return False


def check_road_one_px(x, y, img_mat):
    if img_mat[x][y] >= neural.THRESHOLD:
        return True
    else:
        return False


def learn_directory_base():
    filenames_map = np.array(next(os.walk("train/map/"))[2])
    filenames_sat = np.array(next(os.walk("train/sat/"))[2])
    filenames_map, filenames_sat = shuffle(filenames_map, filenames_sat)
    model = neural.get_base_network()

    counter = 0
    l = 0
    for _ in range(45):
        for map, sat in zip(filenames_map, filenames_sat):
            img_map = load_img("train/map/" + map)
            img_sat = load_img("train/sat/" + sat)
            # print(map)
            print(str(l) + "/" + str(len(filenames_sat)) + ": " + str(_))
            x, y = get_data_from_images(img_map, img_sat)
            model.fit(x, y, epochs=1)
            # print(map)
            l += 1
            counter += 1
            if counter == 40:
                counter = 0
                neural.save_base_network(model)
        l = 0
    neural.save_base_network(model)


def learn_directory_specialized():
    filenames_map = np.array(next(os.walk("train/map/"))[2])
    filenames_sat = np.array(next(os.walk("train/sat/"))[2])
    filenames_map, filenames_sat = shuffle(filenames_map, filenames_sat)
    model = neural.get_specialized_small_network()
    l = 0
    counter = 0
    for _ in range(256):
        for map, sat in zip(filenames_map, filenames_sat):
            img_map = load_img("train/map/" + map)
            img_sat = load_img("train/sat/" + sat)
            print(str(l) + "/" + str(len(filenames_sat)) + ": " + str(_))
            x, y = get_specialized_small_data_from_images(img_map, img_sat)
            if len(x) != 0:
                model.fit(x, y, epochs=1)
            else:
                print("Empty data for fit")
            l += 1
            counter += 1
            if counter == 40:
                counter = 0
                neural.save_specialized_small_network(model)
        l = 0
    neural.save_specialized_small_network(model)


def test_valid_directory():
    filenames_map = next(os.walk("test/map/"))[2]
    filenames_sat = next(os.walk("test/sat/"))[2]
    model = neural.get_base_network()

    for map, sat in zip(filenames_map, filenames_sat):
        img_map = load_img("test/map/" + map)
        img_sat = load_img("test/sat/" + sat)
        x, y = get_random_base_data_from_images(img_map, img_sat)
        score = model.evaluate(x, y)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


if __name__ == "__main__":
    # img_map = load_img("train/map/10078660_15.tif")
    # img_sat = load_img("train/sat/10078660_15.tiff")
    # x, y = get_data_from_images(img_map, img_sat)
    # x, y = get_random_base_data_from_images(img_map, img_sat)
    learn_directory_base()
    # learn_directory_specialized()
    test_valid_directory()
    print("Test")
