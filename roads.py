from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from skimage import morphology
import neural
import learner


def roads(image):
    model_base = neural.get_base_network()
    model_specs = neural.get_specialized_small_network()

    img_sat = misc.imresize(image, size=(600, 600))
    result = np.zeros((600, 600), dtype=np.uint8)
    slice_sat_data, x_data, y_data = learner.slice_image_with_axis_sep(img_sat, neural.WINDOW)

    npad = ((11, 11), (11, 11), (0, 0))
    img_sat_pad = np.pad(img_sat, pad_width=npad, mode="symmetric")

    predictions = model_base.predict(slice_sat_data)
    predictions_mask = (predictions[:, 0] > predictions[:, 1])  # is a road, see neural constants

    predictions = predictions[predictions_mask]
    x_data = x_data[predictions_mask]
    y_data = y_data[predictions_mask]

    data_list = []
    coords_list = []
    for prediction, x, y in zip(predictions, x_data, y_data):
        for xx in range(20):
            for yy in range(20):
                data_list.append(learner.get_surroundings_with_small_pad(img_sat_pad, x + xx, y + yy))
                coords_list.append((x + xx, y + yy))

    predictions_one_px = model_specs.predict(np.array(data_list))
    predictions_mask_one_px = (predictions_one_px[:, 0] > predictions_one_px[:, 1])

    coords_list = np.array(coords_list)
    coords_list = coords_list[predictions_mask_one_px]

    for x_px, y_px in coords_list:
        result[x_px, y_px] = 255

    # result = morphology.binary_erosion(result, morphology.diamond(1)).astype(np.uint8)
    # result = morphology.binary_erosion(result, morphology.diamond(1)).astype(np.uint8)
    # result = morphology.binary_opening(result, selem=np.ones((3, 3)))
    # result = morphology.remove_small_objects(result, min_size=64, connectivity=2)
    # result = morphology.binary_opening(result, selem=morphology.diamond(2))
    result = morphology.binary_erosion(result, morphology.diamond(1)).astype(np.uint8)
    result = morphology.binary_opening(result, selem=morphology.square(2))
    # result = morphology.binary_erosion(result, morphology.diamond(1)).astype(np.uint8)

    # result = morphology.dilation(result, morphology.diamond(1)).astype(np.uint8)
    # result = morphology.remove_small_objects(result, min_size=20, connectivity=1)

    return misc.imresize(result, size=(1500, 1500))


if __name__ == "__main__":
    # train
    img_map = learner.load_img("train/map/10078660_15.tif")
    img_sat = learner.load_img("train/sat/10078660_15.tiff")

    # test
    img_map = learner.load_img("test/map/15928855_15.tif")
    img_sat = learner.load_img("test/sat/15928855_15.tiff")

    output = roads(img_sat)

    plt.figure(1)
    plt.imshow(img_map)
    plt.figure(2)
    plt.imshow(img_sat)
    plt.figure(3)
    plt.imshow(output)
    plt.show()

    # model_base = neural.get_base_network()
    # model_specs = neural.get_specialized_small_network()
    #
    # plot_model(model_base, to_file='model_base.png', show_shapes=True)
    # plot_model(model_specs, to_file='model_specs.png', show_shapes=True)
    #
    #
    # print(model_base.summary())
    # print(model_specs.summary())
