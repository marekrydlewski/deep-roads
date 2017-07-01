from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt

import neural
import learner

#
# def roads(image):
#     model_base = neural.get_base_network()
#     model_specs = neural.get_specialized_small_network()
#
#     img_sat = misc.imresize(image, size=(600, 600))
#     result = np.zeros((600, 600), dtype=np.uint8)
#     slices_sat = learner.slice_image_with_axis(img_sat, neural.WINDOW)
#     npad = ((11, 11), (11, 11), (0, 0))
#     img_sat_pad = np.pad(img_sat, pad_width=npad, mode="symmetric")
#
#     for slice_sat, x, y in slices_sat:
#         slice_list = []
#         slice_list.append(slice_sat)
#         prediction = model_base.predict(np.array(slice_list))
#         prediction = prediction[0]
#         # print(prediction)
#         if prediction[1] > prediction[0]:   # is a road, see neural constants
#             mini_slices = learner.slice_image_with_axis(slice_sat, 1)
#             for _, xx, yy in mini_slices:
#                 mini_list = []
#                 mini_list.append(learner.get_surroundings_with_small_pad(img_sat_pad, x + xx, y + yy))
#                 mini_prediction = model_specs.predict(np.array(mini_list))
#                 mini_prediction = mini_prediction[0]
#                 if mini_prediction[0] > mini_prediction[1]:   # 1x1 is a road
#                     result[x + xx, y + yy] = 255
#                 else:
#                     pass
#                     # do nothing, zeros already
#         else:
#             pass
#             # do nothing, zeros already
#     # return result
#     return misc.imresize(result, size=(1500, 1500))


def roads(image):
    model_base = neural.get_base_network()
    model_specs = neural.get_specialized_small_network()

    img_sat = misc.imresize(image, size=(600, 600))
    result = np.zeros((600, 600), dtype=np.uint8)
    slice_sat_data, x_data, y_data = learner.slice_image_with_axis_sep(img_sat, neural.WINDOW)

    npad = ((11, 11), (11, 11), (0, 0))
    img_sat_pad = np.pad(img_sat, pad_width=npad, mode="symmetric")

    predictions = model_base.predict(slice_sat_data)
    predictions_mask = (predictions[:, 0] > predictions[:, 1])   # is a road, see neural constants

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

    return misc.imresize(result, size=(1500, 1500))


if __name__ == "__main__":
    img_map = learner.load_img("train/map/10078660_15.tif")
    img_sat = learner.load_img("train/sat/10078660_15.tiff")
    output = roads(img_sat)
    #output = np.full((600, 600), 1, dtype=np.uint8)

    plt.figure(1)
    plt.imshow(img_map)
    plt.figure(2)
    plt.imshow(img_sat)
    plt.figure(3)
    plt.imshow(output)
    plt.show()
