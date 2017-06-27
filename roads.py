from scipy import ndimage, misc
import numpy as np

import neural
import learner


def roads(image):
    model_base = neural.get_base_network()
    model_specs = neural.get_specialized_network()

    img_sat = misc.imresize(image, size=(600, 600))
    result = np.zeros((600, 600))
    slices_sat = learner.slice_image_with_axis(img_sat, neural.WINDOW)

    for slice_sat, x, y in slices_sat:
        prediction = model_base.predict(slices_sat)
        print(prediction)
        if prediction[1] > prediction[0]:   # is a road, see neural constants
            mini_slices = learner.slice_image_with_axis(slice_sat, 2)
            for mini_slice, xx, yy in mini_slices:
                mini_prediction = model_specs.predict(learner.get_surroundings(mini_slice))
                if mini_prediction[1] > mini_prediction[0]:   # 2x2 is a road
                    result[x + xx: x + xx + 2, y + yy: y + yy + 2] = 255
                else:
                    pass
                    # do nothing, zeros already
        else:
            pass
            # do nothing, zeros already

    return misc.imresize(result, size=(1500, 1500))


if __name__ == "__main__":
    pass