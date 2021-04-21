import numpy as np


class HarrisCornerDetector:
    def __init__(self, window, window_max_extend=5, alpha=0.05, threshold=0.01):
        self.window = window

        # used to reduced computational effort since this enables computation of
        # second moment matrix to use only a certain amount of summands
        self.window_max_extend = window_max_extend

        assert 0.04 <= alpha <= 0.06
        self.alpha = alpha
        assert threshold >= 0.
        self.threshold = threshold

    def detect(self, gradient_image):
        # general version: gradient_image.shape = (dim, N_x_1, N_x_2, ..., N_x_dim)

        # current implementation only for 2d-images:
        assert gradient_image.shape[0] == 2
        assert gradient_image.ndim == 3

        dim = gradient_image.shape[0]
        assert dim == gradient_image.ndim - 1
        image_shape = gradient_image.shape[1:]
        response_function_matrix = np.zeros(image_shape)

        for point in np.ndindex(image_shape):
            M = np.zeros((dim, ) * dim)
            for x in np.ndindex((self.window_max_extend, ) * dim):
                position = tuple(np.minimum(np.maximum(np.add(point, x), np.zeros(dim)),
                                            np.subtract(image_shape, np.ones(dim))).astype(int))
                M += (self.window(x)
                      * np.array([[gradient_image[0][position]**2,
                                   gradient_image[0][position] * gradient_image[1][position]],
                                  [gradient_image[0][position] * gradient_image[1][position],
                                   gradient_image[1][position]**2]]))

                position = tuple(np.minimum(np.maximum(np.subtract(point, x), np.zeros(dim)),
                                            np.subtract(image_shape, np.ones(dim))).astype(int))
                M += (self.window(x)
                      * np.array([[gradient_image[0][position]**2,
                                   gradient_image[0][position] * gradient_image[1][position]],
                                  [gradient_image[0][position] * gradient_image[1][position],
                                   gradient_image[1][position]**2]]))
            r = np.linalg.det(M) - self.alpha * np.trace(M)**2
            response_function_matrix[point] = r

        return response_function_matrix > self.threshold


class CannyEdgeDetector:
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def detect(self, gradient_image):
        return np.linalg.norm(gradient_image, axis=0) > self.threshold
