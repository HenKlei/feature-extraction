import matplotlib.pyplot as plt

from feature_extraction import HarrisCornerDetector
from feature_extraction.utils.windows import characteristic
from feature_extraction.utils.create_example_images import make_square
from feature_extraction.utils.grad import finite_difference


image = make_square()
plt.matshow(image)
plt.show()

gradient_image = finite_difference(image)
plt.matshow(gradient_image[0])
plt.show()
plt.matshow(gradient_image[1])
plt.show()

k = 1
window = characteristic(k)
detector = HarrisCornerDetector(window)
result = detector.detect(gradient_image)
plt.matshow(result)
plt.show()
