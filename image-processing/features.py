from scipy import misc
import numpy as np
import cv2

# read an image, and resize it into 256 height, 256 width, maintain deep size
img = misc.imresize(misc.imread('img.img'), (256, 256))
# flatten in 2D
imgflatten = img.reshape([-1, img.shape[2]])

# subdivide rgb vectorizing
# good to determine color differentation, like (day, night)
def process_image(image, blocks = 4):
	# return: size of 4 * 4 * 4
    feature = [0] * blocks * blocks * blocks
    pixel_count = 0
    for i in range(image.shape[0]):
        ridx = int(img[i, 0] / (256/blocks))
        gidx = int(img[i, 1] / (256/blocks))
        bidx = int(img[i, 2] / (256/blocks))
        idx = ridx + gidx * blocks + bidx * blocks * blocks
        feature[idx] += 1
        pixel_count += 1
    return [x/pixel_count for x in feature]

