import cv2
import numpy as np
import glob

# debug
np.random.seed(1)

test_img = np.empty((0, 2500), float)
test_ans = np.empty((0, 10), float)

# import training images
for f in glob.glob('./images/*.png'):
    temp_rows, temp_cols = test_img.shape
    r = np.random.randint(temp_rows + 1)
    im = cv2.imread(f, 0)
    im = im.flatten()
    im = im / 255.0
    # insert to random row
    test_img = np.insert(test_img, r, np.array([im]), axis=0)
    temp_ans = np.array([0.0 for _ in range(10)])
    temp_ans[int(f[9])] = 1.0
    # insert to random row
    test_ans = np.insert(test_ans, r, np.array([temp_ans]), axis=0)

# activation function
def sigmoid(x, d=False):
    if d:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(1) ** (-x))

# configs
iteration = 100000

# weights init
weights_0 = 2 * np.random.random((2500, 2500)) - 1
weights_1 = 2 * np.random.random((2500, 10)) - 1

for i in range(iteration):
    pass
