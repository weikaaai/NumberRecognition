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
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# configs
iteration = 10
learning_rate = 0.01

# weights init
weights_0 = 2 * np.random.random((2500, 2500)) - 1
weights_1 = 2 * np.random.random((2500, 10)) - 1

# training process
for i in range(iteration):
    hidden_layer_1 = np.dot(test_img, weights_0) + 1
    hidden_layer_2 = sigmoid(hidden_layer_1)
    output_1 = np.dot(hidden_layer_2, weights_1) + 1
    output_2 = sigmoid(output_1)
    error = (test_ans - output_2) ** 2
    print("average error:", np.mean(np.sum(error, axis=0)))
    delta = 2 * (test_ans - output_2) * sigmoid(output_1, True)
    delta = np.dot(delta.T, hidden_layer_2).T
    weights_1 -= delta
