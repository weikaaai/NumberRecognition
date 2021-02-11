import cv2
import numpy as np
import glob

# debug
np.random.seed(1)

train_img = np.empty((0, 2500), np.float128)
train_ans = np.empty((0, 10), np.float128)

# import training images
for f in glob.glob('./images/*.png'):
    temp_rows, temp_cols = train_img.shape
    r = np.random.randint(temp_rows + 1)
    im = cv2.imread(f, 0)
    im = im.flatten()
    im = im / 255.0
    # insert to random row
    train_img = np.insert(train_img, r, np.array([im]), axis=0)
    temp_ans = np.array([0.0 for _ in range(10)])
    temp_ans[int(f[9])] = 1.0
    # insert to random row
    train_ans = np.insert(train_ans, r, np.array([temp_ans]), axis=0)

# activation function


def relu(x, d=False):
    if d:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    return np.maximum(0, x)


def sigmoid(x, d=False):
    if d:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


# configs
iteration = 3
learning_rate = 0.01

# weights init
weights_0 = 2 * np.random.random((2500, 2500)) - 1
weights_1 = 2 * np.random.random((2500, 10)) - 1

# bias init
b_0 = 1
b_1 = 1

# training process
for i in range(iteration):
    # forward
    p1 = np.dot(train_img, weights_0) + b_0
    act1 = relu(p1)
    p2 = np.dot(act1, weights_1) + b_1
    act2 = sigmoid(p2)
    err = (train_ans - act2) ** 2
    m_err = np.mean(np.sum(err, axis=0))
    print('mean error:', m_err)
    # backward
    tmp1 = act1.T
    tmp2 = 2 * (train_ans - act2) * sigmoid(p2, True)
    tmp3 = np.dot(tmp1, tmp2)
    tmp11 = train_img.T
    tmp12 = 2 * (train_ans - act2) * sigmoid(p2, True)
    tmp13 = np.dot(tmp12, weights_1.T)
    tmp14 = tmp13 * relu(p1, True)
    tmp15 = np.dot(tmp11, tmp14)
    weights_0 = weights_0 + (learning_rate * tmp15)
    weights_1 = weights_1 + (learning_rate * tmp3)
