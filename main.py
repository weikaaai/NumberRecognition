import cv2
import numpy as np
import glob

# debug
np.random.seed(1)

train_img = np.empty((0, 2500), np.float32)
train_ans = np.empty((0, 10), np.float32)

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


# def relu(x, d=False):
#     if d:
#         x[x <= 0] = 0
#         x[x > 0] = 1
#         return x
#     return np.maximum(0, x)


def sigmoid(x, d=False):
    if d:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


# configs
np.set_printoptions(threshold=np.inf)
iteration = 30000
learning_rate = 0.001
hidden = 100

# weights init
weights_0 = np.random.uniform(-1, 1, (2500, hidden))
weights_1 = np.random.uniform(-1, 1, (hidden, 10))

# bias init
b_0 = 1
b_1 = 1

# training network
for i in range(iteration):
    # forward
    p1 = np.dot(train_img, weights_0) + b_0
    act1 = sigmoid(p1)
    p2 = np.dot(act1, weights_1) + b_1
    act2 = sigmoid(p2)
    err = (train_ans - act2) ** 2
    m_err = np.mean(np.sum(err, axis=0))
    print('mean error:', m_err)
    # backward
    t1 = 2 * (train_ans - act2) * sigmoid(p2, True)
    t2 = np.dot(act1.T, t1)
    t_1 = np.dot(weights_1, t1.T)
    t_2 = sigmoid(p1, True) * t_1.T
    t_3 = np.dot(train_img.T, t_2)
    weights_0 = weights_0 + (learning_rate * t_3)
    weights_1 = weights_1 + (learning_rate * t2)

# testing network
test_img = cv2.imread('./images/5-1.png', 0)
test_img = test_img.flatten()
test_img = test_img / 255.0
p1 = np.dot(test_img, weights_0)
z1 = sigmoid(p1)
p2 = np.dot(z1, weights_1)
z2 = sigmoid(p2)
print(z2)

# print('saving results...')
# results = f'''
# --------------------------------------------------
# iteration: {iteration}
# final mean error: {m_err}
# --------------------------------------------------
# final weights_0:
# {weights_0}
# --------------------------------------------------
# final weights_1:
# {weights_1}
# --------------------------------------------------
# '''
# f = open('output.txt', 'w')
# f.write(results)
# f.close()
# print('results saved to output.txt')
