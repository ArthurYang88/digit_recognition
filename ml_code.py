import numpy as np

# 3 layers
# 1st layer: 784 neurons
# 2nd layer: 128 neurons
# 3rd layer: 10 neurons (output layer)

# constants
FIRST_LAYER_COUNT = 784
SECOND_LAYER_COUNT = 128
THIRD_LAYER_COUNT = 10
learning_rate = 0.01

# layers
layer1 = np.zeros(FIRST_LAYER_COUNT)
layer2 = np.zeros(SECOND_LAYER_COUNT)
layer3 = np.zeros(THIRD_LAYER_COUNT)

# first and second weights
w1 = np.random.randn(SECOND_LAYER_COUNT, FIRST_LAYER_COUNT) * np.sqrt(1 / FIRST_LAYER_COUNT)
w2 = np.random.randn(THIRD_LAYER_COUNT, SECOND_LAYER_COUNT) * np.sqrt(1 / SECOND_LAYER_COUNT)
# first and second bias
b1 = np.zeros(SECOND_LAYER_COUNT)
b2 = np.zeros(THIRD_LAYER_COUNT)

correct = np.zeros(THIRD_LAYER_COUNT)

delta1 = np.zeros(SECOND_LAYER_COUNT)
delta2 = np.zeros(THIRD_LAYER_COUNT)
gradient_w1 = np.zeros((SECOND_LAYER_COUNT, FIRST_LAYER_COUNT))
gradient_w2 = np.zeros((THIRD_LAYER_COUNT, SECOND_LAYER_COUNT))

def act(x):
    # Sigmoid
    return 1 / (1 + np.exp(-x))

def act_diff(x):
    s = act(x)
    return s * (1 - s)

def cost():
    s = 0
    for i in range(THIRD_LAYER_COUNT):
        # cost function : 1/2 (a - z) ^ 2
        s += 0.5 * (layer3[i] - correct[i]) ** 2
    return s

def forward():
    global layer2, layer3
    layer2 = act(w1 @ layer1 + b1)
    layer3 = act(w2 @ layer2 + b2)

def update_gradients():
    global gradient_w1, gradient_w2, delta1, delta2

    # update delta 2
    for i in range(THIRD_LAYER_COUNT):
        u_y = np.dot(w2[i, :], layer2) + b2[i]
        delta2[i] = (layer3[i] - correct[i]) * act_diff(u_y)

    # update delta 1
    for i in range(SECOND_LAYER_COUNT):
        u_x = np.dot(w1[i, :], layer1) + b1[i]
        s = 0
        for k in range(THIRD_LAYER_COUNT):
            s += delta2[k] * w2[k][i]
        delta1[i] = act_diff(u_x) * s

    # update gradient_w2
    gradient_w2 = np.outer(delta2, layer2)

    # update gradient_w1
    gradient_w1 = np.outer(delta1, layer1)

    # bias gradient = delta

def update_layers():
    global w1, w2, b1, b2, layer2, layer3

    w1 = w1 - learning_rate * gradient_w1
    w2 = w2 - learning_rate * gradient_w2
    b1 = b1 - learning_rate * delta1
    b2 = b2 - learning_rate * delta2

    layer2 = act(w1 @ layer1 + b1)
    layer3 = act(w2 @ layer2 + b2)

def update(correct_ans):
    global correct

    for i in range(THIRD_LAYER_COUNT):
        correct[i] = 0
        if i == correct_ans:
            correct[i] = 1

    forward()
    update_gradients()
    update_layers()

    print_output() # if you want to see success rate then you will have to enable print_output

best_guess = 0

def print_output():
    global best_guess

    print("########################################################")
    print("########################################################")
    print("The results are: ")
    s = 0
    mx = 0
    mx_v = 0
    for i in range(THIRD_LAYER_COUNT):
        x = layer3[i]
        s += x
        if x > mx_v:
            mx_v = x
            mx = i

    for i in range(THIRD_LAYER_COUNT):
        print(f"{i}: {layer3[i]/s * 100}%")
    print()
    print(f"Most likely: {mx}")
    best_guess = mx
    print()
    print("########################################################")
    print("########################################################")

