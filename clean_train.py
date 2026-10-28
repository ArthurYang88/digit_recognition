import main
import ml_code
import numpy as np

train_num = 10000
train_rounds = 5

with open("train_data.txt") as f:
    line = f.readline()
    if line.strip() == '1':
        main.get_data(f)

with np.load("mnist.npz") as data:
    x_train = data["x_train"]
    y_train = data["y_train"]

    x_train = x_train.reshape(-1, 784)
    x_train = x_train.astype(float)/255

    for _ in range(train_rounds):
        for i in range(train_num):
            ml_code.layer1 = x_train[i]
            ml_code.update(y_train[i])

main.save()
