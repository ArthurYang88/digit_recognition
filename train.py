import main
import ml_code
import numpy as np
import plotly.graph_objects as go

fig = go.Figure()

train_num = 10000
train_rounds = 5

with open("train_data.txt") as f:
    line = f.readline()
    if line.strip() == '1':
        main.get_data(f)

with open("success.txt", "r") as f:
    yes_lst = list(map(int, f.readline().split()))

yes = 0
if yes_lst:
    yes = yes_lst[-1]

with np.load("mnist.npz") as data:
    x_train = data["x_train"]
    y_train = data["y_train"]

    x_train = x_train.reshape(-1, 784)
    x_train = x_train.astype(float)/255

    for _ in range(train_rounds):
        for i in range(train_num):
            ml_code.layer1 = x_train[i]
            ml_code.update(y_train[i])
            if ml_code.best_guess == y_train[i]:
                yes += 1
            yes_lst.append(yes)

        x_axis = list(range(len(yes_lst)))

        fig.add_scatter(
            x=x_axis,
            y=[yes_lst[i]/x_axis[i] for i in range(1, len(yes_lst))],
            mode="lines+markers"
        )
        fig.show(renderer="browser")

with open("success.txt", "w") as f:
    for x in yes_lst:
        f.write(f"{x} ")

main.save()
