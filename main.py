import ml_code

FIRST_LAYER_COUNT = 784
SECOND_LAYER_COUNT = 128
THIRD_LAYER_COUNT = 10

def get_data(f):
    for i in range(SECOND_LAYER_COUNT):
        line = list(map(float, f.readline().split()))
        for j in range(FIRST_LAYER_COUNT):
            ml_code.w1[i][j] = line[j]

    for i in range(THIRD_LAYER_COUNT):
        line = list(map(float, f.readline().split()))
        for j in range(SECOND_LAYER_COUNT):
            ml_code.w2[i][j] = line[j]

    line = list(map(float, f.readline().split()))
    for i in range(SECOND_LAYER_COUNT):
        ml_code.b1[i] = line[i]

    line = list(map(float, f.readline().split()))
    for i in range(THIRD_LAYER_COUNT):
        ml_code.b2[i] = line[i]

def save():
    with open("train_data.txt", "w") as f:
        f.write("1\n")

        for i in range(SECOND_LAYER_COUNT):
            for j in range(FIRST_LAYER_COUNT):
                f.write(f"{ml_code.w1[i][j]} ")
            f.write("\n")

        for i in range(THIRD_LAYER_COUNT):
            for j in range(SECOND_LAYER_COUNT):
                f.write(f"{ml_code.w2[i][j]} ")
            f.write("\n")

        for i in range(SECOND_LAYER_COUNT):
            f.write(f"{ml_code.b1[i]} ")
        f.write("\n")

        for i in range(THIRD_LAYER_COUNT):
            f.write(f"{ml_code.b2[i]} ")
        f.write("\n")

