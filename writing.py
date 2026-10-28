import tkinter as tk
import numpy as np
import ml_code
import main

with open("train_data.txt") as f:
    line = f.readline()
    if line.strip() == '1':
        main.get_data(f)


WIDTH = 28
HEIGHT = 28
PIXEL_SIZE = 20
INVALID_POS = (-1, -1)

pixels = [WIDTH * [0] for _ in range(HEIGHT)]

root = tk.Tk()
root.title("Writing Canvas")

canvas = tk.Canvas(
    root,
    width=WIDTH*PIXEL_SIZE,
    height=HEIGHT*PIXEL_SIZE,
    bg="black"
)
canvas.pack()


def cursor_canvas_pos(event):
    x = int(event.x / PIXEL_SIZE)
    y = int(event.y / PIXEL_SIZE)
    if x < 0 or x >= WIDTH:
        return INVALID_POS
    if y < 0 or y >= HEIGHT:
        return INVALID_POS
    return x, y

def paint_cell(pos):
    x = pos[0]
    y = pos[1]
    pixels[y][x] = 1

def render(clear=False):
    if clear:
        canvas.delete("ink")

    for x in range(WIDTH):
        for y in range(HEIGHT):

            if clear or pixels[y][x] == 0:
                pixels[y][x] = 0
                continue

            canvas.create_rectangle(
                x * PIXEL_SIZE,
                y * PIXEL_SIZE, (x + 1) * PIXEL_SIZE,
                (y + 1) * PIXEL_SIZE,
                fill="white",
                tags="ink"
            )

def clear():
    render(clear=True)

def on_press(event):
    pos = cursor_canvas_pos(event)
    paint_cell(pos)
    render()



canvas.bind("<Button-1>", on_press)
canvas.bind("<B1-Motion>", on_press)

buttons = tk.Frame(root)
buttons.pack(pady=8)

tk.Button(buttons, text="Clear", command=clear).pack(side="left", padx=8)

def read_number():
    ml_code.layer1 = np.asarray(pixels, dtype=float).flatten()
    ml_code.forward()
    ml_code.print_output()

tk.Button(buttons, text="Read", command=read_number).pack(side="right", padx=8)

root.bind("r", lambda event: read_number())

root.mainloop()