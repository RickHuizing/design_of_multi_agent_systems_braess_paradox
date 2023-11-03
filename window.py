import time
import tkinter as tk

CONFIG_FILE = "gui.config"


def configure_event_handler(event):
    print("save", event)
    with open(CONFIG_FILE, "w") as f:
        f.write(f"{event.width}x{event.height}+{event.x}+{event.y}\n")


def load_window_config(window: tk.Tk):
    try:
        with open(CONFIG_FILE, "r") as f:
            geom = f.readline().strip()
            window.geometry(geom)
            return geom
    except FileNotFoundError:
        pass


def create_window():
    window = tk.Tk()

    geom = load_window_config(window)

    # bind the configure event handler to the <Configure> event (change in size or location)
    window.bind("<Configure>", configure_event_handler)

    if geom is not None:
        # sometimes fails the first time
        window.geometry(geom)
    return window


def example_app(window):
    greeting = tk.Label(text="Hello, Tkinter")
    greeting.pack()

    button = tk.Button()
    button.configure(text="Click me!")
    button.pack()

    canvas = tk.Canvas()

    canvas.configure(border=2, relief="sunken", width=400, height=1000, bg="white")
    canvas.create_rectangle(10, 10, 20, 20, fill="black")
    canvas.pack()

    window.bind("<Configure>", configure_event_handler)
    window.mainloop()
