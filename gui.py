import tkinter as tk
from globals import Gl
import window as W
from model import Model_4link, Model_5link

NUMBER_OF_AGENTS = 248
PARTICLE_SIZE = 4


def main():
    # initialise main window (remembers size and location)
    window = W.create_window()

    # initialise model
    model = Model_5link(NUMBER_OF_AGENTS)
    model.initialize_agents_positions()

    # initialise canvas for drawing
    canvas = tk.Canvas()
    canvas.configure(border=1, width=550, height=900, bg="white")
    canvas_scale = 1.5

    def draw():
        junctions = [Gl.J1, Gl.J2, Gl.J3, Gl.J4]

        # for each edge/junction in model.infrastructure, initialise a list of 0's with length of the edge
        edges = {key: [0] if key in junctions else [0] * item.length for key, item in model.infrastructure.items()}
        # sort the edges so that junctions are drawn last
        edges = {key: item for key, item in sorted(edges.items(), key=lambda item: item[0] in junctions)}

        # mark positions that are occupied by agents
        agents = model.schedule.agents
        for agent in agents:
            try:
                [(name, pos)] = agent.in_pos.items()
                edges[name][pos] = 1
            except Exception as e:
                print(e)
                print(agent.in_pos)
                raise e

        # lil bit hacky but it works^^
        # go over the complete network twice
        # - first pass: draw black squares for all empty positions
        # - second pass: draw red squares for all occupied positions
        # (this is what the switch variable is for)
        # makes sure black squares are not drawn over red squares
        for switch in [0, 1]:
            for key in edges.keys():
                for position, occupied in enumerate(edges[key]):
                    edge = model.infrastructure[key]
                    if key is Gl.E_SPAWN:
                        x1 = position % 128
                        y1 = -4 if position < 128 else -2
                    elif key in junctions:
                        x1, y1 = edge.pos
                    else:
                        x1_s, y1_s = edge.pos_start
                        x1_e, y1_e = edge.pos_end
                        x1 = x1_s + (position * (x1_e - x1_s) / edge.length)
                        y1 = y1_s + (position * (y1_e - y1_s) / edge.length)

                    # scale and padding/margins
                    x1 = x1 * PARTICLE_SIZE + 25
                    y1 = y1 * PARTICLE_SIZE + 25

                    x2 = x1 + PARTICLE_SIZE
                    y2 = y1 + PARTICLE_SIZE

                    if occupied == 1:
                        if switch == 1:
                            if key in junctions:
                                canvas.create_rectangle(x1, y1, x2 + 2, y2 + 2, fill="blue")
                            else:
                                canvas.create_rectangle(x1, y1, x2 + 1, y2 + 1, fill="red")
                    else:
                        if switch == 0:
                            canvas.create_rectangle(x1, y1, x2, y2, fill="black")
        canvas.scale("all", 0, 0, canvas_scale, canvas_scale)

    # frame for input objects
    input_frame = tk.Frame(window)
    input_frame.configure(border=2, width=550, height=100)
    input_frame.pack()

    step_button = tk.Button(input_frame, text="Step")
    step_button.grid(row=0, column=1, padx=10, pady=10)
    step_button.configure(command=lambda: step())

    def step():
        model.step()

        canvas.delete("all")
        draw()

    # while run run_model() else toggle stop
    run_button = tk.Button(input_frame, text="Run")
    run_button.grid(row=0, column=0, padx=10, pady=10)

    run_button.configure(command=lambda: start())
    run = True

    def stop():
        nonlocal run
        run_button.configure(text="Run", command=start)
        run = False

    def start():
        nonlocal run
        run = True
        run_button.configure(text="Stop", command=stop)
        window.after(15, run_model())

    def run_model():
        if run:
            step()

            window.after(15, run_model)

    # run for x steps
    run_steps_label = tk.Label(input_frame, text="Steps:")
    run_steps_label.grid(row=1, column=0, padx=10, pady=10)

    run_steps_entry = tk.Entry(input_frame)
    run_steps_entry.grid(row=1, column=1, padx=10, pady=10)
    run_steps_entry.insert(0, "1000")

    run_steps_button = tk.Button(input_frame, text="Run")
    run_steps_button.grid(row=1, column=2, padx=10, pady=10)
    run_steps_button.configure(command=lambda: run_steps(int(run_steps_entry.get())))

    def run_steps(steps):
        for _ in range(steps):
            model.step()
        canvas.delete("all")
        draw()

    draw()

    canvas.pack()

    # scale canvas
    window.update()
    canvas.configure(width=canvas.winfo_reqwidth() * canvas_scale, height=canvas.winfo_reqheight() * canvas_scale)
    # get preferred canvas width and height


    window.mainloop()


if __name__ == "__main__":
    main()
