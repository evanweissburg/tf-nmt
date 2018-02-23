# Made by Ian Burwell

from tkinter import *


class GUI:
    def __init__(self, font_size):
        root = Tk()

        #input
        Label(root, text="Input:").pack()
        self.sv = StringVar()
        self.e = Entry(root, textvariable=self.sv, font=("Consolas", font_size, "normal"))
        self.e.pack(fill=X, expand=1, padx=10)
        self.e.delete(0, END)

        #output
        Label(root, text="Output:").pack()
        self.t = Text(root, height=1, wrap='none', font=("Consolas", font_size, "normal"))
        self.t.pack(fill=X, expand=1, padx=10)

        #setup scrolling
        s = Scrollbar(root,orient=HORIZONTAL)
        s.pack(side=BOTTOM, fill=X, expand=1, pady=5)

        def setxview(com, ammount, *args):
            if len(args) > 0:
                self.t.xview(com, ammount, args[0])
                self.e.xview(com, ammount, args[0])
            else:
                self.t.xview(com, ammount)
                self.e.xview(com, ammount)

        def scroll_set(lo, hi):
            s.set(lo, hi)
            self.e.xview_moveto(s.get()[0])
            self.t.xview_moveto(s.get()[0])

        s.config(command=setxview)
        self.t.config(xscrollcommand=scroll_set)
        self.e.config(xscrollcommand=scroll_set)

    def run(self):
        mainloop()

    def set_callback(self, edit_callback_func):
        self.sv.trace("w", lambda name, index, mode, sv=self.sv: edit_callback_func(sv.get()))

    def set_out_text(self, text):
        self.t.delete(1.0, END)
        self.t.insert(END, text)


if __name__ == "__main__":
    gui = GUI(50)

    def callback(text):
        gui.set_out_text(text)

    gui.set_callback(callback)
    gui.run()
