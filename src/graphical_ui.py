import sys

from tkinter.ttk import Frame, Button, Label, Style
from tkinter import BOTH, RIGHT, LEFT, TOP, E, Canvas, NW
from PIL import ImageTk, Image


class ImageDetailsConfirmation(Frame):
    """A SIMPLE UI FOR A SIMPLE MAN."""

    def __init__(self, card):
        super().__init__()
        self.card = card

        self.master.title("Image Details Confirmation")
        self.style = Style()
        self.style.theme_use("classic")
        self.pack(fill=BOTH, expand=True)
        self._load_image()
        self.init_ui()

    def init_ui(self):
        card_image = ImageTk.PhotoImage(self._image)
        image = Label(self, image=card_image)
        label_text = Label(self, text=f'Name: {self.card.card_name}, Set: {self.card.card_set}, Price: ${self.card.price}')

        image.image = card_image
        image.pack(side=TOP)
        label_text.pack(side=TOP)

        confirm_button = Button(self, text="Confirm", command=self._confirm_then_quit)
        confirm_button.pack(side=RIGHT, padx=5, pady=5)

        reject_button = Button(self, text="Reject", command=self._quit)
        reject_button.pack(side=RIGHT)

    def _load_image(self):
        try:
            self._image = Image.open(self.card.path_to_card_image).resize((300, 400), resample=0)
        except IOError:
            print("Unable to load card image.")
            sys.exit(1)

    def _confirm_then_quit(self):
        self.card.perform_create()
        self.master.destroy()

    def _quit(self):
        self.master.destroy()
