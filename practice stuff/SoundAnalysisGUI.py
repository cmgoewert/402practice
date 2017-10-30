from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename

root = Tk()
root.title("402 Sound Analysis")
root.geometry("400x400")



class SAGUI(Frame):

    def __init__(self, master):
        Frame.__init__(self,master)
        self.grid()
        self.create_widgets()

    def getfile(self):
        filename = askopenfilename()
        #print(filename)

    def create_widgets(self):
        self.welcomelabel = Label(self, text = "Welcome to the 402 Sample Analyzer", bg="red")
        self.welcomelabel.grid()

        self.instructionlabel = Label(self, text = "To choose a sound or directory of Sounds to upload press the button below!")
        self.instructionlabel.grid()

        self.dirsearchbutton = Button(self, text = "Search for Files", command = self.getfile)
        self.dirsearchbutton.grid()

        self.uploadedfilelabel = Label(self, text = "-------------")
        self.uploadedfilelabel.grid()





app = SAGUI(root)

root.mainloop()


