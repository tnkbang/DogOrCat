import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
from keras.models import load_model

model = load_model('Model.h5')
classes = { 
    0: 'Đây là con mèo!',
    1: 'Đây là con chó!',
}

top = tk.Tk()
top.geometry('500x400')
top.title('Phân loại chó mèo')
top.configure(background = '#CDCDCD')
label = Label(top, background = '#CDCDCD', font = ('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((128, 128))
    image = numpy.expand_dims(image, axis = 0)
    image = numpy.array(image)
    image = image / 255
    pred = numpy.argmax(model.predict([image])[0])
    sign = classes[pred]
    print(sign)
    label.configure(foreground = '#011638', text = sign)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image = im)
        sign_image.image = im
        label.configure(text = '')
        classify(file_path)
    except:
        pass

upload = Button(top, text = "Tải ảnh lên", command = upload_image, padx = 10, pady = 5)
upload.configure(background = '#364156', foreground = 'black', font = ('arial', 10, 'bold'))
upload.pack(side = BOTTOM, pady = 50)
sign_image.pack(side = BOTTOM, expand = True)
label.pack(side = BOTTOM, expand = True)
heading = Label(top, text = "Phân loại chó mèo", pady = 10, font = ('arial', 20, 'bold'))
heading.configure(background = '#CDCDCD', foreground = 'black')
heading.pack()
top.mainloop()
