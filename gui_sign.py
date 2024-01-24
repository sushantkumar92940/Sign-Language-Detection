import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Loading the Sign Language Detection Model
asl_model = load_model("asl_recognition_model.h5")

# Class labels for digits and letters
digit_labels = [str(i) for i in range(10)]
letter_labels = [chr(ord('a') + i) for i in range(26)]

# Initializing the GUI
top = tk.Tk()
top.geometry('800x600')
top.title("Sign Language Detector")
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, "bold"))
label2 = Label(top, background='#CDCDCD', font=('arial', 15, "bold"))
image_label = Label(top)

# Function to detect sign language
def detect_sign_language(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Making prediction using the Sign Language Detection Model
    prediction = asl_model.predict(image_array)[0]
    predicted_class_index = np.argmax(prediction)

    if predicted_class_index < 10:
        # Digit prediction
        predicted_label = digit_labels[predicted_class_index]
    else:
        # Letter prediction
        predicted_label = letter_labels[predicted_class_index - 10]

    label1.configure(foreground="#011638", text=f"Predicted Class: {predicted_label}")
    label2.configure(foreground="#011638", text=f"Probability: {prediction[predicted_class_index]:.4f}")

# Function to display the Detect Sign Language button
def show_detect_button(file_path):
    detect_button = Button(top, text="Detect Sign Language",
                      command=lambda: detect_sign_language(file_path), padx=10, pady=5)
    detect_button.configure(background="#364156",
                       foreground='white', font=("arial", 10, "bold"))
    detect_button.place(relx=0.79, rely=0.46)

# Function to upload an image
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded_image = Image.open(file_path)
        uploaded_image.thumbnail(
            ((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        image_tk = ImageTk.PhotoImage(uploaded_image)

        image_label.configure(image=image_tk)
        image_label.image = image_tk
        label1.configure(text="")
        label2.configure(text="")
        show_detect_button(file_path)
    except Exception as e:
        print(e)

upload_button = Button(top, text="Upload an Image",
                command=upload_image, padx=10, pady=5)
upload_button.configure(background="#364156", foreground="white",
                 font=("arial", 10, "bold"))
upload_button.pack(side="bottom", pady=50)

image_label.pack(side="bottom", expand=True)
label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)

heading = Label(top, text="Sign Language Detector",
                pady=20, font=("arial", 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()
