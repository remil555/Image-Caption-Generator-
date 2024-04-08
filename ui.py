import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2

# Load the trained model to classify sign
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from pickle import load
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load InceptionV3 model
base_model = InceptionV3(weights='inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
vgg_model = Model(base_model.input, base_model.layers[-2].output)

# Load the trained model
model = load_model('new-model-1.h5')

# Load wordtoix and ixtoword dictionaries
with open("wordtoix.pkl", "rb") as pickle_in:
    wordtoix = load(pickle_in)
with open("ixtoword.pkl", "rb") as pickle_in:
    ixtoword = load(pickle_in)
max_length = 74

# Function to preprocess image for prediction
def preprocess_img(img_path):
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Function to encode image
def encode(image):
    image = preprocess_img(image)
    vec = vgg_model.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec

# Function for greedy search
def greedy_search(pic, model):
    start = 'startseq'
    for i in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([pic, seq])
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        start += ' ' + word
        if word == 'endseq':
            break
    final = start.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

# Function for beam search
def beam_search(image, model, beam_index=3):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length)
            e = image
            preds = model.predict([e, np.array(par_caps)])
            word_preds = np.argsort(preds[0])[-beam_index:]
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Caption Generator')
top.configure(background='#F0F0F0')  # Set background color

# Function to classify the uploaded image
def classify(file_path):
    enc = encode(file_path)
    image = enc.reshape(1, 2048)
    
    # Perform caption generation using greedy search
    greedy_caption = greedy_search(image, model)
    
    # Perform caption generation using beam search (beam width = 3)
    beam_3_caption = beam_search(image, model)
    
    # Compare the captions and choose the better one
    if len(greedy_caption) > len(beam_3_caption):
        final_caption = greedy_caption
    else:
        final_caption = beam_3_caption
    
    label.configure(foreground='#000', text='Caption: ' + final_caption)
    label.pack(side=BOTTOM, expand=True)

# Function to display the classify button after uploading an image
def show_classify_button(file_path):
    classify_b = Button(top, text="Generate Caption", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#007BFF', foreground='white', font=('Arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

# Function to upload image
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

# Button to upload image
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background='#007BFF', foreground='white', font=('Arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)

# Display uploaded image
sign_image = Label(top)
sign_image.pack(side=BOTTOM, expand=True)

# Label for the title
heading = Label(top, text="Image Caption Generator", pady=20, font=('Arial', 22, 'bold'))
heading.configure(background='#F0F0F0', foreground='#007BFF')  # Set background and foreground color
heading.pack()

# Label for displaying the caption
label = Label(top, background='#F0F0F0', font=('Arial', 15))
label.pack(side=BOTTOM, expand=True)

top.mainloop()
