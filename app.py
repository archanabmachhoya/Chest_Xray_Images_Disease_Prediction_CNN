from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
model = load_model('CNN_model.h5')
lb = LabelBinarizer()

# Load labels from CSV
labels_df = pd.read_csv('labels.csv')
class_names = labels_df.columns.tolist()

# Convert DataFrame back to numpy array
labels = labels_df.values
labels = lb.fit_transform(labels)
labels = np.array(labels)

def load_image(file, target_size=(224, 224)):
    img = Image.open(file).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file is not None:
        filename = file.filename
        file_path = os.path.join('static', 'images', filename)
        file.save(file_path)
        img = load_image(file)
        prediction = model.predict(img)
        
        # Collect class names with prediction probability > 0.5
        predicted_labels = [class_names[i] for i in range(len(prediction[0])) if prediction[0][i] > 0.5]
        
        return render_template('index.html', prediction=predicted_labels, img_path='images/' + file.filename)
    return None

if __name__ == '__main__':
    app.run(debug=True)
