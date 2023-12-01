
from flask import Flask, request, render_template
from transformers import ViTForImageClassification,  ViTFeatureExtractor
import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./uploads"

checkpoint = 'google/vit-base-patch16-224'
model = ViTForImageClassification.from_pretrained(checkpoint, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load('model.pth'))
feature_extractor = ViTFeatureExtractor.from_pretrained(checkpoint)

labels = os.listdir("./Micro_Organism")
decode_labels = {0: 'Amoeba',
                 1: 'Euglena', 2: 'Hydra',
                 3: 'Paramecium', 4: 'Rod_bacteria',
                 5: 'Spherical_bacteria', 6: 'Spiral_bacteria', 
                 7: 'Yeast'}
@app.route("/", methods=['GET','POST'])
def home():
    if request.method == "POST":
        image = request.files.get("img", '')
        img = Image.open(image)
        image = os.path.join(app.config['UPLOAD_FOLDER'],request.form.get("img", ''))
        inputs = feature_extractor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        pred = logits.argmax(-1).item()
        label = decode_labels[pred]
        return render_template('index.html', label=label, image=image)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=5000)