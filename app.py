from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import requests
import wikipedia
from googletrans import Translator

app = Flask(__name__)
model = load_model('traffic_classifier.h5')

classes = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)', 9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited', 17:'No entry',
    18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 22:'Bumpy road',
    23:'Slippery road', 24:'Road narrows on the right', 25:'Road work', 26:'Traffic signals', 27:'Pedestrians',
    28:'Children crossing', 29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only',
    36:'Go straight or right', 37:'Go straight or left', 38:'Keep right', 39:'Keep left',
    40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    mode = None
    prediction = None
    confidence = None
    image_path = None
    search_images = []
    search_descs = []
    wiki_description = ""
    wiki_full = ""
    query = None
    full_info = None

    if request.method == 'POST':
        mode = request.form.get('mode')

        # Handle classification
        if mode == 'classify':
            file = request.files.get('file')
            if file and file.filename != '':
                filepath = os.path.join('static', file.filename)
                file.save(filepath)
                class_idx, confidence = model_predict(filepath)
                prediction = classes.get(class_idx, "Unknown")
                image_path = filepath
            elif 'file' in request.files:
                return render_template('index.html', mode=mode, error="Please select an image.", classes=classes)

        # Handle search
        elif mode == 'search':
            query = request.form.get('sign_name')
            full_info = request.form.get('full_info')

            if query:
                # Google Images
                api_key = "AIzaSyAxlSkcip8MtFRXNIVt0GT8AYMHAaoP7O8"
                cx = "401d18a9be38b4035"
                response = requests.get(
                    f"https://www.googleapis.com/customsearch/v1?q={query}+traffic+sign&searchType=image&num=5&key={api_key}&cx={cx}"
                )
                data = response.json()
                if 'items' in data:
                    for item in data['items']:
                        search_images.append(item['link'])
                        search_descs.append(item.get('title', 'Traffic Sign'))

                # Wikipedia Description
                try:
                    wiki_full = wikipedia.summary(f"{query} traffic sign", sentences=5)
                    if not full_info:
                        wiki_description = wikipedia.summary(f"{query} traffic sign", sentences=2)
                    else:
                        wiki_description = wiki_full
                except:
                    wiki_description = "No detailed description found."

                # Translation to English (can be re-translated on frontend)
                try:
                    translator = Translator()
                    translation = translator.translate(wiki_description, dest='en')
                    wiki_description = translation.text
                except:
                    pass

    return render_template(
        'index.html',
        mode=mode,
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        search_images=search_images,
        search_query=query,
        search_descs=search_descs,
        wiki_description=wiki_description,
        wiki_full=wiki_full,
        full_info=full_info,
        classes=classes,
        zip=zip
    )

def model_predict(img_path):
    image = Image.open(img_path)
    image = image.resize((30, 30))
    image = np.array(image)
    image = image.reshape(1, 30, 30, 3)
    prediction = model.predict(image)
    confidence = float(np.max(prediction))
    class_index = int(np.argmax(prediction))
    return class_index, confidence

if __name__ == '__main__':
    app.run(debug=True)
