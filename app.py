import base64
import openai.upload_progress
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
import urllib.parse
from googletrans import Translator
import openai

app = Flask(__name__)

model = load_model("google_model(86).h5")


class_indices_mobile = {
    'Amanita muscaria - Yenilemez': 0,
    'Amanita pantherina - Yenilemez': 1,
    'Armillaria borealis - Yenilebilir': 2,
    'Cantharellus cibarius - Yenilebilir':3,
    'Cerioporus squamosus - Yenilebilir': 4,
    'Clitocybe nebularis - Yenilemez': 5,
    'Coltricia perennis - Yenilemez': 6,
    'Gyromitra infula - Yenilemez': 7,
    'Hericium coralloides - Yenilebilir': 8,
    'Lactarius torminosus - Yenilemez':     9,
    'Leccinum albostipitatum - Yenilemez':  10,
    'Leccinum scabrum - Yenilebilir':  11,
    'Lepista nuda - Yenilebilir':  12,
    'Macrolepiota procera - Yenilebilir':  13,
    'Mutinus ravenelii - Yenilemez':  14,
    'Paxillus involutus - Yenilemez':  15,
    'Phallus impudicus - Yenilemez':  16,
    'Pholiota aurivella - Yenilebilir': 17,
    'Sarcosoma globosum - Yenilemez' :  18,
    'Stereum hirsutum - Yenilemez':  19,
    'Trametes versicolor - Yenilebilir':  20,
    'Xanthoria parietina - Yenilemez':  21,
}



openai.api_key = 'sk-proj-Wf5pvmDri1gKKJnVbKDZT3BlbkFJbqTsLh9IEruCuZQRkcMY'

# api url
url = 'https://api.openai.com/v1/engines/gpt-3.5-turbo/completions'


def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((256, 256))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0


        return img_array
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        return None
    except OSError as e:
        print(f"Error opening image: {e}")
        return None
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def classify_image(image_array):
    try:
        predicted_prob = model.predict(image_array)
        predicted_class = np.argmax(predicted_prob)
        predicted_prob_value = np.max(predicted_prob) * 100

        return predicted_class, int(predicted_prob_value)
    except Exception as e:
        print(f"Error classifying image: {e}")
        return None, None


def get_chatgpt_data(class_name):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f'{class_name} hakkında bilgi verir misin'
                }
            ],
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error getting ChatGPT data: {e}")
        return None


@app.route('/classify', methods=['POST'])
def classify_from_post():
    if 'photo' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['photo']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('', filename)
        file.save(file_path)

        try:
            image_array = preprocess_image(file_path)

            if image_array is None:
                return jsonify({'error': 'Error preprocessing image.'})

            predicted_class, predicted_prob = classify_image(image_array)

            if predicted_class is None:
                return jsonify({'error': 'Error predicting image class.'})

            true_labels = [key for key, value in class_indices_mobile.items() if value == predicted_class]

            if not true_labels:
                return jsonify({'error': 'Error retrieving true labels.'})

            chatgpt_data = get_chatgpt_data(true_labels[0])

            os.remove(file_path)

            can_eat = 'Yenilebilir' in true_labels[0]

            return jsonify({
                'predicted_class': int(predicted_class),
                'predicted_prob': float(predicted_prob),
                'true_labels': true_labels[0].split(' - ')[0],
                'canEat': can_eat,
                'chatgpt_data': chatgpt_data if chatgpt_data else ''
            })

        except Exception as e:
            print(f"An error occurred: {e}")
            return jsonify({'error': 'An error occurred.'})

    return jsonify({'error': 'Unknown error occurred.'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
