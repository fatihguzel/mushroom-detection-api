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

model = load_model("google_model.h5")

# Class indices dictionary
class_indices_mobile = {
    'Amanita citrina - Yenilemez': 0, 'Amanita muscaria - Yenilemez': 1, 'Amanita pantherina - Yenilemez': 2, 'Amanita rubescens - Yenilebilir': 3,
    'Apioperdon pyriforme - Yenilebilir': 4, 'Armillaria borealis - Yenilebilir': 5, 'Artomyces pyxidatus - Yenilebilir': 6, 'Bjerkandera adusta - Yenilemez': 7,
    'Boletus edulis - Yenilebilir': 8, 'Boletus reticulatus - Yenilebilir': 9, 'Calocera viscosa - Yenilemez': 10, 'Calycina citrina - Yenilemez': 11,
    'Cantharellus cibarius - Yenilebilir': 12, 'Cerioporus squamosus - Yenilebilir': 13, 'Cetraria islandica - Yenilebilir': 14, 'Chlorociboria aeruginascens - Yenilemez': 15,
    'Chondrostereum purpureum - Yenilemez': 16, 'Cladonia fimbriata - Yenilemez': 17, 'Cladonia rangiferina - Yenilebilir': 18, 'Cladonia stellaris - Yenilebilir': 19,
    'Clitocybe nebularis - Yenilemez': 20, 'Coltricia perennis - Yenilemez': 21, 'Coprinellus disseminatus - Yenilebilir': 22, 'Coprinellus micaceus - Yenilebilir': 23,
    'Coprinopsis atramentaria - Yenilemez': 24, 'Coprinus comatus - Yenilebilir': 25, 'Crucibulum laeve - Yenilemez': 26, 'Daedaleopsis confragosa - Yenilemez': 27,
    'Daedaleopsis tricolor - Yenilemez': 28, 'Evernia mesomorpha - Yenilemez': 29, 'Evernia prunastri - Yenilemez': 30, 'Flammulina velutipes - Yenilebilir': 31,
    'Fomes fomentarius - Yenilemez': 32, 'Fomitopsis betulina - Yenilemez': 33, 'Fomitopsis pinicola - Yenilemez': 34, 'Ganoderma applanatum - Yenilemez': 35,
    'Graphis scripta - Yenilemez': 36, 'Gyromitra esculenta - Yenilemez': 37, 'Gyromitra gigas - Yenilemez': 38, 'Gyromitra infula - Yenilemez': 39,
    'Hericium coralloides - Yenilebilir': 40, 'Hygrophoropsis aurantiaca - Yenilemez': 41, 'Hypholoma fasciculare - Yenilemez': 42, 'Hypholoma lateritium - Yenilebilir': 43,
    'Hypogymnia physodes - Yenilemez': 44, 'Imleria badia - Yenilebilir': 45, 'Inonotus obliquus - Yenilebilir': 46, 'Kuehneromyces mutabilis - Yenilebilir': 47,
    'Lactarius deliciosus - Yenilebilir': 48, 'Lactarius torminosus - Yenilemez': 49, 'Lactarius turpis - Yenilemez': 50, 'Laetiporus sulphureus - Yenilebilir': 51,
    'Leccinum albostipitatum - Yenilebilir': 52, 'Leccinum aurantiacum - Yenilebilir': 53, 'Leccinum scabrum - Yenilebilir': 54, 'Leccinum versipelle - Yenilebilir': 55,
    'Lepista nuda - Yenilebilir': 56, 'Lobaria pulmonaria - Yenilebilir': 57, 'Lycoperdon perlatum - Yenilebilir': 58, 'Macrolepiota procera - Yenilebilir': 59,
    'Merulius tremellosus - Yenilemez': 60, 'Mutinus ravenelii - Yenilemez': 61, 'Nectria cinnabarina - Yenilemez': 62, 'Panellus stipticus - Yenilemez': 63,
    'Parmelia sulcata - Yenilemez': 64, 'Paxillus involutus - Yenilemez': 65, 'Peltigera aphthosa - Yenilebilir': 66, 'Peltigera praetextata - Yenilemez': 67,
    'Phaeophyscia orbicularis - Yenilemez': 68, 'Phallus impudicus - Yenilemez': 69, 'Phellinus igniarius - Yenilemez': 70, 'Phellinus tremulae - Yenilemez': 71,
    'Phlebia radiata - Yenilemez': 72, 'Pholiota aurivella - Yenilebilir': 73, 'Pholiota squarrosa - Yenilemez': 74, 'Physcia adscendens - Yenilemez': 75,
    'Platismatia glauca - Yenilemez': 76, 'Pleurotus ostreatus - Yenilebilir': 77, 'Pleurotus pulmonarius - Yenilebilir': 78, 'Pseudevernia furfuracea - Yenilemez': 79,
    'Rhytisma acerinum - Yenilemez': 80, 'Sarcomyxa serotina - Yenilebilir': 81, 'Sarcoscypha austriaca - Yenilemez': 82, 'Sarcosoma globosum - Yenilemez': 83,
    'Schizophyllum commune - Yenilebilir': 84, 'Stereum hirsutum - Yenilemez': 85, 'Stropharia aeruginosa - Yenilemez': 86, 'Suillus granulatus - Yenilebilir': 87,
    'Suillus grevillei - Yenilebilir': 88, 'Suillus luteus - Yenilebilir': 89, 'Trametes hirsuta - Yenilemez': 90, 'Trametes ochracea - Yenilemez': 91,
    'Trametes versicolor - Yenilebilir': 92, 'Tremella mesenterica - Yenilebilir': 93, 'Trichaptum biforme - Yenilemez': 94, 'Tricholomopsis rutilans - Yenilemez': 95,
    'Urnula craterium - Yenilemez': 96, 'Verpa bohemica - Yenilebilir': 97, 'Vulpicida pinastri - Yenilemez': 98, 'Xanthoria parietina - Yenilemez': 99
}

# OpenAI API key configuration
openai.api_key = 'sk-proj-Wf5pvmDri1gKKJnVbKDZT3BlbkFJbqTsLh9IEruCuZQRkcMY'

# api url
url = 'https://api.openai.com/v1/engines/gpt-3.5-turbo/completions'


def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((256, 256))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
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
        print(f'Image array shape: {image_array.shape}')
        predicted_prob = model.predict(image_array)
        predicted_class = np.argmax(predicted_prob)
        predicted_prob_value = predicted_prob[0][predicted_class]
        print(f'Predicted probabilities: {predicted_prob}')
        print(f'Predicted class: {predicted_class}')
        print(f'Predicted probability: {predicted_prob_value}')
        return predicted_class, float(predicted_prob_value)
    except Exception as e:
        print(f"Error classifying image: {e}")
        return None, None


def get_wikipedia_data(class_name):
    try:
        print(f'Getting Wikipedia data for class: {class_name}')
        encoded_class_name = urllib.parse.quote(class_name)
        wikipedia_url = f"https://en.wikipedia.org/wiki/{encoded_class_name}"

        response = requests.get(wikipedia_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join([paragraph.text for paragraph in paragraphs])
            content = content.replace('\n', ' ')
            content = content.replace('\xa0', ' ')

            return content
        else:
            print(f"Failed to fetch Wikipedia data. Status code: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error getting Wikipedia data: {e}")
        return None


def get_chatgpt_data(class_name):
    try:
        print(f'Getting ChatGPT data for class: {class_name}')
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f'{class_name} hakkında bilgi verir misin'
                }
            ],
        )
        print(response)
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

            # Determine if the mushroom is edible
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
