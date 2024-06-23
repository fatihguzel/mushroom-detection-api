import base64
import openai.upload_progress
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model # type: ignore
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
    'Amanita citrina': 0, 'Amanita muscaria': 1, 'Amanita pantherina': 2, 'Amanita rubescens': 3,
    'Apioperdon pyriforme': 4, 'Armillaria borealis': 5, 'Artomyces pyxidatus': 6, 'Bjerkandera adusta': 7,
    'Boletus edulis': 8, 'Boletus reticulatus': 9, 'Calocera viscosa': 10, 'Calycina citrina': 11,
    'Cantharellus cibarius': 12, 'Cerioporus squamosus': 13, 'Cetraria islandica': 14, 'Chlorociboria aeruginascens': 15,
    'Chondrostereum purpureum': 16, 'Cladonia fimbriata': 17, 'Cladonia rangiferina': 18, 'Cladonia stellaris': 19,
    'Clitocybe nebularis': 20, 'Coltricia perennis': 21, 'Coprinellus disseminatus': 22, 'Coprinellus micaceus': 23,
    'Coprinopsis atramentaria': 24, 'Coprinus comatus': 25, 'Crucibulum laeve': 26, 'Daedaleopsis confragosa': 27,
    'Daedaleopsis tricolor': 28, 'Evernia mesomorpha': 29, 'Evernia prunastri': 30, 'Flammulina velutipes': 31,
    'Fomes fomentarius': 32, 'Fomitopsis betulina': 33, 'Fomitopsis pinicola': 34, 'Ganoderma applanatum': 35,
    'Graphis scripta': 36, 'Gyromitra esculenta': 37, 'Gyromitra gigas': 38, 'Gyromitra infula': 39,
    'Hericium coralloides': 40, 'Hygrophoropsis aurantiaca': 41, 'Hypholoma fasciculare': 42, 'Hypholoma lateritium': 43,
    'Hypogymnia physodes': 44, 'Imleria badia': 45, 'Inonotus obliquus': 46, 'Kuehneromyces mutabilis': 47,
    'Lactarius deliciosus': 48, 'Lactarius torminosus': 49, 'Lactarius turpis': 50, 'Laetiporus sulphureus': 51,
    'Leccinum albostipitatum': 52, 'Leccinum aurantiacum': 53, 'Leccinum scabrum': 54, 'Leccinum versipelle': 55,
    'Lepista nuda': 56, 'Lobaria pulmonaria': 57, 'Lycoperdon perlatum': 58, 'Macrolepiota procera': 59,
    'Merulius tremellosus': 60, 'Mutinus ravenelii': 61, 'Nectria cinnabarina': 62, 'Panellus stipticus': 63,
    'Parmelia sulcata': 64, 'Paxillus involutus': 65, 'Peltigera aphthosa': 66, 'Peltigera praetextata': 67,
    'Phaeophyscia orbicularis': 68, 'Phallus impudicus': 69, 'Phellinus igniarius': 70, 'Phellinus tremulae': 71,
    'Phlebia radiata': 72, 'Pholiota aurivella': 73, 'Pholiota squarrosa': 74, 'Physcia adscendens': 75,
    'Platismatia glauca': 76, 'Pleurotus ostreatus': 77, 'Pleurotus pulmonarius': 78, 'Pseudevernia furfuracea': 79,
    'Rhytisma acerinum': 80, 'Sarcomyxa serotina': 81, 'Sarcoscypha austriaca': 82, 'Sarcosoma globosum': 83,
    'Schizophyllum commune': 84, 'Stereum hirsutum': 85, 'Stropharia aeruginosa': 86, 'Suillus granulatus': 87,
    'Suillus grevillei': 88, 'Suillus luteus': 89, 'Trametes hirsuta': 90, 'Trametes ochracea': 91,
    'Trametes versicolor': 92, 'Tremella mesenterica': 93, 'Trichaptum biforme': 94, 'Tricholomopsis rutilans': 95,
    'Urnula craterium': 96, 'Verpa bohemica': 97, 'Vulpicida pinastri': 98, 'Xanthoria parietina': 99
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

            wikipedia_data = get_wikipedia_data(true_labels[0])
            chatgpt_data = get_chatgpt_data(true_labels[0])

            if wikipedia_data is None and chatgpt_data is None:
                return jsonify({'error': 'Error retrieving information data.'})
            
            os.remove(file_path)

            return jsonify({
                'predicted_class': int(predicted_class),
                'predicted_prob': float(predicted_prob),
                'true_labels': true_labels,
                'wikipedia_data': wikipedia_data.split('.')[0] if wikipedia_data else '',
                'chatgpt_data': chatgpt_data if chatgpt_data else ''
            })

        except Exception as e:
            print(f"An error occurred: {e}")
            return jsonify({'error': 'An error occurred.'})

    return jsonify({'error': 'Unknown error occurred.'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
