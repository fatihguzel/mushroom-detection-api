from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model once during application startup
model = load_model("mobile_model.h5")

# Class indices dictionary
class_indices_mobile = {'Amanita citrina': 0,
'Amanita muscaria': 1,
'Amanita pantherina': 2,
'Amanita rubescens': 3,
'Apioperdon pyriforme': 4,
'Armillaria borealis': 5,
'Artomyces pyxidatus': 6,
'Bjerkandera adusta': 7,
'Boletus edulis': 8,
'Boletus reticulatus': 9,
'Calocera viscosa': 10,
'Calycina citrina': 11,
'Cantharellus cibarius': 12,
'Cerioporus squamosus': 13,
'Cetraria islandica': 14,
'Chlorociboria aeruginascens': 15,
'Chondrostereum purpureum': 16,
'Cladonia fimbriata': 17,
'Cladonia rangiferina': 18,
'Cladonia stellaris': 19,
'Clitocybe nebularis': 20,
'Coltricia perennis': 21,
'Coprinellus disseminatus': 22,
'Coprinellus micaceus': 23,
'Coprinopsis atramentaria': 24,
'Coprinus comatus': 25,
'Crucibulum laeve': 26,
'Daedaleopsis confragosa': 27,
'Daedaleopsis tricolor': 28,
'Evernia mesomorpha': 29,
'Evernia prunastri': 30,
'Flammulina velutipes': 31,
'Fomes fomentarius': 32,
'Fomitopsis betulina': 33,
'Fomitopsis pinicola': 34,
'Ganoderma applanatum': 35,
'Graphis scripta': 36,
'Gyromitra esculenta': 37,
'Gyromitra gigas': 38,
'Gyromitra infula': 39,
'Hericium coralloides': 40,
'Hygrophoropsis aurantiaca': 41,
'Hypholoma fasciculare': 42,
'Hypholoma lateritium': 43,
'Hypogymnia physodes': 44,
'Imleria badia': 45,
'Inonotus obliquus': 46,
'Kuehneromyces mutabilis': 47,
'Lactarius deliciosus': 48,
'Lactarius torminosus': 49,
'Lactarius turpis': 50,
'Laetiporus sulphureus': 51,
'Leccinum albostipitatum': 52,
'Leccinum aurantiacum': 53,
'Leccinum scabrum': 54,
'Leccinum versipelle': 55,
'Lepista nuda': 56,
'Lobaria pulmonaria': 57,
'Lycoperdon perlatum': 58,
'Macrolepiota procera': 59,
'Merulius tremellosus': 60,
'Mutinus ravenelii': 61,
'Nectria cinnabarina': 62,
'Panellus stipticus': 63,
'Parmelia sulcata': 64,
'Paxillus involutus': 65,
'Peltigera aphthosa': 66,
'Peltigera praetextata': 67,
'Phaeophyscia orbicularis': 68,
'Phallus impudicus': 69,
'Phellinus igniarius': 70,
'Phellinus tremulae': 71,
'Phlebia radiata': 72,
'Pholiota aurivella': 73,
'Pholiota squarrosa': 74,
'Physcia adscendens': 75,
'Platismatia glauca': 76,
'Pleurotus ostreatus': 77,
'Pleurotus pulmonarius': 78,
'Pseudevernia furfuracea': 79,
'Rhytisma acerinum': 80,
'Sarcomyxa serotina': 81,
'Sarcoscypha austriaca': 82,
'Sarcosoma globosum': 83,
'Schizophyllum commune': 84,
'Stereum hirsutum': 85,
'Stropharia aeruginosa': 86,
'Suillus granulatus': 87,
'Suillus grevillei': 88,
'Suillus luteus': 89,
'Trametes hirsuta': 90,
'Trametes ochracea': 91,
'Trametes versicolor': 92,
'Tremella mesenterica': 93,
'Trichaptum biforme': 94,
'Tricholomopsis rutilans': 95,
'Urnula craterium': 96,
'Verpa bohemica': 97,
'Vulpicida pinastri': 98,
'Xanthoria parietina': 99}


def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  
    img_array = np.array(img)  
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def classify_image(image_array):
    print(f'Image array shape: {image_array.shape}')
    
    predicted_prob = model.predict(image_array)
    
    predicted_class = np.argmax(predicted_prob)
    
    predicted_prob_value = predicted_prob[0][predicted_class]
    
    print(f'Predicted probabilities: {predicted_prob}')
    
    print(f'Predicted class: {predicted_class}')
    print(f'Predicted probability: {predicted_prob_value}')
    
    return predicted_class, float(predicted_prob_value)


def get_true_labels(predicted_class):
    print(f'Predicted class: {predicted_class}')
    true_labels = [key for key, value in class_indices_mobile.items() if value == predicted_class]
    print(f'True labels: {true_labels}')
    return true_labels

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
        
        image_array = preprocess_image(file_path)
        
        predicted_class, predicted_prob = classify_image(image_array)
        
        true_labels = get_true_labels(predicted_class)
        
        predicted_class = int(predicted_class)
        
        os.remove(file_path)
        
        return jsonify({
            'predicted_class': predicted_class,
            'predicted_prob': predicted_prob,
            'true_labels': true_labels
        })

    

if __name__ == '__main__':
    app.run(debug=True)
