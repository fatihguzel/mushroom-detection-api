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

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize image to match model's expected sizing
    img_array = np.array(img)  # Convert PIL image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def classify_image(image_array):
    print(f'Image array shape: {image_array.shape}')
    
    # Modelin tahminlerini al
    predicted_prob = model.predict(image_array)
    
    # Tahmin edilen sınıfı belirle
    predicted_class = np.argmax(predicted_prob)
    
    # Tahmin edilen sınıfın olasılığını al
    predicted_prob_value = predicted_prob[0][predicted_class]
    
    # Loglama ile tüm sınıfların olasılıklarını yazdır
    print(f'Predicted probabilities: {predicted_prob}')
    
    # En yüksek olasılığa sahip sınıfı ve bu sınıfın olasılığını yazdır
    print(f'Predicted class: {predicted_class}')
    print(f'Predicted probability: {predicted_prob_value}')
    
    # Tahmin edilen sınıf ve olasılığını döndür
    return predicted_class, float(predicted_prob_value)


def get_true_labels(predicted_class):
    print(f'Predicted class: {predicted_class}')
    true_labels = [key for key, value in class_indices_mobile.items() if value == predicted_class]
    print(f'True labels: {true_labels}')
    return true_labels

# Endpoint to handle image classification requests
@app.route('/classify', methods=['POST'])
def classify_from_post():
    # Check if the post request has the file part
    if 'photo' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['photo']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('', filename)
        file.save(file_path)
        
        # Resim dosyasını ön işleme tabi tut
        image_array = preprocess_image(file_path)
        
        # Model ile sınıflandırma yap
        predicted_class, predicted_prob = classify_image(image_array)
        
        # Tahmin edilen sınıfı, class_indices_mobile sözlüğü ile eşleştir
        true_labels = get_true_labels(predicted_class)
        
        # Tahmin edilen sınıfı integer olarak döndür (JSON serileştirme hatalarını önlemek için)
        predicted_class = int(predicted_class)
        
        # Kaydedilen dosyayı sil
        # os.remove(file_path)
        
        # Sonucu JSON formatında döndür
        return jsonify({
            'predicted_class': predicted_class,
            'predicted_prob': predicted_prob,
            'true_labels': true_labels
        })

    

if __name__ == '__main__':
    app.run(debug=True)
