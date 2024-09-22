from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

# Modeli yükle
model = load_model('ethnicity_model.h5')

# Etnik köken etiketleri
ethnic_labels = {
    0: 'Beyaz',
    1: 'Siyah',
    2: 'Asyalı',
    3: 'İspanyol',
    4: 'Diğer'
}

# Resim tahmini yapan fonksiyon
def predict_ethnicity(image_path):
    img = load_img(image_path, target_size=(64, 64))  # Boyutu kontrol et
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return ethnic_labels[predicted_class]

# Ana sayfa route'u
@app.route('/')
def home():
    return render_template('index.html')

# Tahmin endpointi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['file']
        image_path = './' + image_file.filename
        image_file.save(image_path)
        prediction = predict_ethnicity(image_path)

        # Geçici dosyayı sil
        os.remove(image_path)

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
