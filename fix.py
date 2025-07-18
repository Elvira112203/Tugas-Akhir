from flask import Flask, request, jsonify
import pandas as pd
from sklearn.svm import SVC
import numpy as np

app = Flask(__name__)  

# Load dataset
print("Memuat dataset...")
data = pd.read_csv('/storage/emulated/0/Documents/Svm/dataset_kategori_3kelas.csv')

# Preprocess
x = data[['BPM', 'Kemiringan']]
y = data['Kelas']

# Train model
print("Melatih model SVM...")
model = SVC()
model.fit(x, y)

# Define label names
label_names = {
    1: "Normal",
    2: "Mengantuk Sedang",
    3: "Mengantuk Berat"
}

print("Label yang tersedia:", np.unique(y))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Menerima request dari Flutter")
        print("Data masuk:", data)

        # Konversi BPM ke integer dan Kemiringan ke float
        try:
            bpm = int(data.get('BPM', 0))
            angle = float(data.get('Kemiringan', 0))
            print(f"Data setelah konversi - BPM: {bpm} (type: {type(bpm)}), Kemiringan: {angle:.2f} (type: {type(angle)})")
        except (TypeError, ValueError) as e:
            print(f"Error konversi data: {e}")
            return jsonify({'error': 'Invalid data type'}), 400

        if bpm == 0 or angle == 0:
            print("Data tidak lengkap: BPM atau Kemiringan kosong")
            return jsonify({'error': 'Missing data'}), 400

        # Format data sesuai dengan format training
        input_data = [[bpm, angle]]
        print(f"Data yang akan diprediksi: {input_data}")
        
        # Lakukan prediksi
        prediction = model.predict(input_data)[0]
        # Konversi numpy.int64 ke integer Python biasa
        pred_label = int(prediction)
        # Dapatkan label nama
        label_name = label_names.get(pred_label, f"Tidak Dikenal ({pred_label})")
        
        print(f"Hasil prediksi (numerik): {pred_label}")
        print(f"Hasil prediksi (label): {label_name}")
        
        return jsonify({
            'label': label_name,
            'code': pred_label
        })

    except Exception as e:
        print(f"Terjadi error saat memproses request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':   
    print("Menjalankan Flask server di port 8080...")
    app.run(host='0.0.0.0', port=8080)