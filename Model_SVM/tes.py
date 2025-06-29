import os
import numpy as np
import joblib

def load_model(model_filename='svm_model.pkl'):
    """
    Fungsi untuk memuat model dari file .pkl dengan logging dan error handling.
    """
    try:
        # Mendapatkan path relatif file model
        model_path = os.path.join(os.path.dirname(__file__), model_filename)
        print(f"[INFO] Mencoba memuat model dari: {model_path}")
        
        # Memuat model
        model = joblib.load(model_path)
        print("[INFO] Model berhasil dimuat.")
        return model
    except FileNotFoundError:
        print(f"[ERROR] File model tidak ditemukan: {model_filename}")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}")
    return None

def validate_input(input_data):
    """
    Fungsi untuk memvalidasi input data sebelum prediksi.
    Pastikan input berupa numpy array 2D dengan tipe float dan jumlah fitur sesuai.
    """
    if not isinstance(input_data, np.ndarray):
        print("[ERROR] Input data harus berupa numpy array.")
        return False
    if input_data.ndim != 2:
        print("[ERROR] Input data harus 2 dimensi (batch_size, fitur).")
        return False
    if input_data.dtype not in [np.float32, np.float64]:
        print("[WARNING] Mengkonversi input data ke float32.")
        try:
            input_data = input_data.astype(np.float32)
        except Exception as e:
            print(f"[ERROR] Gagal mengkonversi tipe data input: {e}")
            return False
    # Contoh validasi jumlah fitur, sesuaikan dengan model Anda
    expected_features = 2  # misal model Anda pakai 2 fitur: BPM dan Angle
    if input_data.shape[1] != expected_features:
        print(f"[ERROR] Jumlah fitur input harus {expected_features}, tapi diterima {input_data.shape[1]}.")
        return False
    return True

def predict(model, input_data):
    """
    Fungsi untuk melakukan prediksi dengan logging dan error handling.
    """
    if model is None:
        print("[ERROR] Model belum dimuat, tidak bisa melakukan prediksi.")
        return None
    
    if not validate_input(input_data):
        print("[ERROR] Validasi input gagal, prediksi dibatalkan.")
        return None
    
    try:
        prediction = model.predict(input_data)
        print(f"[INFO] Prediksi berhasil: {prediction}")
        return prediction
    except Exception as e:
        print(f"[ERROR] Gagal melakukan prediksi: {e}")
        return None

if __name__ == "__main__":
    # Contoh penggunaan
    model = load_model()
    
    # Contoh input data yang valid (batch 1, 2 fitur)
    sample_input = np.array([[90, 75]], dtype=np.float32)
    
    # Melakukan prediksi
    result = predict(model, sample_input)
    
    if result is not None:
        print(f"Hasil prediksi: {result}")
    else:
        print("Prediksi gagal.")
