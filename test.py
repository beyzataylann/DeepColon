import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# --- Argparse ile dosya adı alınması ---
parser = argparse.ArgumentParser(description="Tek test verisi ile model testi.")
parser.add_argument('filename', type=str, help="Test verisi dosyası (örnek: test_sample.csv)")
args = parser.parse_args()

# Dosya adı
filename = args.filename
basename = os.path.splitext(filename)[0]

# Dosya yolları
pca_csv_path = os.path.join("combined_features", f"{basename}_combined_features.csv")
test_df = pd.read_csv(pca_csv_path)  # Dosya adı buraya yazılacak

X_test = test_df.drop(columns=["image_name"]).values  # 'image_name' dışında her şey özellik olarak kullanılır

# --- EĞİTİLMİŞ MODELİ YÜKLE ---
model = load("best_random_forest_modelrglbp.joblib")

# --- TAHMİN YAP ---
y_pred = model.predict(X_test)

# Tahmin sonuçlarını yazdır
print(f"Tahmin edilen sınıflar: {y_pred}")


predicted_class_names = ['colon_n' if label == 0 else 'colon_aca' for label in y_pred]
print(f"Tahmin edilen sınıf adları: {predicted_class_names}")



