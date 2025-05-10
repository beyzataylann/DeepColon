import pandas as pd
import os
import argparse

# Komut satırı argümanları
parser = argparse.ArgumentParser(description="Belirli bir görsel için özellik dosyalarını birleştir.")
parser.add_argument('filename', type=str, help="Birleştirilecek dosyanın adı (örneğin: train)")
args = parser.parse_args()

# Dosya adı işlemleri
filename = args.filename
basename = os.path.splitext(filename)[0]

# Klasör yolları
lbp_dir = "lbp_result"
googlenet_dir = "googlenet"
resnet_dir = "resnet"
output_dir = "combined_features"

# Çıktı klasörünü oluştur
os.makedirs(output_dir, exist_ok=True)

# Dosya yolları
lbp_path = os.path.join(lbp_dir, f"{basename}_lbp_features.csv")
googlenet_path = os.path.join(googlenet_dir, f"{basename}_googlenet_features.csv")
resnet_path = os.path.join(resnet_dir, f"{basename}_resnet_features.csv")

# CSV'leri oku
try:
    lbp_df = pd.read_csv(lbp_path)
    googlenet_df = pd.read_csv(googlenet_path)
    resnet_df = pd.read_csv(resnet_path)
except Exception as e:
    print(f"CSV dosyaları okunurken hata oluştu: {e}")
    exit(1)

'''
for df in [googlenet_df, resnet_df, lbp_df]:
    if 'filename' in df.columns:
        df.rename(columns={'filename': 'image_name'}, inplace=True)'''



# Sonucu yeni bir CSV dosyasına yaz
merged_df = pd.merge(googlenet_df, resnet_df, on=['image_name'], how='inner')
merged_df = pd.merge(merged_df, lbp_df, on=['image_name'], how='inner')


# Çıktıyı kaydet
combined_output_path = os.path.join(output_dir, f"{basename}_combined_features.csv")
merged_df.to_csv(combined_output_path, index=False)

# Örnek çıktı göster
print(f"{basename} için özellikler başarıyla birleştirildi.")
print(f"CSV kaydedildi: {combined_output_path}")
print(merged_df.head())
