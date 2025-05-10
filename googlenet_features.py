import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
from torchvision.models import GoogLeNet_Weights
import argparse

def main():
    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Görüntü yolu dosyadan okunur
    parser = argparse.ArgumentParser(description="Belirli bir görsel için özellik dosyalarını birleştir.")
    parser.add_argument('filename', type=str, help="")
    args = parser.parse_args()

    # Dosya adı işlemleri
    filename = args.filename
    basename = os.path.splitext(filename)[0]
    pre_dir = "preprocessing"


    # GoogleNet modelini yükle
    googlenet = models.googlenet(weights=GoogLeNet_Weights.DEFAULT).to(device)
    googlenet.eval()

    # Özellik çıkarıcı sınıf
    class FeatureExtractor(nn.Module):
        def __init__(self, model):
            super(FeatureExtractor, self).__init__()
            self.features = nn.Sequential(*list(model.children())[:-2])  # avgpool'dan önce

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            return x

    feature_extractor = FeatureExtractor(googlenet).to(device)

    # Görüntü dönüşümü
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    pre_output_path= os.path.join(pre_dir, f"{basename}.jpeg")


    # Görüntüyü yükle
    try:
        image = Image.open(pre_output_path).convert('RGB')
    except Exception as e:
        print(f"Görüntü açılırken hata oluştu: {e}")
        return

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Özellik çıkarımı
    with torch.no_grad():
        features = feature_extractor(image_tensor)
        features_np = features.cpu().numpy().squeeze()

    # Kayıt
    output_dir = "googlenet"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(pre_output_path))[0]
    csv_filename = f"{filename}_googlenet_features.csv"
    g_csv_path = os.path.join(output_dir, csv_filename)

    # DataFrame oluştur
    feature_columns = [f'feature_{i}' for i in range(features_np.shape[0])]
    data = [os.path.basename(pre_output_path)] + features_np.tolist()
    columns = ['image_name'] + feature_columns
    df = pd.DataFrame([data], columns=columns)

    # CSV dosyasını kaydet
    try:
        df.to_csv(g_csv_path, index=False)
        print(f"Özellikler CSV olarak kaydedildi: {g_csv_path}")
    except Exception as e:
        print(f"CSV kaydedilirken hata oluştu: {e}")
        return

    # Çıktı dosya yolunu metin dosyasına yaz

if __name__ == "__main__":
    main()
