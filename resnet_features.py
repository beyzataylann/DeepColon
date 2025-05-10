import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
from torchvision import models
from torchvision.models import ResNet18_Weights
import argparse

def main():
    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Argüman ayarları
    parser = argparse.ArgumentParser(description="Belirli bir görsel için ResNet18 ile özellik çıkarımı.")
    parser.add_argument('filename', type=str, help="Girdi görselinin adı (ör: colonn5000.jpeg)")
    args = parser.parse_args()

    # Dosya adı işlemleri
    filename = args.filename
    basename = os.path.splitext(filename)[0]
    pre_dir = "preprocessing"
    pre_output_path = os.path.join(pre_dir, f"{basename}.jpeg")

    # Görüntü dönüşümü
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ResNet18 modeli yükle (FC katmanı hariç)
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # FC katmanı çıkarıldı
    feature_extractor.to(device)
    feature_extractor.eval()

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
        feature_np = features.cpu().numpy().reshape(-1)  # 512 özellik

    # DataFrame oluştur
    feature_columns = [f'feature_{i}' for i in range(feature_np.shape[0])]
    df = pd.DataFrame([feature_np], columns=feature_columns)
    df.insert(0, 'image_name', filename)

    # CSV dosyasına kaydet
    output_dir = "resnet"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"{basename}_resnet_features.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    df.to_csv(csv_path, index=False)
    print(f"Özellik çıkarımı tamamlandı ve CSV kaydedildi: {csv_path}")

if __name__ == "__main__":
    main()
