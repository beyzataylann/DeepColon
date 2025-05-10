import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from skimage.feature import local_binary_pattern
import csv
from unet import UNet
import argparse


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Görüntü yolu dosyadan okunur
    parser = argparse.ArgumentParser(description="Belirli bir görsel için özellik dosyalarını birleştir.")
    parser.add_argument('filename', type=str, help="Girdi görselinin adı (ör: colonn5000.jpeg)")
    args = parser.parse_args()

    # Dosya adı işlemleri
    filename = args.filename
    basename = os.path.splitext(filename)[0]
    pre_dir = "preprocessing"

    

    # UNet modeli yükleniyor
    try:
        model = UNet(in_channels=3, num_classes=1)
        model.load_state_dict(torch.load('unet3.pth', map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"UNet modeli yüklenirken hata oluştu: {e}")
        return

    # LBP parametreleri
    radius = 2
    n_points = 8 * radius
    n_bins = n_points + 2

    # Görüntüyü oku
    pre_output_path= os.path.join(pre_dir, f"{basename}.jpeg")

    image = cv2.imread(pre_output_path)
    if image is None:
        print(f"Hata: Görüntü okunamadı: {pre_output_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (256, 256))

    # Görüntüyü tensöre dönüştür
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    input_tensor = transform(image_resized).unsqueeze(0).to(device)

    # Segmentasyon maskesi tahmini
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        predicted_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Griye çevir ve maske uygula
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    masked_image = cv2.bitwise_and(gray_image, gray_image, mask=predicted_mask)

    # LBP hesapla
    lbp_result = local_binary_pattern(masked_image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp_result.ravel(), bins=np.arange(0, n_bins + 1), density=True)

    # Çıktı klasörü
    output_dir = "lbp_result"
    os.makedirs(output_dir, exist_ok=True)

    # Görselleri kaydet
    lbp_save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_lbp.jpeg")
    mask_save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_mask.jpeg")
    cv2.imwrite(lbp_save_path, np.uint8(255 * lbp_result / lbp_result.max()))
    cv2.imwrite(mask_save_path, predicted_mask * 255)

    # Histogramı CSV olarak kaydet
    csv_filename = f"{os.path.splitext(filename)[0]}_lbp_features.csv"
    l_csv_path = os.path.join(output_dir, csv_filename)

    with open(l_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [f'bin_{i}' for i in range(n_bins)]
        writer.writerow(['image_name'] + header)
        writer.writerow([filename] + hist.tolist())


    
    print(f"LBP görseli kaydedildi: {lbp_save_path}")
    print(f"Maske kaydedildi: {mask_save_path}")
    print(f"Lbp özellikleri kaydedildi: {l_csv_path}")

if __name__ == "__main__":
    main()
