import cv2
import os
import argparse

def main():

    parser = argparse.ArgumentParser(description="CLAHE ve ortalama filtreleme uygulaması")
    parser.add_argument('image_path', type=str, help="")
    args = parser.parse_args()

    image_path = args.image_path
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Çıktı dizini
    output_dir = "preprocessing"
    os.makedirs(output_dir, exist_ok=True)

    # Görüntüyü oku
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")

    # LAB renk uzayına çevir
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # CLAHE uygula
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)

    # LAB görüntüsünü yeniden oluştur ve BGR’a dönüştür
    lab_clahe = cv2.merge([L_clahe, A, B])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Ortalama filtre uygula
    blurred = cv2.blur(enhanced, (3, 3))

    # Görüntüyü kaydet
    output_path = os.path.join(output_dir, f"{basename}.jpeg")
    cv2.imwrite(output_path, blurred)

    print(f"Ön işlenmiş görsel kaydedildi: {output_path}")

if __name__ == "__main__":
    main()
