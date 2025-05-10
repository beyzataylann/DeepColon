import sys
import os
import subprocess
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QTextEdit, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QMessageBox, QProgressBar
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class ImageUploader(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧬DeepColon")
        self.setGeometry(100, 100, 600, 700)
        self.filename = None
        self.image_path = None
        self.results_history = []  # geçmiş sonuçlar listesi
        self.init_ui()

    def init_ui(self):
        self.image_label = QLabel("Henüz görsel yüklenmedi", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedHeight(300)  # Sabit yükseklik yeterli
        self.image_label.setScaledContents(True)  # Görsel düzgün yerleşsin

        self.upload_button = QPushButton("Görsel Yükle")
        self.test_button = QPushButton("Test Et")
        self.save_button = QPushButton("Excel'e Kaydet")

        self.upload_button.setFixedWidth(150)
        self.test_button.setFixedWidth(150)
        self.save_button.setFixedWidth(150)

        self.upload_button.clicked.connect(self.upload_image)
        self.test_button.clicked.connect(self.test_function)
        self.save_button.clicked.connect(self.save_to_excel)

        self.result_box = QTextEdit(self)
        self.result_box.setReadOnly(True)
        self.result_box.setFixedHeight(200)

        self.table = QTableWidget(self)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Görsel", "Tahmin"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(200)
        self.table.cellClicked.connect(self.on_table_click)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.test_button)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.result_box)
        main_layout.addWidget(self.table)

        self.setLayout(main_layout)



    def upload_image(self):
        options = QFileDialog.Options()
        file_filter = "Image Files (*.png *.jpg *.jpeg *.bmp)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Görsel Seç", "", file_filter, options=options)

        if file_path:
            self.image_path = file_path
            self.filename = os.path.basename(file_path)
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
        else:
            print("Görsel seçilmedi!")

    def test_function(self):
        self.result_box.clear()
        self.progress_bar.setValue(0)
        self.result_box.append("Test Etme İşlemi Başladı...")

        try:
            if self.image_path:
               

                self.filename = os.path.basename(self.image_path)

                subprocess.run(["python", "clahe_ortfil.py", self.image_path])
                self.progress_bar.setValue(15)

                subprocess.run(["python", "unet.py"])
                self.progress_bar.setValue(30)

                subprocess.run(["python", "lbp_features.py", self.filename])
                self.progress_bar.setValue(45)

                subprocess.run(["python", "googlenet_features.py", self.filename])
                self.progress_bar.setValue(60)

                subprocess.run(["python", "resnet_features.py", self.filename])
                self.progress_bar.setValue(75)

                subprocess.run(["python", "combined_features.py", self.filename])
                self.progress_bar.setValue(90)
                


                self.result_box.append("🧪 Model ile Test Ediliyor...")
                result = subprocess.run(["python", "test.py", self.filename], capture_output=True, text=True)
                prediction_raw = result.stdout.strip()

                if "colon_n" in prediction_raw:
                    prediction = "Sağlıklı"
                elif "colon_aca" in prediction_raw:
                    prediction = "Kanserli"
                else:
                    prediction = prediction_raw

                self.result_box.append("✔ Test Tamamlandı.\n")
                self.result_box.append("📋 Tahmin Sonucu:\n")
                self.result_box.append(prediction)

                self.results_history.append([self.image_path, prediction])
                self.add_result_to_table(self.image_path, prediction)
                self.progress_bar.setValue(100)
            else:
                self.result_box.append("⚠ Görsel yüklenmedi!")
        except Exception as e:
            self.result_box.append(f"❌ İşlem sırasında hata oluştu: {e}")
            self.progress_bar.setValue(0)

    def add_result_to_table(self, image_path, prediction):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
        self.table.setItem(row_position, 0, QTableWidgetItem(image_path))
        self.table.setItem(row_position, 1, QTableWidgetItem(prediction))

    def on_table_click(self, row, column):
        image_path_item = self.table.item(row, 0)
        prediction_item = self.table.item(row, 1)

        if image_path_item and prediction_item:
            image_path = image_path_item.text()
            prediction = prediction_item.text()

            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
            self.result_box.setText(prediction)

    def save_to_excel(self):
        if not self.results_history:
            QMessageBox.warning(self, "Uyarı", "Henüz kaydedilecek veri yok!")
            return

        try:
            # Debugging: `results_history`'yi kontrol et
            print(f"Saving results: {self.results_history}")
            
            df = pd.DataFrame(self.results_history, columns=["Görsel Yolu", "Tahmin"])

            save_path = "sonuclar/tahminler.xlsx"
            
            # Dosya yolu kontrolü: Tam bir yol kullanarak kaydetmeyi deneyelim
            if not os.path.isdir("sonuclar"):
                os.makedirs("sonuclar", exist_ok=True)
            
            # Tam dosya yolu kullanmak faydalı olabilir
            #save_path = os.path.join(os.getcwd(), "sonuclar", "tahminler.xlsx")

            print(f"Kaydedilecek dosya yolu: {save_path}")

            # Verileri Excel dosyasına kaydet
            df.to_excel(save_path, index=False)

            QMessageBox.information(self, "Başarılı", f"Excel dosyası başarıyla kaydedildi:\n{save_path}")
        except Exception as e:
            # Hata mesajını daha ayrıntılı gösterelim
            print(f"Hata: {e}")
            QMessageBox.critical(self, "Hata", f"Excel'e kaydederken hata oluştu:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageUploader()
    window.show()
    sys.exit(app.exec_())
