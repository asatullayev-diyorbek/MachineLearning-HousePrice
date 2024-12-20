import sys
import joblib
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QLabel, QPushButton
from PyQt5.QtGui import QFont

class HousePricePredictor(QWidget):
    def __init__(self):
        super().__init__()

        self.model = joblib.load('house_price_model.pkl')
        self.scaler = joblib.load('scaler_custom.pkl')

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Uy Narxini Bashorat qilish')

        # Fontlarni o'rnatish
        font = QFont('Arial', 12)

        layout = QVBoxLayout()

        # Maydon uchun label va input
        self.maydon_label = QLabel('Uy maydoni (m²):', self)
        self.maydon_label.setFont(font)
        self.maydon_input = QLineEdit(self)
        self.maydon_input.setPlaceholderText('Masalan, 1000')
        self.maydon_input.setFont(font)
        self.maydon_input.setStyleSheet("padding: 10px;")
        layout.addWidget(self.maydon_label)
        layout.addWidget(self.maydon_input)

        # Xonalar uchun label va input
        self.xonalar_label = QLabel('Xonalar soni:', self)
        self.xonalar_label.setFont(font)
        self.xonalar_input = QLineEdit(self)
        self.xonalar_input.setPlaceholderText('Masalan, 3')
        self.xonalar_input.setFont(font)
        self.xonalar_input.setStyleSheet("padding: 10px;")
        layout.addWidget(self.xonalar_label)
        layout.addWidget(self.xonalar_input)

        # Joylashuv uchun label va input
        self.joylashuv_label = QLabel('Joylashuv (markaz, shahar_cheti, qishloq):', self)
        self.joylashuv_label.setFont(font)
        self.joylashuv_input = QLineEdit(self)
        self.joylashuv_input.setPlaceholderText('Masalan, qishloq')
        self.joylashuv_input.setFont(font)
        self.joylashuv_input.setStyleSheet("padding: 10px;")
        layout.addWidget(self.joylashuv_label)
        layout.addWidget(self.joylashuv_input)

        # Yoshi uchun label va input
        self.yoshi_label = QLabel('Uy yoshi:', self)
        self.yoshi_label.setFont(font)
        self.yoshi_input = QLineEdit(self)
        self.yoshi_input.setPlaceholderText('Masalan, 5')
        self.yoshi_input.setFont(font)
        self.yoshi_input.setStyleSheet("padding: 10px;")
        layout.addWidget(self.yoshi_label)
        layout.addWidget(self.yoshi_input)

        # Bashorat qilish tugmasi
        self.bashorat_button = QPushButton('Narxni Bashorat Qilish', self)
        self.bashorat_button.setFont(QFont('Arial', 14))
        self.bashorat_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.bashorat_button.clicked.connect(self.predict_price)
        layout.addWidget(self.bashorat_button)

        # Natija uchun label
        self.result_label = QLabel('Bashorat qilingan narx: ', self)
        self.result_label.setFont(QFont('Arial', 14))
        self.result_label.setStyleSheet("padding: 10px;")
        layout.addWidget(self.result_label)

        self.setLayout(layout)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                padding: 20px;
            }
        """)

        self.resize(400, 400)

    def predict_price(self):
        # Foydalanuvchidan ma'lumotlarni olish
        try:
            maydon = float(self.maydon_input.text())
            xonalar = int(self.xonalar_input.text())
            joylashuv = self.joylashuv_input.text()
            yoshi = int(self.yoshi_input.text())

            # Joylashuvni OneHotEncoding qilish
            joylashuv_encoded = [0, 0, 0]
            if joylashuv == 'markaz':
                joylashuv_encoded[0] = 1
            elif joylashuv == 'shahar_cheti':
                joylashuv_encoded[1] = 1
            elif joylashuv == 'qishloq':
                joylashuv_encoded[2] = 1
            else:
                self.result_label.setText("Noto'g'ri joylashuv kiritildi!")
                return

            # Ma'lumotlarni DataFrame formatiga o'zgartirish
            new_data = pd.DataFrame([[maydon, xonalar, yoshi] + joylashuv_encoded],
                                    columns=['maydon', 'xonalar', 'yoshi', 'joylashuv_markaz', 'joylashuv_qishloq', 'joylashuv_shahar_cheti'])

            # Yangi ma'lumotlarni standartizatsiya qilish (o'qitilgan scaler yordamida)
            new_data_scaled = self.scaler.transform(new_data)

            # Narxni bashorat qilish
            predicted_price = self.model.predict(new_data_scaled)

            # Natijani ekranda ko'rsatish
            self.result_label.setText(f"Bashorat qilingan uy narxi: ${predicted_price[0]:.2f}")

        except Exception as e:
            self.result_label.setText(f"Xato: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = HousePricePredictor()
    ex.show()
    sys.exit(app.exec_())
