# MNIST_PyQt6_Platform.py is a PyQt6 platform for handwritten number recognition using a convolutional neural network (CNN).
import sys, os
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, \
    QMessageBox
from PyQt6.QtGui import QPainter, QPixmap, QPen, QImage
from PyQt6.QtCore import Qt, QPoint
import torch
import torch.nn.functional as F
from model import SimpleCNN

CANVAS_SIZE = 280


class DrawBoard(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(CANVAS_SIZE, CANVAS_SIZE)
        self.pix = QPixmap(self.size())
        self.pix.fill(Qt.GlobalColor.black)
        self.last_point = QPoint()
        self.pen_width = 20

    def clear(self):
        # print("clearing...")
        self.pix.fill(Qt.GlobalColor.black)
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.last_point = e.position().toPoint()
            self.draw_line(self.last_point, self.last_point)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            new_point = e.position().toPoint()
            self.draw_line(self.last_point, new_point)
            self.last_point = new_point

    def draw_line(self, p1, p2):
        painter = QPainter(self.pix)
        pen = QPen()
        pen.setWidth(self.pen_width)
        pen.setColor(Qt.GlobalColor.white)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        painter.end()
        self.update()

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pix)
        painter.end()

    def get_image_28(self):
        qimg = self.pix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        width, height = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.sizeInBytes())
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        # 灰度
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        img = np.array(np.array(gray, dtype=np.float32) / 255.0)
        # resize to 28x28
        from PIL import Image
        img = Image.fromarray(img).resize((28, 28))
        arr28 = np.array(img, dtype=np.float32)
        return arr28


class MainWindow(QMainWindow):
    def __init__(self, model_path='mnist_cnn.pth'):
        super().__init__()
        self.setWindowTitle("手写数字识别")
        self.model = SimpleCNN()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists(model_path):
            QMessageBox.warning(None, "错误", f"{model_path} 不存在，请先训练模型")
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()

        self.board = DrawBoard()
        left.addWidget(self.board)

        btn_layout = QHBoxLayout()
        clear_btn = QPushButton("清除")
        clear_btn.clicked.connect(self.clear_all)
        pred_btn = QPushButton("识别")
        pred_btn.clicked.connect(self.predict)
        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(pred_btn)
        left.addLayout(btn_layout)

        self.result_label = QLabel("结果: -")
        self.result_label.setStyleSheet("font-size: 24px;")
        right.addWidget(self.result_label)

        self.prob_labels = []
        for i in range(10):
            l = QLabel(f"{i}: -")
            right.addWidget(l)
            self.prob_labels.append(l)

        layout.addLayout(left)
        layout.addLayout(right)
        central.setLayout(layout)

    def clear_all(self):
        # 1. 清空画板
        self.board.clear()
        # 2. 清零右侧概率标签
        self.reset_prob_labels()

    def reset_prob_labels(self):
        for i, l in enumerate(self.prob_labels):
            l.setText(f"{i}: -")
        # 同时把结果标签也清空
        self.result_label.setText("结果: -")

    def predict(self):
        try:
            img28 = self.board.get_image_28()
            # 画板空白时，不进行识别
            if np.max(img28) == 0:
                QMessageBox.information(self, "提示", "请在画板上输入数字再进行识别！")
                return
            tensor = torch.from_numpy(img28).unsqueeze(0).unsqueeze(0).to(self.device)
            tensor = (tensor - 0.1307) / 0.3081
            with torch.no_grad():
                out = self.model(tensor)
                prob = F.softmax(out, dim=1).cpu().numpy()[0]
                pred = int(prob.argmax())
            self.result_label.setText(f"结果: {pred}")
            for i in range(10):
                self.prob_labels[i].setText(f"{i}: {prob[i] * 100:.2f}%")
        except Exception as e:
            print("Predict error:", e, flush=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MNIST_PyQt6_Platform = MainWindow()

    MNIST_PyQt6_Platform.show()
    sys.exit(app.exec())
