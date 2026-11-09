import sys, os, json, re, time
from pathlib import Path

import cv2
import numpy as np
import mss
from PIL import Image
import pytesseract
from scanner.capture import capture_region
from scanner.ocr import recognize_text
from scanner.parser import group_words_into_lines, parse_offer_lines, offer_anchor_seen

from PySide6.QtCore import Qt, QRect, QPoint, QTimer, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QMessageBox, QDialog, QLineEdit,
    QFormLayout, QDialogButtonBox, QHeaderView, QSpinBox, QDoubleSpinBox
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TEMPLATES = ROOT / "templates"
PRICES_FILE = DATA / "prices.json"
CONFIG_FILE = DATA / "config.json"
DATA.mkdir(exist_ok=True)
TEMPLATES.mkdir(exist_ok=True)

def load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

class RegionSelector(QWidget):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.origin = QPoint()
        self.current = QPoint()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setWindowState(Qt.WindowFullScreen)
        self.setCursor(Qt.CrossCursor)
        self.setWindowOpacity(0.28)
        self.show()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 120))
        if not self.origin.isNull():
            r = QRect(self.origin, self.current).normalized()
            p.setCompositionMode(QPainter.CompositionMode_Clear)
            p.fillRect(r, Qt.transparent)
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            p.setPen(QPen(QColor(70, 220, 140), 3))
            p.drawRect(r)
        p.setPen(Qt.white)
        p.setFont(QFont("Segoe UI", 14))
        p.drawText(30, 45, "Выдели область окна обмена мышью. ESC — отмена.")

    def mousePressEvent(self, e):
        self.origin = e.position().toPoint()
        self.current = self.origin
        self.update()

    def mouseMoveEvent(self, e):
        self.current = e.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, e):
        self.current = e.position().toPoint()
        r = QRect(self.origin, self.current).normalized()
        if r.width() > 30 and r.height() > 30:
            self.callback([r.x(), r.y(), r.width(), r.height()])
        self.close()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

class TemplateCropDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбрать иконку предмета")
        self.image = image
        self.rect = None
        self.start = None
        self.end = None
        self.setMinimumSize(700, 520)
        layout = QVBoxLayout(self)
        self.preview = QLabel("Выдели прямоугольником только иконку предмета")
        self.preview.setAlignment(Qt.AlignCenter)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
        self.pix = QPixmap.fromImage(qimg)
        self.preview.setPixmap(self.pix.scaled(900, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.preview.mousePressEvent = self.press
        self.preview.mouseMoveEvent = self.move
        self.preview.mouseReleaseEvent = self.release
        layout.addWidget(self.preview)
        self.info = QLabel("Выделение: не выбрано")
        layout.addWidget(self.info)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def map_point(self, p):
        pm = self.preview.pixmap()
        if not pm:
            return QPoint()
        label_w, label_h = self.preview.width(), self.preview.height()
        xoff = (label_w - pm.width()) // 2
        yoff = (label_h - pm.height()) // 2
        x = max(0, min(pm.width(), p.x() - xoff))
        y = max(0, min(pm.height(), p.y() - yoff))
        sx = self.image.shape[1] / pm.width()
        sy = self.image.shape[0] / pm.height()
        return QPoint(int(x*sx), int(y*sy))

    def press(self, e):
        self.start = self.map_point(e.position().toPoint())
        self.end = self.start

    def move(self, e):
        if self.start is not None:
            self.end = self.map_point(e.position().toPoint())
            self.info.setText(f"Выделение: {abs(self.end.x()-self.start.x())} × {abs(self.end.y()-self.start.y())} px")

    def release(self, e):
        if self.start is not None:
            self.end = self.map_point(e.position().toPoint())
            self.rect = QRect(self.start, self.end).normalized()

    def get_crop(self):
        if not self.rect or self.rect.width() < 12 or self.rect.height() < 12:
            return None
        r = self.rect
        return self.image[r.y():r.y()+r.height(), r.x():r.x()+r.width()].copy()

class ItemDialog(QDialog):
    def __init__(self, name="", price=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Предмет и цена")
        form = QFormLayout(self)
        self.name = QLineEdit(name)
        self.price = QSpinBox()
        self.price.setRange(0, 2_000_000_000)
        self.price.setValue(int(price))
        form.addRow("Название:", self.name)
        form.addRow("Цена за 1 шт. (руб.):", self.price)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

class OCRScanThread(QThread):
    completed = Signal(object)

    def __init__(self, region, price_names, parent=None):
        super().__init__(parent)
        self.region = list(region)
        self.price_names = list(price_names)

    def run(self):
        try:
            frame = capture_region(self.region)
            words = recognize_text(frame)
            lines = group_words_into_lines(words)
            found = parse_offer_lines(lines, self.price_names)
            self.completed.emit({
                "found": found,
                "line_count": len(lines),
                "anchor_seen": offer_anchor_seen(lines)
            })
        except Exception as exc:
            self.completed.emit({"error": str(exc)})

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STALCRAFT Trade Assistant")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setMinimumWidth(390)
        self.prices = load_json(PRICES_FILE, {})
        self.config = load_json(CONFIG_FILE, {"region": None, "scan_interval": 1.2})
        self.last_results = []
        self.empty_anchor_scans = 0
        self.scan_thread = None
        self.build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.scan)
        # Poll frequently; skip a tick while the previous OCR job is running.
        self.timer.start(300)
        self.refresh_price_table()

    def build_ui(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        title = QLabel("◈ TRADE ASSISTANT")
        title.setStyleSheet("font-size:16px;font-weight:700;color:#b8f3d0;")
        self.status = QLabel("Ожидание области")
        self.status.setStyleSheet("color:#d8b66a;")
        top.addWidget(title)
        top.addStretch()
        top.addWidget(self.status)
        root.addLayout(top)

        controls = QHBoxLayout()
        self.region_btn = QPushButton("Выбрать область обмена")
        self.region_btn.clicked.connect(self.select_region)
        self.pause_btn = QPushButton("Пауза")
        self.pause_btn.clicked.connect(self.toggle_scan)
        controls.addWidget(self.region_btn)
        controls.addWidget(self.pause_btn)
        root.addLayout(controls)

        self.results = QTableWidget(0, 4)
        self.results.setHorizontalHeaderLabels(["Предмет", "Кол-во", "Цена/шт.", "Сумма"])
        self.results.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1,4):
            self.results.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.results.setMaximumHeight(190)
        root.addWidget(self.results)

        totalrow = QHBoxLayout()
        totalrow.addWidget(QLabel("ИТОГО:"))
        self.total = QLabel("0 ₽")
        self.total.setStyleSheet("font-size:23px;font-weight:800;color:#59d995;")
        totalrow.addStretch()
        totalrow.addWidget(self.total)
        root.addLayout(totalrow)

        self.copy_btn = QPushButton("⧉  СКОПИРОВАТЬ СУММУ")
        self.copy_btn.setMinimumHeight(42)
        self.copy_btn.setStyleSheet("background:#237b51;color:white;font-weight:700;padding:8px;border-radius:5px;")
        self.copy_btn.clicked.connect(self.copy_total)
        root.addWidget(self.copy_btn)

        price_head = QHBoxLayout()
        price_head.addWidget(QLabel("Мой прайс-лист"))
        price_head.addStretch()
        add_btn = QPushButton("+ Предмет")
        add_btn.clicked.connect(self.add_item)
        edit_btn = QPushButton("Изменить")
        edit_btn.clicked.connect(self.edit_item)
        del_btn = QPushButton("Удалить")
        del_btn.clicked.connect(self.delete_item)
        price_head.addWidget(add_btn)
        price_head.addWidget(edit_btn)
        price_head.addWidget(del_btn)
        root.addLayout(price_head)
        self.price_table = QTableWidget(0, 2)
        self.price_table.setHorizontalHeaderLabels(["Предмет", "Цена за шт."])
        self.price_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.price_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.price_table.setMaximumHeight(150)
        root.addWidget(self.price_table)

        self.hint = QLabel("Добавь предметы и цены в прайс-лист. OCR считывает нижнюю часть окна обмена.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#999;font-size:10px;")
        root.addWidget(self.hint)
        self.setStyleSheet("""
            QWidget { background:#202328; color:#eeeeee; font-family:'Segoe UI'; }
            QTableWidget { background:#17191d; gridline-color:#41454b; }
            QPushButton { background:#343941; border:1px solid #555b64; padding:6px; border-radius:4px; }
            QPushButton:hover { background:#454d57; }
            QHeaderView::section { background:#30353c; padding:5px; border:0; }
        """)

    def refresh_price_table(self):
        self.price_table.setRowCount(0)
        for name, price in sorted(self.prices.items()):
            r = self.price_table.rowCount()
            self.price_table.insertRow(r)
            self.price_table.setItem(r,0,QTableWidgetItem(name))
            self.price_table.setItem(r,1,QTableWidgetItem(f"{int(price):,}".replace(","," ")))

    def add_item(self):
        d = ItemDialog(parent=self)
        d.setWindowFlags(d.windowFlags() | Qt.WindowStaysOnTopHint)
        d.setWindowModality(Qt.ApplicationModal)
        d.adjustSize()

        x = self.x() + (self.width() - d.width()) // 2
        y = self.y() + (self.height() - d.height()) // 2
        d.move(x, y)

        d.raise_()
        d.activateWindow()
        result = d.exec()

        self.raise_()
        self.activateWindow()

        if result == QDialog.Accepted and d.name.text().strip():
            self.prices[d.name.text().strip()] = d.price.value()
            save_json(PRICES_FILE, self.prices)
            self.refresh_price_table()

    def edit_item(self):
        r = self.price_table.currentRow()
        if r < 0: return
        name = self.price_table.item(r,0).text()
        d = ItemDialog(name, self.prices.get(name,0), self)
        if d.exec() == QDialog.Accepted and d.name.text().strip():
            self.prices.pop(name, None)
            self.prices[d.name.text().strip()] = d.price.value()
            save_json(PRICES_FILE, self.prices)
            self.refresh_price_table()

    def delete_item(self):
        r = self.price_table.currentRow()
        if r < 0: return
        name = self.price_table.item(r,0).text()
        self.prices.pop(name, None)
        save_json(PRICES_FILE, self.prices)
        self.refresh_price_table()

    def select_region(self):
        self.hide()
        QTimer.singleShot(350, self.show_region_selector)

    def show_region_selector(self):
        self.region_selector = RegionSelector(self.set_region)

    def set_region(self, region):
        self.config["region"] = region
        save_json(CONFIG_FILE, self.config)
        self.show()
        self.status.setText("Мониторинг обмена")
        self.scan()

    def toggle_scan(self):
        if self.timer.isActive():
            self.timer.stop()
            self.pause_btn.setText("Продолжить")
            self.status.setText("Пауза")
        else:
            self.timer.start(300)
            self.pause_btn.setText("Пауза")
            self.status.setText("Мониторинг обмена")

    def add_template(self):
        if not self.config.get("region"):
            QMessageBox.information(self, "Сначала выбери область", "Нажми «Выбрать область обмена».")
            return
        self.hide()
        QTimer.singleShot(350, self.capture_template)

    def capture_template(self):
        try:
            x,y,w,h = self.config["region"]
            with mss.mss() as sct:
                shot = np.array(sct.grab({"left":x,"top":y,"width":w,"height":h}))
            image = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
            dlg = TemplateCropDialog(image, self)
            if dlg.exec() == QDialog.Accepted:
                crop = dlg.get_crop()
                if crop is not None:
                    itemdlg = ItemDialog(parent=self)
                    if itemdlg.exec() == QDialog.Accepted and itemdlg.name.text().strip():
                        name = itemdlg.name.text().strip()
                        self.prices[name] = itemdlg.price.value()
                        save_json(PRICES_FILE, self.prices)
                        cv2.imwrite(str(TEMPLATES / (re.sub(r'[^a-zA-Zа-яА-Я0-9_-]','_',name)+".png")), crop)
                        self.refresh_price_table()
            self.show()
        except Exception as e:
            self.show()
            QMessageBox.warning(self, "Ошибка", str(e))

    def scan(self):
        region = self.config.get("region")
        if not region:
            return

        # Never queue OCR jobs: keep at most one scan active at a time.
        if self.scan_thread is not None and self.scan_thread.isRunning():
            return

        self.scan_thread = OCRScanThread(region, list(self.prices.keys()), self)
        self.scan_thread.completed.connect(self.apply_scan_result)
        self.scan_thread.finished.connect(self.on_scan_finished)
        self.scan_thread.finished.connect(self.scan_thread.deleteLater)
        self.scan_thread.start()

    def on_scan_finished(self):
        # Clear the Python reference before the finished QThread is deleted.
        self.scan_thread = None

    def _render_results(self, found):
        self.last_results = []
        self.results.setRowCount(0)
        grand = 0
        for name, qty in found.items():
            price = int(self.prices.get(name, 0))
            amount = qty * price
            grand += amount
            self.last_results.append((name, qty, price))
            row = self.results.rowCount()
            self.results.insertRow(row)
            values = [name, str(qty), f"{price:,}".replace(",", " "),
                      f"{amount:,}".replace(",", " ")]
            for col, value in enumerate(values):
                self.results.setItem(row, col, QTableWidgetItem(value))
        self.total.setText(f"{grand:,}".replace(",", " ") + " ₽")

    def apply_scan_result(self, result):
        if "error" in result:
            self.hint.setText(
                f"Временная ошибка OCR (предыдущий результат сохранён): {result['error']}"
            )
            return

        found = result.get("found", {})
        anchor_seen = result.get("anchor_seen", False)
        if found:
            self.empty_anchor_scans = 0
            self._render_results(found)
            self.status.setText(f"Найдено предметов: {len(found)}")
            unpriced = [name for name, qty, price in self.last_results if price == 0]
            if unpriced:
                self.hint.setText(
                    "Без цены в прайс-листе: " + ", ".join(unpriced) +
                    ". Добавь цену, если хочешь включить их в итог."
                )
            else:
                self.hint.setText(
                    "Распознаны предметы в нижней части обмена. Проверь результат перед сделкой."
                )
            return

        if not anchor_seen:
            self.status.setText("Ожидание строки «Вы получите»")
            self.hint.setText(
                "Последний подтверждённый результат сохранён: строка «Вы получите» "
                "пока не распознана."
            )
            return

        self.empty_anchor_scans += 1
        if self.empty_anchor_scans >= 3:
            self._render_results({})
            self.status.setText("Предложение пустое")
            self.hint.setText(
                "Строка «Вы получите» распознана в трёх последовательных кадрах, "
                "но предметы с количеством xN не найдены."
            )
        else:
            self.status.setText("Проверка предложения…")
            self.hint.setText(
                f"Предметы пока не распознаны ({self.empty_anchor_scans}/3). "
                "Предыдущая сумма сохранена до подтверждения пустого предложения."
            )

    def copy_total(self):
        QApplication.clipboard().setText(str(sum(q*p for _,q,p in self.last_results)))
        self.status.setText("Сумма скопирована")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
