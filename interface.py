import main as backend 
import os, sys, cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class ComparisonWorker(QThread):
    finished = pyqtSignal(np.ndarray, np.ndarray, float)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    
    def __init__(self, path_a, path_b, mode, idx_a, idx_b, device, show_debug):
        super().__init__()
        self.params = (path_a, path_b, mode, idx_a, idx_b, device, show_debug)
    
    def run(self):
        try:
            res1, res2, success_rate = backend.process_comparison(*self.params, progress_callback=self.progress.emit)
            self.finished.emit(res1, res2, success_rate)
        except Exception as e:
            self.error.emit(str(e))

class FaceAnalysisWorker(QThread):
    finished = pyqtSignal(str, list)
    
    def __init__(self, path, img_key, device):
        super().__init__()
        self.path, self.img_key, self.device = path, img_key, device
    
    def run(self):
        try:
            faces = backend.get_faces_info(self.path, self.device)
            self.finished.emit(self.img_key, faces)
        except: pass

class FaceSwapApp(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceSwap Application")
        self.setGeometry(100, 100, 1200, 750)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #fff; }
            QWidget { font-family: 'Segoe UI', Arial, sans-serif; }
            QLabel { color: #1a1a1a; }
            QGroupBox { 
                border: 1px solid #e0e0e0; border-radius: 8px; margin-top: 12px; 
                padding-top: 12px; font-weight: 500; color: #2a2a2a; background-color: #fafafa; 
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 8px; }
        """)
        
        if os.path.exists("icon/icon.png"):
            self.setWindowIcon(QIcon("icon/icon.png"))
        
        self.path_a = self.path_b = self.res1 = self.res2 = None
        self.info_a = self.info_b = []
        
        self._init_ui()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("FaceSwap Application")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Segoe UI', 18, QFont.Light))
        title.setStyleSheet("color: #1a1a1a; padding: 12px; background-color: transparent;")
        layout.addWidget(title)

        self.pbar = QProgressBar()
        self.pbar.setStyleSheet("""
            QProgressBar { 
                border: 1px solid #e0e0e0; border-radius: 4px; text-align: center; 
                height: 5px; background-color: #f5f5f5; 
            }
            QProgressBar::chunk { background-color: #1a1a1a; border-radius: 3px; }
        """)
        self.pbar.setTextVisible(False)
        self.pbar.setVisible(False)
        layout.addWidget(self.pbar)

        images_container = QFrame()
        images_container.setStyleSheet("background-color: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px;")
        images_layout = QHBoxLayout(images_container)
        images_layout.setSpacing(16)

        input_section = QVBoxLayout()
        input_label = QLabel("Input Images")
        input_label.setFont(QFont('Segoe UI', 9, QFont.Medium))
        input_label.setAlignment(Qt.AlignCenter)
        input_label.setStyleSheet("color: #2a2a2a; background: transparent; border: none;")
        input_section.addWidget(input_label)
        
        input_images = QHBoxLayout()
        input_images.setSpacing(12)
        self.lbl_img_a = self._create_img_label("Image A")
        self.lbl_img_b = self._create_img_label("Image B")
        input_images.addWidget(self.lbl_img_a)
        input_images.addWidget(self.lbl_img_b)
        input_section.addLayout(input_images)
        images_layout.addLayout(input_section)

        separator = QLabel("→")
        separator.setStyleSheet("font-size: 32px; color: #666; background: transparent; border: none;")
        separator.setAlignment(Qt.AlignCenter)
        images_layout.addWidget(separator)

        output_section = QVBoxLayout()
        output_label = QLabel("Results")
        output_label.setFont(QFont('Segoe UI', 9, QFont.Medium))
        output_label.setAlignment(Qt.AlignCenter)
        output_label.setStyleSheet("color: #2a2a2a; background: transparent; border: none;")
        output_section.addWidget(output_label)
        
        output_images = QHBoxLayout()
        output_images.setSpacing(12)
        self.lbl_res1 = self._create_img_label("Face A → Body B")
        self.lbl_res2 = self._create_img_label("Face B → Body A")
        output_images.addWidget(self.lbl_res1)
        output_images.addWidget(self.lbl_res2)
        output_section.addLayout(output_images)
        images_layout.addLayout(output_section)
        layout.addWidget(images_container)

        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("QFrame { background-color: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; }")
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setSpacing(10)

        row1 = QHBoxLayout()
        row1.setSpacing(10)
        
        btn_img_a = QPushButton("Select Image A")
        btn_img_a.setStyleSheet(self._btn_style())
        btn_img_a.clicked.connect(self.select_image_a)
        
        btn_img_b = QPushButton("Select Image B")
        btn_img_b.setStyleSheet(self._btn_style())
        btn_img_b.clicked.connect(self.select_image_b)
        
        self.chk_debug = QCheckBox("Show Mesh Analysis")
        self.chk_debug.setStyleSheet("""
            QCheckBox { color: #4a4a4a; font-weight: 500; spacing: 8px; }
            QCheckBox::indicator { 
                width: 18px; height: 18px; border: 2px solid #d0d0d0; 
                border-radius: 3px; background-color: white; 
            }
            QCheckBox::indicator:checked { background-color: #1a1a1a; border-color: #1a1a1a; }
        """)
        
        row1.addWidget(btn_img_a)
        row1.addWidget(btn_img_b)
        row1.addStretch()
        row1.addWidget(self.chk_debug)
        ctrl_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)
        
        lbl_mode = QLabel("Engine:")
        lbl_mode.setStyleSheet("color: #4a4a4a; font-weight: 500;")
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Deep Learning", "Geometric"])
        self.combo_mode.setStyleSheet(self._combo_style())
        
        lbl_device = QLabel("Device:")
        lbl_device.setStyleSheet("color: #4a4a4a; font-weight: 500;")
        self.combo_device = QComboBox()
        for dev in backend.get_available_devices():
            self.combo_device.addItem(dev['name'], dev['id'])
        self.combo_device.setStyleSheet(self._combo_style())
        
        row2.addWidget(lbl_mode)
        row2.addWidget(self.combo_mode, 1)
        row2.addWidget(lbl_device)
        row2.addWidget(self.combo_device, 1)
        ctrl_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(10)
        
        lbl_face_a = QLabel("Face A:")
        lbl_face_a.setStyleSheet("color: #4a4a4a; font-weight: 500;")
        self.combo_face_a = QComboBox()
        self.combo_face_a.addItem("Detecting...", -1)
        self.combo_face_a.setStyleSheet(self._combo_style())
        
        lbl_face_b = QLabel("Face B:")
        lbl_face_b.setStyleSheet("color: #4a4a4a; font-weight: 500;")
        self.combo_face_b = QComboBox()
        self.combo_face_b.addItem("Detecting...", -1)
        self.combo_face_b.setStyleSheet(self._combo_style())
        
        row3.addWidget(lbl_face_a)
        row3.addWidget(self.combo_face_a, 1)
        row3.addWidget(lbl_face_b)
        row3.addWidget(self.combo_face_b, 1)
        ctrl_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.setSpacing(10)
        
        btn_run = QPushButton("Process")
        btn_run.setStyleSheet(self._btn_style(primary=True))
        btn_run.clicked.connect(self.run_process)
        
        btn_save = QPushButton("Save Results")
        btn_save.setStyleSheet(self._btn_style())
        btn_save.clicked.connect(self.save_results)
        
        row4.addWidget(btn_run)
        row4.addWidget(btn_save)
        ctrl_layout.addLayout(row4)
        
        layout.addWidget(ctrl_frame)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #888; font-size: 11px; padding: 8px;")
        layout.addWidget(self.lbl_status)

    def _create_img_label(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("""
            border: 2px dashed #d0d0d0; background: #fff; color: #999; 
            font-size: 11px; border-radius: 6px; font-weight: 500;
        """)
        lbl.setFixedSize(420, 420)
        return lbl

    def _btn_style(self, primary=False):
        if primary:
            return """
                QPushButton { 
                    background-color: #1a1a1a; color: white; padding: 8px 20px; border: none; 
                    border-radius: 6px; font-weight: 500; font-size: 12px; min-width: 100px; 
                }
                QPushButton:hover { background-color: #2a2a2a; }
                QPushButton:pressed { background-color: #0a0a0a; }
            """
        return """
            QPushButton { 
                background-color: white; color: #1a1a1a; padding: 8px 16px; border: 1px solid #d0d0d0; 
                border-radius: 6px; font-weight: 500; font-size: 12px; 
            }
            QPushButton:hover { background-color: #f5f5f5; border-color: #b0b0b0; }
            QPushButton:pressed { background-color: #e8e8e8; }
        """

    def _combo_style(self):
        return """
            QComboBox { 
                padding: 6px 10px; border: 1px solid #d0d0d0; border-radius: 6px; 
                background-color: white; color: #1a1a1a; font-size: 11px; min-height: 18px; 
            }
            QComboBox:hover { border-color: #b0b0b0; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow { 
                image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; 
                border-top: 5px solid #666; margin-right: 8px; 
            }
            QComboBox QAbstractItemView { 
                border: 1px solid #d0d0d0; background-color: white; 
                selection-background-color: #f0f0f0; selection-color: #1a1a1a; outline: none; 
            }
        """

    def select_image_a(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image A", "", "Images (*.jpg *.png *.jpeg)")
        if path:
            self.path_a = path
            self.show_img(path, self.lbl_img_a)
            self.start_analysis(path, "image_a")

    def select_image_b(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image B", "", "Images (*.jpg *.png *.jpeg)")
        if path:
            self.path_b = path
            self.show_img(path, self.lbl_img_b)
            self.start_analysis(path, "image_b")

    def show_img(self, path, label, face_idx=None):
        img = cv2.imread(path)
        if img is None:
            label.setPixmap(QPixmap(path).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            return
        
        if face_idx is not None:
            faces = self.info_a if label == self.lbl_img_a else self.info_b
            face = next((f for f in faces if f['index'] == face_idx), None)
            if face:
                b = face['bbox']
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 3)
                cv2.putText(img, f"Selected #{face_idx+1}", (int(b[0]), int(b[1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        label.setPixmap(QPixmap.fromImage(QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def display_result(self, cv_img, label):
        if cv_img is None: return
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        label.setPixmap(QPixmap.fromImage(QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_analysis(self, path, img_key):
        self.lbl_status.setText("Analyzing faces...")
        worker = FaceAnalysisWorker(path, img_key, self.combo_device.currentData())
        worker.finished.connect(self.on_analyzed)
        worker.start()
        self.worker_temp = worker

    def on_analyzed(self, img_key, faces):
        combo = self.combo_face_a if img_key == "image_a" else self.combo_face_b
        combo.clear()
        
        if img_key == "image_a": 
            self.info_a = faces
        else: 
            self.info_b = faces
        
        if not faces: 
            combo.addItem("No face detected", -1)
        else:
            for f in faces: 
                combo.addItem(f"Face #{f['index']+1} ({f['confidence']:.0%})", f['index'])
            self.on_face_selection_changed(img_key)
            combo.currentIndexChanged.connect(lambda: self.on_face_selection_changed(img_key))
            
        self.lbl_status.setText(f"{'Image A' if img_key == 'image_a' else 'Image B'} analysis completed")
    
    def on_face_selection_changed(self, img_key):
        if img_key == "image_a" and self.path_a:
            idx = self.combo_face_a.currentData()
            if idx is not None and idx >= 0:
                self.show_img(self.path_a, self.lbl_img_a, idx)
        elif img_key == "image_b" and self.path_b:
            idx = self.combo_face_b.currentData()
            if idx is not None and idx >= 0:
                self.show_img(self.path_b, self.lbl_img_b, idx)

    def run_process(self):
        if not self.path_a or not self.path_b:
            QMessageBox.warning(self, "Warning", "Please select both images")
            return

        self.pbar.setVisible(True)
        self.pbar.setValue(0)
        
        idx_a = self.combo_face_a.currentData() or 0
        idx_b = self.combo_face_b.currentData() or 0
        if idx_a < 0: idx_a = 0
        if idx_b < 0: idx_b = 0

        self.worker = ComparisonWorker(
            self.path_a, self.path_b, 
            self.combo_mode.currentText(),
            idx_a, idx_b, 
            self.combo_device.currentData(),
            self.chk_debug.isChecked()
        )
        
        self.worker.progress.connect(lambda v, m: (self.pbar.setValue(v), self.lbl_status.setText(m)))
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        
        self.worker.start()

    def on_finished(self, r1, r2, success_rate):
        self.res1, self.res2 = r1, r2
        self.display_result(r1, self.lbl_res1)
        self.display_result(r2, self.lbl_res2)
        self.pbar.setVisible(False)
        self.lbl_status.setText("Process completed successfully")
        
        QMessageBox.information(self, "Success", f"Face swap completed\n\nSuccess Rate: {round(success_rate, 2)}%")

    def save_results(self):
        if self.res1 is None: return
        
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)

        counter = 1
        while os.path.exists(os.path.join(save_dir, f"swap_{counter:03d}_FaceA_to_BodyB.jpg")):
            counter += 1

        file1 = f"swap_{counter:03d}_FaceA_to_BodyB.jpg"
        file2 = f"swap_{counter:03d}_FaceB_to_BodyA.jpg"
        
        try:
            cv2.imwrite(os.path.join(save_dir, file1), self.res1)
            cv2.imwrite(os.path.join(save_dir, file2), self.res2)
            
            QMessageBox.information(self, "Saved", 
                f"Images saved successfully\n\nFolder: {save_dir}\nFile 1: {file1}\nFile 2: {file2}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if os.path.exists("icon/icon.png"):
        app.setWindowIcon(QIcon("icon/icon.png"))
    win = FaceSwapApp()
    win.show()
    sys.exit(app.exec_())