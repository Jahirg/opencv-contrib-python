import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

########################################################################
# CLAUDE
########################################################################
#python 						3.12.9
#opencv-contrib-python        	4.14.0.94 # Modificado para SURF
#opencv 						4.14.0.94
#numpy							2.5.2
#matplotlib						3.11.1
#PyQt5							5.15.11
########################################################################

DETECTORES_DISPONIBLES = {
    'ORB':   lambda: cv2.ORB_create(nfeatures=1500),
    'SIFT':  lambda: cv2.SIFT_create(),
    'BRISK': lambda: cv2.BRISK_create(),
    'AKAZE': lambda: cv2.AKAZE_create(),
    'KAZE':  lambda: cv2.KAZE_create(),
    'SURF': lambda: cv2.xfeatures2d.SURF_create(400),
}
BINARIOS = ('ORB', 'BRISK', 'AKAZE')
RATIO_TEST = 0.75


def cv_a_qpixmap(img_bgr, tamano_max=None):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if tamano_max:
        pix = pix.scaled(*tamano_max, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


class VentanaDeteccion(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteccion de imagen en video")
        self.resize(1000, 600)

        self.ref_bgr = None
        self.ref_gray = None
        self.kp_ref = None
        self.des_ref = None
        self.esquinas_ref = None
        self.cap = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.procesar_frame)

        self._construir_ui()

    # ------------------------------------------------------------
    def _construir_ui(self):
        # ---------- Panel izquierdo: opciones (cuadro azul) ----------
        panel_opciones = QGroupBox("Opciones")
        layout_opciones = QVBoxLayout()

        self.btn_seleccionar = QPushButton("Seleccionar imagen...")
        self.btn_seleccionar.clicked.connect(self.seleccionar_imagen)
        layout_opciones.addWidget(self.btn_seleccionar)

        form = QFormLayout()
        self.combo_detector = QComboBox()
        self.combo_detector.addItems(DETECTORES_DISPONIBLES.keys())
        form.addRow("Algoritmo:", self.combo_detector)

        self.spin_min_matches = QSpinBox()
        self.spin_min_matches.setRange(4, 200)
        self.spin_min_matches.setValue(15)
        form.addRow("Min. matches:", self.spin_min_matches)
        layout_opciones.addLayout(form)

        # ---------- Miniatura de la imagen a buscar (cuadro negro) ----------
        layout_opciones.addWidget(QLabel("Imagen a buscar:"))
        self.label_miniatura = QLabel("Sin imagen")
        self.label_miniatura.setFixedSize(180, 140)
        self.label_miniatura.setAlignment(Qt.AlignCenter)
        self.label_miniatura.setStyleSheet(
            "border: 2px solid black; background-color: #f0f0f0;"
        )
        layout_opciones.addWidget(self.label_miniatura)

        self.label_estado = QLabel("Estado: sin iniciar")
        self.label_estado.setWordWrap(True)
        layout_opciones.addWidget(self.label_estado)

        self.btn_iniciar = QPushButton("Iniciar deteccion")
        self.btn_iniciar.clicked.connect(self.iniciar_deteccion)
        self.btn_iniciar.setEnabled(False)
        layout_opciones.addWidget(self.btn_iniciar)

        self.btn_detener = QPushButton("Detener")
        self.btn_detener.clicked.connect(self.detener_deteccion)
        self.btn_detener.setEnabled(False)
        layout_opciones.addWidget(self.btn_detener)

        layout_opciones.addStretch()
        panel_opciones.setLayout(layout_opciones)
        panel_opciones.setFixedWidth(220)

        # ---------- Panel derecho: video (cuadro rojo) ----------
        self.label_video = QLabel("La camara aparecera aqui")
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setStyleSheet(
            "border: 2px solid red; background-color: black; color: white;"
        )
        self.label_video.setMinimumSize(640, 480)

        # ---------- Layout principal ----------
        contenedor = QWidget()
        layout_principal = QHBoxLayout()
        layout_principal.addWidget(panel_opciones)
        layout_principal.addWidget(self.label_video, stretch=1)
        contenedor.setLayout(layout_principal)
        self.setCentralWidget(contenedor)

    # ------------------------------------------------------------
    def seleccionar_imagen(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen de referencia", "",
            "Imagen (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if not file_path:
            return

        self.ref_bgr = cv2.imread(file_path)
        self.ref_gray = cv2.cvtColor(self.ref_bgr, cv2.COLOR_BGR2GRAY)

        self.label_miniatura.setPixmap(cv_a_qpixmap(self.ref_bgr, (170, 130)))
        self.btn_iniciar.setEnabled(True)
        self.label_estado.setText("Imagen cargada. Lista para iniciar.")

    def iniciar_deteccion(self):
        if self.ref_gray is None:
            return

        nombre_detector = self.combo_detector.currentText()
        self.detector = DETECTORES_DISPONIBLES[nombre_detector]()
        self.es_binario = nombre_detector in BINARIOS
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING if self.es_binario else cv2.NORM_L2)
        self.min_matches = self.spin_min_matches.value()

        self.kp_ref, self.des_ref = self.detector.detectAndCompute(self.ref_gray, None)
        if self.des_ref is None or len(self.kp_ref) < self.min_matches:
            self.label_estado.setText("La imagen no tiene suficientes puntos detectables.")
            return

        h_ref, w_ref = self.ref_gray.shape
        self.esquinas_ref = np.float32([
            [0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]
        ]).reshape(-1, 1, 2)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.label_estado.setText("No se pudo abrir la camara.")
            self.cap = None
            return

        self.btn_seleccionar.setEnabled(False)
        self.combo_detector.setEnabled(False)
        self.spin_min_matches.setEnabled(False)
        self.btn_iniciar.setEnabled(False)
        self.btn_detener.setEnabled(True)

        self.label_estado.setText(f"[{nombre_detector}] {len(self.kp_ref)} keypoints. Buscando...")
        self.timer.start(30)  # ~33 fps

    def detener_deteccion(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.label_video.clear()
        self.label_video.setText("La camara aparecera aqui")

        self.btn_seleccionar.setEnabled(True)
        self.combo_detector.setEnabled(True)
        self.spin_min_matches.setEnabled(True)
        self.btn_iniciar.setEnabled(True)
        self.btn_detener.setEnabled(False)
        self.label_estado.setText("Detenido.")

    # ------------------------------------------------------------
    def procesar_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.detector.detectAndCompute(gray, None)

        estado = f"Buscando... 0/{self.min_matches} matches"
        detectado = False

        if des_frame is not None and len(kp_frame) >= 2:
            matches_knn = self.matcher.knnMatch(self.des_ref, des_frame, k=2)
            buenos = [m for par in matches_knn if len(par) == 2
                      for m, n in [par] if m.distance < RATIO_TEST * n.distance]

            estado = f"Buscando... {len(buenos)}/{self.min_matches} matches"

            if len(buenos) >= self.min_matches:
                pts_ref = np.float32([self.kp_ref[m.queryIdx].pt for m in buenos]).reshape(-1, 1, 2)
                pts_frame = np.float32([kp_frame[m.trainIdx].pt for m in buenos]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(pts_ref, pts_frame, cv2.RANSAC, 5.0)

                if H is not None:
                    esquinas_frame = cv2.perspectiveTransform(self.esquinas_ref, H)
                    frame = cv2.polylines(
                        frame, [np.int32(esquinas_frame)], True, (0, 255, 0), 3, cv2.LINE_AA
                    )
                    estado = f"Objeto detectado ({len(buenos)} matches)"
                    detectado = True

        self.label_estado.setText(estado)
        self.label_estado.setStyleSheet("color: green;" if detectado else "color: red;")

        pix = cv_a_qpixmap(frame)
        pix = pix.scaled(self.label_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_video.setPixmap(pix)

    def closeEvent(self, event):
        self.detener_deteccion()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = VentanaDeteccion()
    ventana.show()
    sys.exit(app.exec_())
