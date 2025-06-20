import os 
import json 
import numpy as np 

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QDockWidget,
    QColorDialog,
    QLabel,
    QToolBar,
    QTextEdit,
    QSizePolicy,
    QFileDialog,
    QMessageBox,
    QGraphicsView
)

from PySide6.QtGui import (
    QColor,
    QTransform,
    QIcon,
    QImage,
    QAction,
    QPixmap
)

from PySide6.QtCore import (
    Qt,
    QTimer,
    QRectF,
    QSize
)

import Icons_rc
from graphics_componets import CustomGraphicsScene, CustomGraphicsView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MathAI Whiteboard')
        self.setGeometry(100, 100, 1200, 800)

        self.current_file_path = None 
        self.is_dirty = False

        # ---Central Widgget & Layout---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        
        # ---Graphics Scene and View---
        self.scene = CustomGraphicsScene(self)
        self.view = CustomGraphicsView(self.scene, self)

        main_layout.addWidget(self.view)

        # Connect the new Signal for selection
        self.view.rubber_band_selection_finished.connect(self.handle_rubber_band_selection)
        self.last_selected_nparray = None 

        # Connect a signal to detect changes in the scene for 'is_dirty' flag
        self.scene.changed.connect(self.set_dirty)


        # ---Left Sidebar (Tools) ---
        self.create_left_sidebar()
        main_layout.insertWidget(0, self.left_sidebar_widget)


        # --- Right Sidebar (AI Predictions) ---
        self.ai_dock_widget = self.create_right_sidebar()
        self.addDockWidget(Qt.RightDockWidgetArea, self.ai_dock_widget)
        self.ai_dock_widget.hide()

        # --- Top Toolbar for AI Window Button & Zoom Controls ---
        self.create_top_toolbar()

        # --- Menu Bar ---
        self.create_menu_bar()

        self.current_pen_color = Qt.black
        self.update_window_title()


    def set_dirty(self):
        if not self.is_dirty:
            self.is_dirty = True
            self.update_window_title()


    def set_clean(self):
        if self.is_dirty:
            self.is_dirty = False
            self.update_window_title()


    def update_window_title(self):
        title = "MathAI Whiteboard"
        if self.current_file_path:
            title += f' - {os.path.basename(self.current_file_path)}'
        
        if self.is_dirty:
            title += ' *'
        
        self.setWindowTitle(title)


    def create_menu_bar(self):
        menu_bar = self.menuBar()

        # file menu 
        file_menu = menu_bar.addMenu('File')

        new_action = QAction('New', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.file_new)
        file_menu.addAction(new_action)
        
        open_action = QAction('Open...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered(self.file_open)

        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.file_save)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.file_save_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close) # Connect to close event for prompt
        file_menu.addAction(exit_action)


    def file_new(self):
        if self.maybe_save():
            self.scene.clear_canvas()
            self.current_file_path = None
            self.set_clean()
            self.update_window_title()


    def file_open(self):
        if self.maybe_save():
            file_path, _ = QFileDialog.getOpenFileName(
                self, 'Open Drawing', '', 'MathAI Drawing File (*.maijson);;All Files (*)'
            )
            
            if file_path:
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)

                    self.scene.deserialize(data)
                    self.current_file_path = file_path
                    self.set_clean()
                    self.update_window_title()

                except Exception as e:
                    QMessageBox.critical(self, "Error", f'Cloud not open file: {e}')


    def file_save(self):
        if self.current_file_path:
            return self._save_to_file(self.current_file_path)
        else:
            return self.file_save_as()


    def file_save_as(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Drawing As", "", "MathAI Drawing Files (*.maijson);;All Files (*)"
        )
        if file_path:
            self.current_file_path = file_path
            return self._save_to_file(self.current_file_path)
        return False # User cancelled


    def _save_to_file(self, file_path):
        try:
            data = self.scene.serialize()
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            self.set_clean()
            print(f"Drawing saved to: {file_path}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save file: {e}")
            return False


    def maybe_save(self):
        """Asks the user to save changes if the document has been modified."""
        if self.is_dirty:
            reply = QMessageBox.warning(
                self,
                "MathAI Whiteboard",
                "The document has been modified.\nDo you want to save your changes?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                return self.file_save()
            elif reply == QMessageBox.Cancel:
                return False
        return True # No changes or user chose to discard


    def closeEvent(self, event):
        """Handles closing the application, prompting to save if dirty."""
        if self.maybe_save():
            event.accept()
        else:
            event.ignore()


    def handle_rubber_band_selection(self, selected_rect: QRectF):
        print(f'Rubber band selection finished: {selected_rect.width()}x{selected_rect.height()} at {selected_rect.topLeft()}')

        if selected_rect.width() > 0 and selected_rect.height() > 0:
            pixmap = self.scene.render_area_to_pixmap(selected_rect)
            self.last_selected_nparray = self.pixmap_to_nparray(pixmap)
            
            print(f"Converted selection to numpy array with shape: {self.last_selected_nparray.shape}")

            self.ai_text_output.append(f"<br><b>Image Selected:</b> {selected_rect.width()}x{selected_rect.height()} pixels.")

        else:
            self.last_selected_numpy_array = None
            print("Selection rectangle is too small or invalid.")
            self.ai_text_output.append("<br>No valid selection made for prediction.")


    def pixmap_to_nparray(self, pixmap: QPixmap) -> np.ndarray:
        image = pixmap.toImage()

        if image.format() not in (QImage.Format_ARGB32, QImage.Format_RGB32):
            image = image.convertToFormat(QImage.Format_ARGB32)

        width = image.width()
        height = image.height()
        bytes_per_line = image.bytesPerLine()

        ptr = image.constBits() # Use constBits() for read-only access
        ptr.setsize(height * bytes_per_line)
        arr = np.array(ptr).reshape(height, width, 4)
        return arr


    def create_left_sidebar(self):
        self.left_sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(self.left_sidebar_widget)
        sidebar_layout.setAlignment(Qt.AlignTop)
        self.left_sidebar_widget.setFixedWidth(80)
        sidebar_layout.addSpacing(10)

        icon_size = QSize(40, 40)

        self.pen_button = QPushButton()
        self.pen_button.setIcon(QIcon(':/Icons/pen.png'))
        self.pen_button.setIconSize(icon_size)
        self.pen_button.setCheckable(True)
        self.pen_button.setChecked(True)
        self.pen_button.setToolTip('Pen Tool (Ctrl+P)')
        self.pen_button.clicked.connect(self.set_tool_pen)
        sidebar_layout.addWidget(self.pen_button)


        self.eraser_button = QPushButton()
        self.eraser_button.setIcon(':/Icons/eraser.png')
        self.eraser_button.setIconSize(icon_size)
        self.eraser_button.setCheckable(True)
        self.eraser_button.setToolTip('Eraser Tool (Ctrl+E)')
        self.eraser_button.clicked.connect(self.set_tool_eraser)
        sidebar_layout.addWidget(self.eraser_button)

        self.select_button = QPushButton()
        self.select_button.setIcon(':/Icons/select.png')
        self.select_button.setIconSize(icon_size)
        self.select_button.setCheckable(True)
        self.select_button.setToolTip('Selection Tool (Ctrl+S)')
        self.select_button.clicked.connect(self.set_tool_select)
        sidebar_layout.addWidget(self.select_button)

        sidebar_layout.addSpacing(20)

        # ---Action buttons (Predict, Clear, Color)---
        self.predict_button = QPushButton()
        self.predict_button.setIcon(':/Icons/ellipse.png')
        self.predict_button.setIconSize(icon_size)
        self.predict_button.setToolTip('Run AI Prediction')
        self.predict_button.clicked.connect(self.run_prediction)
        sidebar_layout.addWidget(self.predict_button)

        self.clear_button = QPushButton('Clear')
        self.clear_button.setToolTip('Clear Canvas / New Drawing (Ctrl+N)')
        self.clear_button.clicked.connect(self.file_new)
        sidebar_layout.addWidget(self.clear_button)

        self.color_button = QPushButton('Color')
        self.color_button.setToolTip('Choose Pen Color')
        self.color_button.clicked.connect(self.choose_pen_color)
        sidebar_layout.addWidget(self.color_button)

        sidebar_layout.addSpacing(20)


        # --- Undo/Redo Buttons ---
        self.undo_button = QPushButton()
        self.undo_button.setIcon(QIcon(":/Icons/undo.png")) # <-- CHANGED PATH
        self.undo_button.setIconSize(icon_size)
        self.undo_button.setToolTip("Undo Last Action (Ctrl+Z)")
        self.undo_button.clicked.connect(self.scene.undo)
        sidebar_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton()
        self.redo_button.setIcon(QIcon(":/Icons/redo.png")) # <-- CHANGED PATH
        self.redo_button.setIconSize(icon_size)
        self.redo_button.setToolTip("Redo Last Action (Ctrl+Y)")
        self.redo_button.clicked.connect(self.scene.redo)
        sidebar_layout.addWidget(self.redo_button)


    def create_top_toolbar(self):
        toolbar = self.addToolBar("Main Controls")
        toolbar.setObjectName("MainControlsToolbar")
        toolbar.setMovable(True)

        self.ai_window_button = QPushButton("AI Window")
        self.ai_window_button.setIcon(QIcon(":/Icons/rounded_rect.png")) # <-- CHANGED PATH
        self.ai_window_button.setIconSize(QSize(24, 24))
        self.ai_window_button.clicked.connect(self.toggle_ai_dock)
        toolbar.addWidget(self.ai_window_button)

        toolbar.addSeparator()

        self.zoom_toggle_button = QPushButton("Zoom")
        self.zoom_toggle_button.setFixedSize(40, 40)
        self.zoom_toggle_button.setStyleSheet(
            "QPushButton { border-radius: 20px; background-color: #4CAF50; color: white; }"
            "QPushButton:pressed { background-color: #388E3C; }"
        )
        self.zoom_toggle_button.clicked.connect(self.toggle_zoom_slider)
        toolbar.addWidget(self.zoom_toggle_button)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(200)
        self.zoom_slider.hide()

        toolbar.addWidget(self.zoom_slider)
        self.zoom_slider.valueChanged.connect(self.apply_zoom_from_slider)


    def update_tool_selection(self):
        sender_button = self.sender()
        for btn in self.tool_buttons:
            if btn is not sender_button:
                btn.setChecked(False)
            else:
                btn.setChecked(True)


    def set_tool_pen(self):
        self.scene.drawing_mode = 'pen'
        self.scene.set_pen_color(self.current_pen_color)
        self.view.setDragMode(Qt.NoDrag)


    def set_tool_eraser(self):
        self.scene.drawing_mode = 'eraser'
        self.view.setDragMode(Qt.NoDrag)


    def set_tool_select(self):
        self.scene.drawing_mode = 'select'
        self.view.setDragMode(QGraphicsView.RubberBandDrag)


    def choose_pen_color(self):
        color = QColorDialog.getColor(self.current_pen_color, self, "Choose Pen Color")
        if color.isValid():
            self.current_pen_color = color
            if self.scene.drawing_mode == 'pen':
                self.scene.set_pen_color(color)


    def create_right_sidebar(self):
        dock = QDockWidget("AI Solutions", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setObjectName("AISolutionsDock")

        dock_content_widget = QWidget()
        dock_layout = QVBoxLayout(dock_content_widget)

        self.ai_text_output = QTextEdit()
        self.ai_text_output.setReadOnly(True)
        self.ai_text_output.setText("<b>AI Predictions and solutions will be displayed here.</b><br><br>"
                                    "1. Use the 'Select' tool to drag a rectangle on the canvas.<br>"
                                    "2. Click the 'Predict' button on the left sidebar to analyze the selected area.")
        self.ai_text_output.setMinimumWidth(200)

        dock_layout.addWidget(self.ai_text_output)
        dock.setWidget(dock_content_widget)

        return dock


    def toggle_ai_dock(self):
        if self.ai_dock_widget.isVisible():
            self.ai_dock_widget.hide()
        else:
            self.ai_dock_widget.show()
            self.ai_dock_widget.raise_()


    def toggle_zoom_slider(self):
        self.zoom_slider.setVisible(not self.zoom_slider.isVisible())


    def apply_zoom_from_slider(self, value):
        scale_factor = value / 100.0
        transform = QTransform()
        transform.scale(scale_factor, scale_factor)
        self.view.setTransform(transform)


    def run_prediction(self):
        if self.last_selected_numpy_array is not None:
            print("Running AI Prediction on the last selected area...")
            self.ai_text_output.append("<hr><b>AI Analysis:</b>")
            self.ai_text_output.append(f"Analyzing {self.last_selected_numpy_array.shape[0]}x{self.last_selected_numpy_array.shape[1]} image...")
            QTimer.singleShot(1500, self._display_mock_prediction)
        else:
            self.ai_text_output.append("<br><b>No image selected for prediction.</b> Please use the 'Select' tool to drag a region on the canvas first.")
            print("No image selected for prediction.")


    def _display_mock_prediction(self):
        self.ai_text_output.append("<b>Result:</b> This appears to be a linear equation.")
        self.ai_text_output.append("<b>Equation:</b> $y = mx + c$")
        self.ai_text_output.append("<b>Solution Steps:</b><br>"
                                    "1. Identify slope (m) and y-intercept (c).<br>"
                                    "2. Use algebraic manipulation to find unknowns.")
        self.ai_text_output.verticalScrollBar().setValue(self.ai_text_output.verticalScrollBar().maximum())


import sys 
from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
