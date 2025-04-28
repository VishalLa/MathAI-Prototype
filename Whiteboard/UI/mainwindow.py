import os 
import sys 

root_dir = "C:\\Users\\visha\\OneDrive\\Desktop\\MathAI"
sys.path.append(root_dir)

from PySide6.QtWidgets import (
    QMainWindow,
    QApplication,
    QColorDialog,
    QButtonGroup
)
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPainterPath
)
from PySide6.QtCore import Qt

from board import Ui_MainWindow
from boardscene import BoardScene
from CNN_Model.Utils.pre_process import predict_chars 


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.tool_button_group = QButtonGroup()
        self.tool_button_group.setExclusive(True)
        self.tool_button_group.addButton(self.pb_Pen)
        self.tool_button_group.addButton(self.pb_Eraser)
        self.tool_button_group.addButton(self.pb_Select)
        self.tool_button_group.addButton(self.pb_Clear)
        self.tool_button_group.addButton(self.pb_Predict)

        self.current_color = QColor("#000000")

        # Scene setup
        self.scene = BoardScene()
        self.gv_Canvas.setScene(self.scene)
        self.gv_Canvas.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        self.redo_list = []

        # Button connections
        self.pb_Pen.setChecked(True)
        self.pb_Pen.clicked.connect(self.on_pen_selection)
        self.pb_Eraser.clicked.connect(self.on_eraser_selected)
        self.pb_Clear.clicked.connect(self.toggle_clear_canvas)
        self.pb_Select.clicked.connect(self.toggle_selection_mode)
        self.pb_Predict.clicked.connect(self.predict)
        self.pb_Color.clicked.connect(self.color_dialog)
        self.pb_Undo.clicked.connect(self.undo)
        self.pb_Redo.clicked.connect(self.redo)

        self.pb_Pen.clicked.connect(lambda e: self.color_changed(self.current_color))
        self.pb_Eraser.clicked.connect(lambda e: self.color_changed(QColor("#FFFFFF")))

        # Dial settings
        self.dial.sliderMoved.connect(self.change_size)
        self.dial.setMinimum(1)
        self.dial.setMaximum(40)
        self.dial.setWrapping(False)


    def predict(self):
        selected_area = self.scene.get_selected_region()
        if selected_area.size == 0:
            print("No area selected")
            return
        print("Selected area:", selected_area.shape)
        chars = predict_chars(selected_area)
        print(f'Predicted chracters: {chars}')


    def on_pen_selection(self):
        self.scene.enable_selection_mode(False)
        self.color_changed(self.current_color)


    def on_eraser_selected(self):
        self.scene.enable_selection_mode(False)
        self.color_changed(QColor('#FFFFFF'))


    def toggle_selection_mode(self):
        if self.pb_Select.isChecked():
            self.scene.enable_selection_mode(True)
            self.pb_Pen.setChecked(False)
            self.pb_Eraser.setChecked(False)
            self.pb_Clear.setChecked(False)


    def toggle_clear_canvas(self):
        self.clear_canvas()
        self.scene.selection_path = QPainterPath()
        self.tool_button_group.setExclusive(False)
        self.pb_Pen.setChecked(False)
        self.pb_Eraser.setChecked(False)
        self.pb_Select.setChecked(False)
        self.tool_button_group.setExclusive(True)


    def change_size(self):
        self.scene.change_size(self.dial.value())


    def undo(self):
        if self.scene.items():
            latest_item = self.scene.items()
            self.redo_list.append(latest_item)
            self.scene.removeItem(latest_item[0])


    def redo(self):
        if self.redo_list:
            item = self.redo_list.pop(-1)
            self.scene.addItem(item[0])

    def clear_canvas(self):
        self.scene.clear()


    def color_dialog(self):
        color_dialog = QColorDialog()
        color_dialog.show()
        color_dialog.currentColorChanged.connect(lambda e: self.color_dialog_color_changed(color_dialog.currentColor()))
        self.current_color = color_dialog.currentColor()


    def color_dialog_color_changed(self, current_color):
        self.color_changed(current_color)
        if self.pb_Eraser.isChecked():
            self.pb_Eraser.setChecked(False)
        self.pb_Pen.setChecked(True)


    def color_changed(self, color):
        self.scene.change_color(color)


if __name__ == '__main__':
    app = QApplication()
    window = MainWindow()
    window.show()
    app.exec()
