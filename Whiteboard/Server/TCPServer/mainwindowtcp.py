import os 
import sys
import cv2
import torch  

root_dir = "C:\\Users\\visha\\OneDrive\\Desktop\\MathAI"
sys.path.append(root_dir)

from PySide6.QtWidgets import (
    QMainWindow,
    QGraphicsScene,
    QApplication,
    QGraphicsPathItem,
    QGraphicsItem,
    QColorDialog,
    QFileDialog,
    QButtonGroup
)

from PySide6.QtGui import (
    QPen,
    Qt,
    QPainter,
    QPainterPath,
    QColor,
    QImage
)

from PySide6.QtCore import (
    Qt,
    QRectF,
    QPointF
)
import json

from Whiteboard.board import Ui_MainWindow
from .boardscenetcp import BoardScene

from CNN_Model.Utils.pre_process import prepare_canvas
from CNN_Model.Utils.pred import predict_characters

from Server.tcpServerNet import start_server, MyServer, signal_manager

myserver = MyServer()


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, model):
        super().__init__()
        self.setupUi(self)

        self.child_windows = []

        self.predict_result = None 

        self.network = model

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


        self.actionCoolDude.triggered.connect(self.New_file)
        self.actionOpen.triggered.connect(self.load_file)
        self.actionSave.triggered.connect(self.save_file)
        self.actionSave_As.triggered.connect(self.save_file)

        self.statusBar().showMessage("Ready")
        self.statusBar().showMessage(f"Prediction: {self.predict_result}")


    def save_file(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'Whiteboard File (*.json)')
        if filename:
            if not filename.endswith('.json'):
                filename += '.json'
            data = {
                'lines': [],
                'scene_rect': [self.scene.sceneRect().width(), self.scene.sceneRect().height()],
                'color': self.scene.color.name(),
                'size': self.scene.size
            }
            for item in reversed(self.scene.items()):
                if isinstance(item, QGraphicsPathItem):
                    line_data = {
                        'color': item.pen().color().name(),
                        'width': item.pen().widthF(),
                        'points': []
                    }
                    for subpath in item.path().toSubpathPolygons():
                        line_data['points'].extend([(point.x(), point.y()) for point in subpath])
                    data['lines'].append(line_data)
            with open(filename, 'w') as file:
                json.dump(data, file)


    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Whiteboard Files (*.json)")
        if filename:
            try:
                with open(filename, 'r') as file:
                    data = json.load(file)

                self.scene.clear()
                self.scene.setSceneRect(0, 0, data['scene_rect'][0], data['scene_rect'][1])
                self.scene.change_color(QColor(data['color']))
                self.scene.change_size(data['size'])

                for line_data in data['lines']:
                    path = QPainterPath()
                    path.moveTo(line_data['points'][0][0], line_data['points'][0][1])
                    for subpath in line_data['points'][1:]:
                        path.lineTo(subpath[0], subpath[1])
                    pathItem = QGraphicsPathItem(path)
                    my_pen = QPen(QColor(line_data['color']), line_data['width'])
                    my_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    pathItem.setPen(my_pen)
                    self.scene.addItem(pathItem)
            except Exception as e:
                print(f"Error loading file: {e}")


    def build_scene_file(self, data):
        self.scene.clear()
        scene_file = data['scene_info']
        self.scene.setSceneRect(0, 0, *scene_file.get('scene_rect', [600, 500]))
        self.scene.change_color(QColor(scene_file.get('color', '#000000')))
        self.scene.change_size(scene_file.get('size', 5))

        for line_data in scene_file.get('lines', []):
            path = QPainterPath()
            path.moveTo(line_data['points'][0][0], line_data['points'][0][1])
            for subpath in line_data['points'][1:]:
                path.lineTo(subpath[0], subpath[1])
            pathItem = QGraphicsPathItem(path)
            my_pen = QPen(QColor(line_data['color']), line_data['width'])
            my_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pathItem.setPen(my_pen)
            self.scene.addItem(pathItem)

    
    def New_file(self):
        new_file = MainWindow()
        new_file.show()
        self.child_windows.append(new_file)


    def predict(self):
        selected_area = self.scene.get_selected_region()
        cv2.imwrite("Selected area.png", selected_area)
        if selected_area.size == 0:
            print("No area selected")
            return
        
        print("Selected area:", selected_area.shape)
        
        processed_canvas = prepare_canvas(selected_area)
        self.predict_result = predict_characters(model=self.network, canvas_array=processed_canvas)
        print(f'Predicted chracters: {self.predict_result}')


    def on_pen_selection(self):
        self.scene.enable_selection_mode(False)
        self.pb_Select.setChecked(False)
        self.pb_Eraser.setChecked(False)
        self.pb_Pen.setChecked(True)
        self.color_changed(self.current_color)


    def on_eraser_selected(self):
        self.scene.enable_selection_mode(False)
        self.pb_Select.setChecked(False)
        self.pb_Eraser.setChecked(True)
        self.pb_Pen.setChecked(False)
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
