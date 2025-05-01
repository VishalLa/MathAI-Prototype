import os 
import sys 

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
from boardscenetcp import BoardScene

from CNN_Model.Utils.pre_process import predict_chars 
from Server.tcpServerNet import start_server, MyServer, signal_manager

myserver = MyServer()


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


        self.actionCoolDude.triggered.connect(self.New_file)
        self.actionSave_As.triggered.connect(self.save_file)
        self.actionNew_3.triggered.connect(self.load_file)


    def save_file(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'Whiteboard File (*.json)') # open dialog
        # window to save file

        if filename:
            data = {
                'lines': [], # Store info of each line
                'scene_rect': [self.scene.width(), self.scene.height()], # Store dimension of scene
                'color': self.scene.color.name(), # Store the used coloe
                'size': self.scene.size # store the size of the pen
            }

            # loop for checking of drawin path
            for item in reversed(self.scene.items()):
                if isinstance(item, QGraphicsPathItem):
                    line_data = line_data = {
                        'color': item.pen().color().name(),
                        'width': item.pen().widthF(),
                        'points': [],  # stores the (X,Y) coordinate of the line
                        # 'z_value': item.zValue()  # Store the z-value
                    }

                    # Extract points form the path
                    for subpath in item.path().toSubpathPolygons():
                        # to SubpathPolygons method is used to break down
                        # the complex line into sub parts and store it
                        line_data['points'].extend([(point.x(), point.y()) for point in subpath])

                    data['lines'].append(line_data)

            with open(filename, 'w') as file:
                json.dump(data, file)


    def load_file(self):
        self.scene.z_index_counter = 0
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Whiteboard Files (*.json)")  # open dialog
        # window to Open the file
        if filename:  # reading the file
            with open(filename, 'r') as file:
                data = json.load(file)

            self.scene.clear()

            # Set scene properties
            self.scene.setSceneRect(0, 0, data['scene_rect'][0], data['scene_rect'][1])
            self.scene.change_color(QColor(data['color']))
            self.scene.change_size(data['size'])

            z_index_counter = 0

            items = []  # List to hold items before sorting
            # Add lines to the scene
            for line_data in data['lines']:
                path = QPainterPath()
                path.moveTo(line_data['points'][0][0], line_data['points'][0][1])

                for subpath in line_data['points'][1:]:
                    path.lineTo(subpath[0], subpath[1])

                pathItem = QGraphicsPathItem(path)
                pathItem.setZValue(z_index_counter)  # Assign unique z-index
                z_index_counter += 1  # Increment counter
                my_pen = QPen(QColor(line_data['color']), line_data['width'])
                my_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pathItem.setPen(my_pen)

                # items.append(pathItem)

                self.scene.addItem(pathItem)
            #
            # items.sort(key=lambda x: x.zValue())
            # for item in items:
                # self.scene.addItem(pathItem)


    def build_scene_file(self, data):
        self.scene.clear()
        scene_file = data['scene_info']

        undo_flag = data['flag']

        self.scene.drawing = True
        prev = {'lines': [],  # stores info of each line drawn
                'scene_rect': [],  # stores dimension of scene
                'color': "",  # store the color used
                'size': 20  # store the size of the pen
                }

        top_z = self.scene.get_topmost_z_index()
        try:
            if 'scene_info' in data:
                if 'scene_rect' in scene_file:
                    scene_rect = scene_file['scene_rect']
                    self.scene.setSceneRect(0, 0, scene_rect[0], scene_rect[1])
                else:
                    # Provide default scene rectangle if 'scene_rect' key is missing
                    self.scene.setSceneRect(0, 0, 600, 500)  # Adjust the default values as needed
                self.scene.change_color(QColor(scene_file['color']))
                self.scene.color.setAlpha(255)

                if 'size' in scene_file.keys():
                    self.scene.change_size(scene_file['size'])
                    prev = scene_file
                else:
                    pass
                # Add lines to the scene
                if 'lines' in scene_file:
                    for line_data in scene_file['lines']:
                        path = QPainterPath()
                        path.moveTo(line_data['points'][0][0], line_data['points'][0][1])
                        print("line_data is cool")

                        for subpath in line_data['points'][1:]:
                            path.lineTo(subpath[0], subpath[1])
                        print("Whatever tf subpath is, it's cool")

                        self.scene.temppath.clear()

                        if path not in self.scene.temppath:
                            self.scene.temppath.append(path)
                        print(3)
                        pathItem = QGraphicsPathItem(path)
                        self.scene.next_z_index += 10  # Adjust increment as needed
                        pathItem.setZValue(self.scene.next_z_index)
                        my_pen = QPen(QColor(line_data['color']), line_data['width'])
                        my_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                        pathItem.setPen(my_pen)
                        pathItem.setZValue(self.scene.itemsBoundingRect().height() + 1)
                        self.scene.addItem(pathItem)

        except IndexError as e:
            print(e)
            pass

    
    def New_file(self):
        new_file = MainWindow()
        new_file.show()


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
    start_server(myserver)
    window = MainWindow()
    signal_manager.action_signal.connect(window.scene.get_drawing_events)
    signal_manager.data_ack.connect(window.build_scene_file)
    window.show()

    app.exec()
