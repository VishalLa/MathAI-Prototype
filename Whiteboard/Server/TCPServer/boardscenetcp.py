import sys
import os

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)


from PySide6.QtWidgets import (
    QMainWindow,
    QGraphicsScene,
    QApplication,
    QGraphicsPathItem,
    QGraphicsItem,
    QColorDialog,
    QFileDialog
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

import numpy as np

from Server.tcpServerNet import start_server, MyServer, signal_manager

itemType = set()
myserver = MyServer()


class BoardScene(QGraphicsScene):
    def __init__(self):
        super().__init__()

        self.setSceneRect(0, 0, 600, 500)
        self.flag = 0
        self.next_z_index = 0

        self.temppath = []
        self.path = None
        self.previous_possition = None
        self.drawing = False
        self.selection_mode = False

        self.selected_area = None 
        self.selection_path = QPainterPath()

        self.selection_start = None
        self.selection_end = None

        self.color = QColor("#000000")
        self.size = 5
        self.pathItem = None 
        self.drawn_path = []

        self.my_pen = None 

        self.data = {
            'event': '',
            'state': False,
            'position': None,
            'color': None,
            'widthF': None,
            'width': None,
            'capStyle': None,
            'joinStyle': None,
            'style': None,
            'pattern': None,
            'patterenOffset': None 
        }

        self.default_z_index = 0
        self.eraser_z_index = None 

    
    def get_topmost_z_index(self):
        highest_z = float('-inf')
        for item in self.items():
            highest_z = max(highest_z, item.zValue())
        return highest_z
    
    
    def set_eraser_z_index(self, z_index):
        self.eraser_z_index = z_index
        for item in self.items():
            if isinstance(item, QGraphicsPathItem):
                if item.pen().color() != Qt.white:
                    item.setZValue(self.default_z_index)

    
    def set_default_z_index(self):
        for item in self.items():
            if isinstance(item, QGraphicsPathItem):
                if item.pen().color() != Qt.white:
                    item.setZValue(self.default_z_index)


    def change_color(self, color):
        self.color = color

    
    def change_size(self, size):
        self.size = size

    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.selection_mode:
                self.drawing = False
                self.selection_start = event.scenePos()
                self.selected_path = QPainterPath()
                self.selection_path.moveTo(self.selection_start)

            else:
                self.start_drawing(event.scenePos())
                
                for item in self.items():
                    itemType.add(type(item).__name__)
                
                self.drawing_events('mousePressEvent')

                signal_manager.function_call.emit(True)


    def mouseMoveEvent(self, event):
        if self.selection_mode and self.selection_start is not None:
                self.selection_end = event.scenePos()
                self.update()
            
        elif self.drawing and self.pathItem:
            curr_position = event.scenePos()
            self.path.lineTo(curr_position)
            self.pathItem.setPath(self.path)
            self.previous_position = curr_position

            for item in self.items():
                itemType.add(type(item).__name__)

            self.drawing_events('mouseMoveEvent')
            signal_manager.function_call.emit(True)

    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.selection_mode and self.selection_path is not None:
                self.selected_area = self.get_selected_region() 
                self.selection_path = QPainterPath()
                self.update()
            else:
                self.end_drawing()

        
    def get_drawing_path(self):
        return self.drawn_path  # was self.drawn_paths

    def clear_drawn_paths(self):
        self.drawn_path = []
        self.clear()


    def drawing_events(self, event_name: str):
        self.data["state"] = self.drawing
        self.data["event"] = event_name
        self.data["position"] = self.previous_position

        if self.my_pen:
            self.data["color"] = self.my_pen.color()
            self.data["widthF"] = self.my_pen.widthF()
            self.data["width"] = self.my_pen.width()
            self.data["capStyle"] = self.my_pen.capStyle()
            self.data["joinStyle"] = self.my_pen.joinStyle()
            self.data["style"] = self.my_pen.style()
            self.data["pattern"] = self.my_pen.dashPattern()
            self.data["patternOffset"] = self.my_pen.dashOffset()
        else:
            # Default or safe fallback values
            self.data["color"] = QColor(Qt.black)
            self.data["widthF"] = 1.0
            self.data["width"] = 1
            self.data["capStyle"] = Qt.RoundCap
            self.data["joinStyle"] = Qt.RoundJoin
            self.data["style"] = Qt.SolidLine
            self.data["pattern"] = []
            self.data["patternOffset"] = 0
    
    
    def start_drawing(self, pos):
        self.drawing = True
        self.path = QPainterPath()
        self.previous_position = pos
        self.path.moveTo(self.previous_position)

        self.pathItem = QGraphicsPathItem()
        my_pen = QPen(self.color, self.size)
        my_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.pathItem.setPen(my_pen)
        self.addItem(self.pathItem)


    def end_drawing(self):
        self.previous_position = None 
        self.drawing = False 
        self.drawn_path.append(self.path)

        for item in self.items():
            itemType.add(type(item).__name__)
        
        self.drawing_events('mouseReleaseEvent')
        signal_manager.function_call.emit(True)


    def enable_selection_mode(self, enable):
        self.selection_mode = enable

        self.selection_start = None 
        self.selection_end = None 
        self.selection_path = QPainterPath()
        self.update()

        for item in self.items():
            item.setFlag(QGraphicsItem.ItemIsSelectable, enable)
            item.setFlag(QGraphicsItem.ItemIsMovable, enable)

    
    def configure_pen(self, scene_info):
        color = QColor(scene_info['color'])
        size = scene_info['width']
        style = getattr(Qt.PenStyle, scene_info['style'])
        pattern = scene_info['pattern']

        self.my_pen = QPen(color, size)
        self.my_pen.setStyle(style)
        self.my_pen.setDashPattern(pattern)
        if self.pathItem:
            self.pathItem.setPen(self.my_pen)


    def get_selected_region(self):
        if self.selection_start is None or self.selection_end is None:
            return None 
        
        rect = QRectF(self.selection_start, self.selection_end).normalized()

        if rect.width() <= 1 or rect.height() <= 1:
            return np.array([])

        img = QImage(int(rect.width()), int(rect.height()), QImage.Format.Format_RGB32)
        img.fill(Qt.white)

        painter = QPainter(img)
        self.render(painter, target=QRectF(img.rect()), source=rect)
        painter.end()

        buffer = img.bits()
        arr = np.array(buffer, dtype=np.uint8).reshape((img.height(), img.width(), 4))

        return arr 


    def get_drawing_events(self, scene_info):
        prev = None 
        point = QPointF(scene_info['position'][0], scene_info['position'][1])
        reconstructed_path = scene_info['path']

        if scene_info['event'] == 'mousePressEvent':
            if not self.drawing:
                self.path = QPainterPath()
                self.pathItem = QGraphicsPathItem()
                self.configure_pen(scene_info)
                self.pathItem.setPen(self.my_pen)
                self.pathItem.setPath(self.path)
                self.addItem(self.pathItem)
                self.drawing = True
                prev = point
                self.flag = 1

        elif scene_info['event'] == 'mouseMoveEvent':
            color = QColor(scene_info['color'])
            size = scene_info['width']
            pattern = scene_info['pattern']

            style = getattr(Qt.PenStyle, scene_info['style'])
                
            if self.drawing and self.flag == 0:
                self.configure_pen(scene_info)
                self.pathItem.setPen(self.my_pen)
                self.pathItem.setPath(self.path)
                self.addItem(self.pathItem)

            elif self.flag == 1:
                self.my_pen = QPen(color, size)
                self.my_pen.setStyle(style)
                self.my_pen.setDashPattern(scene_info["pattern"])
                self.path.moveTo(self.path.currentPosition())
                self.path.lineTo(point)
                self.pathItem.setPen(self.my_pen)
                self.pathItem.setPath(self.path)
                self.addItem(self.pathItem)

        elif scene_info["event"] == "mouseReleaseEvent":
            self.drawing = False
            self.my_pen = None
            self.path = None
            self.pathItem = None
            self.flag = 1


    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)

        if self.selection_mode and self.selection_start is not None and self.selection_end is not None:
            pen = QPen(Qt.black, 0.8, Qt.DotLine)
            painter.setPen(pen)

            selection_rect = QRectF(self.selection_start, self.selection_end).normalized()
            painter.drawRect(selection_rect)


    def get_z_index_range(self):
        highest_z = float("-inf")
        lowest_z = float("inf")
        for item in self.items():
            highest_z = max(highest_z, item.zValue())
            lowest_z = min(lowest_z, item.zValue())
        return highest_z, lowest_z
    