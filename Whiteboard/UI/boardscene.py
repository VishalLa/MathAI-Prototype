import numpy as np

from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsPathItem,
)
from PySide6.QtGui import (
    QPen,
    QPainter,
    QPainterPath,
    QColor,
    QBrush,
    QImage
)
from PySide6.QtCore import Qt, QRectF

class BoardScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.setSceneRect(0, 0, 600, 500)

        self.setBackgroundBrush(QBrush(QColor("#FFFFFF")))

        self.path = None
        self.previous_position = None
        self.drawing = False
        self.selection_mode = False
        
        self.color = QColor("#000000")
        self.size = 5
        self.pathItem = None

        self.selected_area = None
        self.selection_path = QPainterPath()


    def change_color(self, color):
        self.color = color


    def change_size(self, size):
        self.size = size


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.selection_mode:
                self.drawing = False 
                self.selection_start = event.scenePos()
                self.selection_path.moveTo(self.selection_start)
            else:
                self.start_drawing(event.scenePos())


    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self.selection_mode and hasattr(self, 'selection_start'):
                self.selection_end = event.scenePos()
                self.update()
            elif self.drawing and self.pathItem:
                self.update_drawing(event.scenePos())


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.selection_mode and self.selection_path:
                self.update()
            else:
                self.end_drawing()


    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if self.selection_mode and self.selection_path and hasattr(self, 'selection_start') and hasattr(self, 'selection_end'):
            pen = QPen(Qt.black)
            pen.setStyle(Qt.DotLine)
            pen.setWidth(2)
            painter.setPen(pen)

            rect = QRectF(self.selection_start, self.selection_end)
            painter.drawRect(rect)


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

    
    def update_drawing(self, pos):
        self.path.lineTo(pos)
        self.pathItem.setPath(self.path)


    def end_drawing(self):
        self.drawing = False 
        self.path = None 
        self.previous_position = None 


    def enable_selection_mode(self, enable):
        self.selection_mode = enable 
        for item in self.items():
            item.setFlag(QGraphicsItem.ItemIsSelectable, enable)
            item.setFlag(QGraphicsItem.ItemIsMovable, enable)


    def get_selected_region(self):
        if not hasattr(self, 'selection_start') or not hasattr(self, 'selection_end'):
            return None 
        
        rect = QRectF(self.selection_start, self.selection_end).normalized()

        img = QImage(int(rect.width()), int(rect.height()), QImage.Format.Format_RGB32)
        img.fill(Qt.white)

        painter = QPainter(img)
        self.render(painter, target=QRectF(img.rect()), source=rect) 
        painter.end()

        buffer = img.bits()
        arr = np.array(buffer, dtype=np.uint8).reshape((img.height(), img.width(), 4))

        return arr 
    
