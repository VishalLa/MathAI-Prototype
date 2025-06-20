from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QGraphicsLineItem
)

from PySide6.QtGui import (
    QPainter,
    QPen, 
    QColor, 
    QTransform,
    QPixmap
)

from PySide6.QtCore import (
    Qt, 
    QPointF, 
    QLineF,
    QRectF,
    Signal
)


class CustomGraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setBackgroundBrush(Qt.white)
        self.setSceneRect(-5000, -5000, 10000, 10000)

        self.drawing = False 
        self.last_point = QPointF()
        self.pen_color = QColor(Qt.black)
        self.pen_width = 5
        self.drawing_mode = 'pen' # 'pen', 'eraser', 'select'

        # For undo/redo
        self.history = []
        self.history_index = -1
        self.add_to_history()

    
    def set_pen_color(self, color):
        self.pen_color = color

    
    def set_pen_width(self, width):
        self.pen_width = width


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.scenePos()

        super().mousePressEvent(event)


    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drawing:
            if self.drawing_mode == 'pen':
                pen = QPen(
                    self.pen_color,
                    self.pen_width,
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin
                )

                self.addLine(
                    self.last_point.x(),
                    self.last_point.y(),
                    event.scenePos().x(),
                    event.scenePod().y(),
                    pen
                )

            elif self.drawing_mode == 'eraser':
                eraser_pen = QPen(
                    self.backgroundBrush().color(),
                    self.pen_width + 5,
                    Qt.SolidLine,
                    Qt.RoundCap,
                    Qt.RoundJoin
                )

                self.addLine(
                    self.last_point.x(),
                    self.last_point.y(),
                    event.scenePos().x(),
                    event.scenePod().y(),
                    eraser_pen
                )
            
            self.last_point = event.scenePos()

        super().mouseMoveEvent(event)


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False 
            self.add_to_history()

        super().mouseReleaseEvent(event)


    def add_to_history(self):
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
            
        current_item_data = []
        for item in self.items():
            current_item_data.append(
                (item.line().p1().x(), item.line().p1().y(),
                item.line().p2().x(), item.line().p2().y(),
                item.pen().color().name(), item.pen().width())
            )

        self.history.append(current_item_data)
        self.history_index += 1


    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.restore_from_history()
        else:
            print('Cannot Undo further.')


    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.restore_from_history()
        else:
            print("Cannot Redo further.")


    def restore_from_history(self):
        self.clear()
        if self.history_index >= 0:
            items_data = self.history[self.history_index]
            for x1, y1, x2, y2, color_name, width in items_data:
                pen = QPen(
                    QColor(color_name), 
                    width, 
                    Qt.SolidLine,
                    Qt.RoundCap, 
                    Qt.RoundJoin
                )
                line = QLineF(x1, y1, x2, y2)
                self.addLine(line, pen)


    def clear_canvas(self):
        self.clear()
        self.history = []
        self.history_index = -1
        self.add_to_history()
        print("Canvas Cleared and History Reset.")


    def render_area_to_pixmap(self, rect: QRectF) -> QPixmap:
        """Renders a specific area of the scene to a QPixmap."""
        pixmap = QPixmap(rect.size().toSize())
        pixmap.fill(self.backgroundBrush().color()) # Fill with scene background
        painter = QPainter(pixmap)
        self.render(painter, pixmap.rect(), rect) # Render sceneRect into pixmapRect
        painter.end()

        return pixmap
    

    def serialize(self) -> list:
        """Serializes the current state of the scene's drawable items."""
        data = []
        for item in self.items():
            if isinstance(item, QGraphicsLineItem):
                line = item.line()
                pen = item.pen()
                data.append({
                    "type": "line",
                    "p1_x": line.p1().x(),
                    "p1_y": line.p1().y(),
                    "p2_x": line.p2().x(),
                    "p2_y": line.p2().y(),
                    "color": pen.color().name(),
                    "width": pen.width()
                })
        return data


    def deserialize(self, data: list):
        """Deserializes data and restores the scene's drawable items."""
        self.clear_canvas() # Clear current content and history
        for item_data in data:
            item_type = item_data.get("type")
            if item_type == "line":
                x1 = item_data.get("p1_x")
                y1 = item_data.get("p1_y")
                x2 = item_data.get("p2_x")
                y2 = item_data.get("p2_y")
                color_name = item_data.get("color")
                width = item_data.get("width")

                if all(v is not None for v in [x1, y1, x2, y2, color_name, width]):
                    pen = QPen(
                        QColor(color_name),
                        width,
                        Qt.SolidLine, 
                        Qt.RoundCap, 
                        Qt.RoundJoin
                        )
                    line = QLineF(x1, y1, x2, y2)
                    self.addLine(line, pen)
        self.add_to_history() # Add the loaded state to history
        print("Canvas loaded from file.")



class CustomGraphicsView(QGraphicsView):
    rubber_band_selection_finished = Signal(QRectF)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMinimumSize(400, 300)

        self._rubber_band_start = QPointF()
        self._rubber_band_end = QPointF()
        self._is_dragging_rubber_band = True


    def mousePressEvent(self, event):
        if self.dragMode() == QGraphicsView.RubberBandDrag and event.button() == Qt.LeftButton:
            self._rubber_band_start = self.mapToScene(event.pos())
            self._is_dragging_rubber_band = True 

        super().mousePressEvent(event)

    
    def mouseReleaseEvent(self, event):
        if self._is_dragging_rubber_band and event.button() == Qt.LeftButton:
            self._rubber_band_end = self.mapToScene(event.pos())
            rubber_band_rect = QRectF(
                self._rubber_band_start,
                self._rubber_band_end
            ).normalized()

            if not rubber_band_rect.isNull() and not rubber_band_rect.isEmpty():
                self.rubber_band_selection_finished.emit(rubber_band_rect)

            self._is_dragging_rubber_band = False 

        super().mouseReleaseEvent(event)


    def wheelEvent(self, event):
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1/zoom_factor, 1/zoom_factor)
