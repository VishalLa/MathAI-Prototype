from PySide6.QtWidgets import QGraphicsView
from PySide6.QtCore import Qt 
from PySide6.QtGui import QWheelEvent


class ZoomableGraphicView(QGraphicsView):
    def __init__(self, *args, zoom_factor=1.15, **kwargs):
        super().__init__(*args, **kwargs) 
        self._zoom = 0
        self._zoom_factor = zoom_factor 
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)



    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() == Qt.ControlModifier:
            old_pos = self.mapToScene(event.position().toPoint())

            if event.angleDelta().y() > 0:
                zoom = self._zoom_factor
                self._zoom += 1
            else:
                zoom = 1 / self._zoom_factor
                self._zoom -= 1

            self.scale(zoom, zoom)
            new_pos = self.mapToScene(event.position().toPoint())
            delta = new_pos - old_pos 
            self.translate(delta.x(), delta.y())
        
        else:
            super().wheelEvent(event)
