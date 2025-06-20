from PySide6.QtWidgets import QButtonGroup
from PySide6.QtGui import QColor

class ToolManager:
    def __init__(self, main_window):
        self.main = main_window
        self.tool_group = QButtonGroup()
        self.tool_group.setExclusive(True)

        # Add buttons
        self.tool_group.addButton(self.main.pb_Pen)
        self.tool_group.addButton(self.main.pb_Eraser)
        self.tool_group.addButton(self.main.pb_Select)
        self.tool_group.addButton(self.main.pb_Clear)
        self.tool_group.addButton(self.main.pb_Predict)

        self.main.pb_Pen.setChecked(True)
        self.main.pb_Pen.clicked.connect(self.pen_selected)
        self.main.pb_Eraser.clicked.connect(self.eraser_selected)
        self.main.pb_Select.clicked.connect(self.select_mode)
        self.main.pb_Clear.clicked.connect(self.clear_canvas)

    def pen_selected(self):
        self.main.scene.enable_selection_mode(False)
        self.main.scene.change_color(self.main.current_color)

    def eraser_selected(self):
        self.main.scene.enable_selection_mode(False)
        self.main.scene.change_color(QColor("#FFFFFF"))

    def select_mode(self):
        self.main.scene.enable_selection_mode(True)

    def clear_canvas(self):
        self.main.scene.clear()
