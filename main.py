import os
import sys 
import cv2 
import torch 
import numpy as np

from Whiteboard.UI.mainwindow import MainWindow

from PySide6.QtWidgets import QApplication

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    app = QApplication()
    win = MainWindow()
    win.show()
    app.exec()

if __name__=='__main__':
    main()
