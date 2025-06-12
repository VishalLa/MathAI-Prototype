import os
import sys 
import cv2 
import torch 
import numpy as np

from Whiteboard.Server.TCPServer.mainwindowtcp import MainWindow

from PySide6.QtWidgets import QApplication

from CNN_Model.Covonutional_neural_network.CNNnetwork import CNN

from CNN_Model.Covonutional_neural_network.ViTnetwork import ViT
from CNN_Model.Covonutional_neural_network.modelUttils.model_utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# network = CNN().to(device)

network = ViT(
    img_size=64,
    patch_size=8,
    in_channels=1,
    num_classes=7,
    emb_size=384,
    depth=8,
    n_heads=12,
    mlp_dim=764
).to(device)

model_path = "C:\\Users\\visha\\OneDrive\\Desktop\\MathAI\\CNN_Model\\model_parameters_for_ViT.pth"

try:
    model = load_model(network, model_path).to(device)
except Exception as e:
    print(f"Error while loading the model: {e}")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    app = QApplication()
    win = MainWindow(model)
    win.show()
    app.exec()

if __name__=='__main__':
    main()
