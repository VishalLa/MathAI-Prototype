from PySide6.QtCore import QCoreApplication, QTimer, Signal, QDataStream
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPainterPath
from PySide6.QtNetwork import QTcpServer, QTcpSocket, QHostAddress, QAbstractSocket
import json

from netManage import SignalManager
from getip import get_local_ip, get_ipv6_address

signal_manager = SignalManager()


class MyServer(QTcpServer):
    pass

