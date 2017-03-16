import os
from PyQt4 import QtCore, QtGui, uic


class Settings(QtGui.QDialog):
    def __init__(self, parent, model):
        super(Settings, self).__init__(parent)

        self.model = model

        path = os.path.join(self.model.dir, 'ui/settings.ui')
        uic.loadUi(path, self)

        self.bind()

    def initialize(self):
        self.lw_profiles.addItems(os.listdir(self.model.profiles_dir))

    def bind(self):
        self.b_add.clicked.connect(self.on_add)

    def on_add(self):
        name = str(self.le_name.text())

        if (name != '' and
            len(self.lw_profiles.findItems(name, QtCore.Qt.MatchExactly)) == 0):
            item = QtGui.QListWidgetItem(name + '.json')

            self.lw_profiles.addItem(item)
            self.lw_profiles.setCurrentItem(item)

            self.le_name.setText('')

            self.cb_default.addItem(name)

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
