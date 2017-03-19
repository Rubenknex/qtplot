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
        self.lw_profiles.currentItemChanged.connect(self.on_profile_changed)

        self.b_add.clicked.connect(self.on_add)
        self.b_remove.clicked.connect(self.on_remove)
        self.b_save.clicked.connect(self.on_save)

    def on_profile_changed(self):
        if self.lw_profiles.currentRow() != -1:
            name = str(self.lw_profiles.currentItem().text())

            self.model.load_profile(name)

    def on_add(self):
        name = str(self.le_name.text()) + '.json'

        if (name != '' and
            len(self.lw_profiles.findItems(name, QtCore.Qt.MatchExactly)) == 0):
            item = QtGui.QListWidgetItem(name)

            self.lw_profiles.addItem(item)
            self.le_name.setText('')
            self.cb_default.addItem(name)

            #self.on_save()

            #self.lw_profiles.setCurrentItem(item)

    def on_remove(self):
        if self.lw_profiles.currentRow != -1:
            name = str(self.lw_profiles.currentItem().text())

            path = os.path.join(self.model.profiles_dir, name)

            if os.path.exists(path):
                os.remove(path)

            self.lw_profiles.takeItem(self.lw_profiles.currentRow())

    def on_save(self):
        if self.lw_profiles.currentRow != -1:
            profile = self.parent().get_profile()
            name = str(self.lw_profiles.currentItem().text())

            self.model.save_profile(profile, name)

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
