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
        # Add the files in .qtplot/profiles/ to the interface
        self.cb_default.addItems(os.listdir(self.model.profiles_dir))
        self.lw_profiles.addItems(os.listdir(self.model.profiles_dir))

        default = self.model.settings['default_profile']

        idx = self.cb_default.findText(default)
        if idx != -1:
            self.cb_default.setCurrentIndex(idx)

        items = self.lw_profiles.findItems(default, QtCore.Qt.MatchExactly)
        if len(items) > 0:
            self.lw_profiles.setCurrentItem(items[0])

    def bind(self):
        """ Connect to the GUI and Model events """
        self.cb_default.activated.connect(self.on_default_changed)
        self.lw_profiles.itemClicked.connect(self.on_item_changed)

        self.b_add.clicked.connect(self.on_add)
        self.b_remove.clicked.connect(self.on_remove)
        self.b_save.clicked.connect(self.on_save)
        self.b_browse_open.clicked.connect(self.on_browse_open)
        self.b_browse_save.clicked.connect(self.on_browse_save)

        self.model.data_file_changed.connect(self.populate_settings_tree)
        self.model.profile_changed.connect(self.on_profile_changed)

    def get_profile(self):
        state = {
            'open_directory': str(self.le_open_dir.currentText()),
            'save_directory': str(self.le_save_dir.currentText()),
        }

        return state

    def on_default_changed(self, index):
        """ The selected default profile changed """
        if index != -1:
            name = str(self.cb_default.currentText())

            self.model.settings['default_profile'] = name
            self.model.save_settings()

    def on_item_changed(self, item):
        """ The selected profile changed """
        if item is not None:
            name = str(item.text())

            self.model.load_profile(name)

    def on_add(self):
        """ Add a profile to the list """
        name = str(self.le_name.text()) + '.json'

        found = self.lw_profiles.findItems(name, QtCore.Qt.MatchExactly)

        if name != '' and len(found) == 0:
            item = QtGui.QListWidgetItem(name)

            self.lw_profiles.addItem(item)
            self.lw_profiles.setCurrentItem(item)

            self.le_name.setText('')
            self.cb_default.addItem(name)

            self.on_save()

    def on_remove(self):
        """ Remove a profile from the list """
        if self.lw_profiles.currentRow != -1:
            name = str(self.lw_profiles.currentItem().text())

            path = os.path.join(self.model.profiles_dir, name)

            if os.path.exists(path):
                os.remove(path)

            self.lw_profiles.takeItem(self.lw_profiles.currentRow())

    def on_save(self):
        """ Save the current state of qtplot to a profile file """
        if self.lw_profiles.currentRow() != -1:
            profile = self.parent().get_profile()
            name = str(self.lw_profiles.currentItem().text())

            self.model.save_profile(profile, name)

    def on_browse_open(self):
        """ Select a directory used for file loading """
        directory = QtGui.QFileDialog.getExistingDirectory(self,
                                                           "Select Directory")
        directory = str(directory)

        if directory != '':
            self.le_open_dir.setText(directory)

            self.model.profile['open_directory'] = directory

    def on_browse_save(self):
        """ Select a directory used for file saving """
        directory = QtGui.QFileDialog.getExistingDirectory(self,
                                                           "Select Directory")
        directory = str(directory)

        if directory != '':
            self.le_save_dir.setText(directory)

            self.model.profile['save_directory'] = directory

    def on_profile_changed(self, profile):
        """ Connection to Model.profile_changed event """
        self.le_open_dir.setText(profile['open_directory'])
        self.le_save_dir.setText(profile['save_directory'])

    def populate_settings_tree(self, different_file):
        if self.model.data_file is not None:
            settings = self.model.data_file.qtlab_settings
            widgets = []

            for key, item in settings.items():
                if isinstance(item, dict):
                    parent = QtGui.QTreeWidgetItem(None, [key, ''])

                    for key, item in item.items():
                        child = QtGui.QTreeWidgetItem(parent, [key, item])
                        child.setCheckState(0, QtCore.Qt.Unchecked)
                else:
                    parent = QtGui.QTreeWidgetItem(None, [key, item])

                parent.setCheckState(0, QtCore.Qt.Unchecked)
                widgets.append(parent)

            self.tw_settings.clear()
            self.tw_settings.insertTopLevelItems(0, widgets)

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
