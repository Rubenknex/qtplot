from PyQt4 import QtGui, QtCore
import os


class Settings(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Settings, self).__init__(parent)

        self.main = parent

        self.create_ui()
        #self.populate_ui()

    def create_ui(self):
        self.setWindowTitle("Settings")

        # Main vbox
        vbl = QtGui.QVBoxLayout()

        # Profile groupbox
        gb_profile = QtGui.QGroupBox('Profile')
        vbl.addWidget(gb_profile)

        # Profile vbox
        vbl_profile = QtGui.QVBoxLayout()
        gb_profile.setLayout(vbl_profile)

        # Profile grid
        hbl = QtGui.QHBoxLayout()
        vbl_profile.addLayout(hbl)
        lbl = QtGui.QLabel('Default (used on startup):', self)
        #lbl_default.setMaximumWidth(10)
        hbl.addWidget(lbl)

        self.cb_default_profile = QtGui.QComboBox(self)
        self.cb_default_profile.activated.connect(self.on_default_profile_changed)
        hbl.addWidget(self.cb_default_profile)

        #lbl = QtGui.QLabel('Current:', self)
        #lbl_default.setMaximumWidth(10)
        #hbl.addWidget(lbl, 2, 1)

        self.lw_profiles = QtGui.QListWidget(self)
        self.lw_profiles.currentItemChanged.connect(self.on_profile_changed)
        vbl_profile.addWidget(self.lw_profiles)

        hbl_list = QtGui.QHBoxLayout()
        vbl_profile.addLayout(hbl_list)

        self.le_profile = QtGui.QLineEdit(self)
        #self.le_profile.setMaximumWidth(100)
        #self.le_profile.returnPressed.connect(self.on_min_max_entered)
        hbl_list.addWidget(self.le_profile)

        self.b_add = QtGui.QPushButton('Add', self)
        self.b_add.clicked.connect(self.on_add)
        #self.b_add.setMaximumWidth(50)
        hbl_list.addWidget(self.b_add)

        self.b_remove = QtGui.QPushButton('Remove', self)
        self.b_remove.clicked.connect(self.on_remove)
        #self.b_remove.setMaximumWidth(50)
        hbl_list.addWidget(self.b_remove)

        self.b_save_state = QtGui.QPushButton('Save current state', self)
        self.b_save_state.clicked.connect(self.on_save_state)
        #self.b_add.setMaximumWidth(50)
        hbl_list.addWidget(self.b_save_state)

        #"""
        # Open directory
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(QtGui.QLabel('Open directory:'))

        self.le_open_directory = QtGui.QLineEdit(self)
        self.le_open_directory.setEnabled(False)
        #self.le_open_directory.setText(self.main.open_directory)
        hbox.addWidget(self.le_open_directory)

        self.b_browse = QtGui.QPushButton('Browse...')
        self.b_browse.clicked.connect(self.on_open_browse)
        hbox.addWidget(self.b_browse)
        vbl_profile.addLayout(hbox)

        # Save directory
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(QtGui.QLabel('Save directory:'))

        self.le_save_directory = QtGui.QLineEdit(self)
        self.le_save_directory.setEnabled(False)
        #self.le_save_directory.setText(self.main.save_directory)
        hbox.addWidget(self.le_save_directory)

        self.b_browse = QtGui.QPushButton('Browse...')
        self.b_browse.clicked.connect(self.on_save_browse)
        hbox.addWidget(self.b_browse)
        vbl_profile.addLayout(hbox)
        #"""

        # QTLab .set file tree view
        self.tree = QtGui.QTreeWidget(self)
        self.tree.setHeaderLabels(['Name', 'Value'])
        self.tree.setColumnWidth(0, 200)
        self.tree.itemClicked.connect(self.on_item_changed)

        self.b_copy = QtGui.QPushButton('Copy')
        self.b_copy.clicked.connect(self.on_copy)

        vbl.addWidget(self.tree)
        vbl.addWidget(self.b_copy)

        self.setLayout(vbl)

        self.setGeometry(900, 300, 400, 500)

    def populate_ui(self):
        profile_files = []
        for file in os.listdir(self.main.profiles_dir):
            if file.endswith('.ini'):
                profile_files.append(file)

        current_profile = os.path.split(self.main.profile_ini_file)[-1]

        # Set selected default profile
        self.cb_default_profile.addItems(profile_files)
        i = self.cb_default_profile.findText(current_profile)
        self.cb_default_profile.setCurrentIndex(i)

        # Set selected profile
        self.lw_profiles.addItems(profile_files)
        a = self.lw_profiles.findItems(current_profile, QtCore.Qt.MatchExactly)
        if len(a) > 0:
            self.lw_profiles.setCurrentItem(a[0])

        self.le_open_directory.setText(self.main.profile_settings['open_directory'])
        self.le_save_directory.setText(self.main.profile_settings['save_directory'])

    def fill_tree(self):
        """
        if self.main.dat_file is not None:
            settings = self.main.dat_file.qtlab_settings
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

            self.tree.insertTopLevelItems(0, widgets)
        """

    def on_open_browse(self, event):
        directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))

        if directory != '':
            self.le_open_directory.setText(directory)

            self.main.profile_settings['open_directory'] = directory

    def on_save_browse(self, event):
        directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))

        if directory != '':
            self.le_save_directory.setText(directory)

            self.main.profile_settings['save_directory'] = directory

    def on_default_profile_changed(self, event):
        file = str(self.cb_default_profile.currentText())

        self.main.save_default_profile(file)

    def on_profile_changed(self, event):
        filename = str(self.lw_profiles.currentItem().text())

        self.main.open_state(filename)

        #self.main.profile_ini_file = os.path.join(self.main.profiles_dir, file)

    def on_add(self, event):
        name = str(self.le_profile.text()) + '.ini'

        if (name == '.ini' or
           len(self.lw_profiles.findItems(name, QtCore.Qt.MatchExactly)) != 0):
            return

        item = QtGui.QListWidgetItem(name)
        self.lw_profiles.addItem(item)
        self.lw_profiles.setCurrentItem(item)

        self.le_profile.setText('')

        self.cb_default_profile.addItem(name)

    def on_remove(self, event):
        file = str(self.lw_profiles.currentItem().text())

        path = os.path.join(self.main.profiles_dir, file)

        if os.path.exists(path):
            os.remove(path)

        self.lw_profiles.takeItem(self.lw_profiles.currentRow())

    def on_save_state(self, event):
        file = str(self.lw_profiles.currentItem().text())
        self.main.save_state(file)

    def on_item_changed(self, widget):
        state = widget.checkState(0)

        for i in range(widget.childCount()):
            child = widget.child(i)
            child.setCheckState(0, state)

        if widget.childCount() == 0 and state == QtCore.Qt.Checked and widget.parent() != None:
            widget.parent().setCheckState(0, state)

    def on_copy(self):
        text = ''

        root = self.tree.invisibleRootItem()
        count = root.childCount()

        for i in range(count):
            parent = root.child(i)

            header = False

            if parent.checkState(0) == QtCore.Qt.Checked:
                text += str(parent.text(0)) + ': ' + str(parent.text(1)) + '\n'
                header = True

            for j in range(parent.childCount()):
                child = parent.child(j)

                if child.checkState(0) == QtCore.Qt.Checked:
                    if not header:
                        text += str(parent.text(0)) + ': ' + str(parent.text(1)) + '\n'
                        header = True

                    text += '  ' + str(child.text(0)) + ': ' + str(child.text(1)) + '\n'

        QtGui.QApplication.clipboard().setText(text)

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
