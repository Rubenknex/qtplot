from PyQt4 import QtGui, QtCore
import os


class Settings(QtGui.QDialog):
    def __init__(self, main, parent=None):
        super(Settings, self).__init__(parent)

        self.main = main

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Settings")

        vbox = QtGui.QVBoxLayout()

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(QtGui.QLabel('Open directory:'))

        self.le_open_directory = QtGui.QLineEdit(self)
        self.le_open_directory.setEnabled(False)
        self.le_open_directory.setText(self.main.open_directory)
        hbox.addWidget(self.le_open_directory)

        self.b_browse = QtGui.QPushButton('Browse...')
        self.b_browse.clicked.connect(self.on_open_browse)
        hbox.addWidget(self.b_browse)
        vbox.addLayout(hbox)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(QtGui.QLabel('Save directory:'))

        self.le_save_directory = QtGui.QLineEdit(self)
        self.le_save_directory.setEnabled(False)
        self.le_save_directory.setText(self.main.save_directory)
        hbox.addWidget(self.le_save_directory)

        self.b_browse = QtGui.QPushButton('Browse...')
        self.b_browse.clicked.connect(self.on_save_browse)
        hbox.addWidget(self.b_browse)
        vbox.addLayout(hbox)


        self.tree = QtGui.QTreeWidget(self)
        self.tree.setHeaderLabels(['Name', 'Value'])
        self.tree.setColumnWidth(0, 200)
        self.tree.itemClicked.connect(self.on_item_changed)

        self.b_copy = QtGui.QPushButton('Copy')
        self.b_copy.clicked.connect(self.on_copy)

        vbox.addWidget(self.tree)
        vbox.addWidget(self.b_copy)

        layout = QtGui.QVBoxLayout()
        layout.addLayout(vbox)
        self.setLayout(layout)

        self.setGeometry(900, 300, 400, 500)

    def load_file(self, filename):
        path, ext = os.path.splitext(filename)
        settings_file = path + '.set'
        settings_file_name = os.path.split(settings_file)[1]

        if os.path.exists(settings_file):
            with open(settings_file) as f:
                self.lines = f.readlines()

            self.fill_tree(self.lines)

            self.setWindowTitle(settings_file_name)
        else:
            self.setWindowTitle('Could not find ' + settings_file_name)

    def fill_tree(self, lines):
        self.tree.clear()

        widgets = []

        for line in lines:
            line = line.rstrip('\n\t\r')

            if line == '':
                continue

            if not line.startswith('\t'):
                name, value = line.split(': ', 1)

                parent = QtGui.QTreeWidgetItem(None, [name, value])
                parent.setCheckState(0, QtCore.Qt.Unchecked)
                widgets.append(parent)
            else:
                name, value = line.split(': ', 1)

                child = QtGui.QTreeWidgetItem(parent, [name.strip(), value])
                child.setCheckState(0, QtCore.Qt.Unchecked)

        self.tree.insertTopLevelItems(0, widgets)

    def on_open_browse(self, event):
        directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))

        if directory != '':
            self.le_open_directory.setText(directory)

            self.main.default_open_directory = directory
            self.main.write_to_ini('Settings', {'OpenDirectory': directory})

    def on_save_browse(self, event):
        directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))

        if directory != '':
            self.le_save_directory.setText(directory)

            self.main.default_save_directory = directory
            self.main.write_to_ini('Settings', {'SaveDirectory': directory})

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
