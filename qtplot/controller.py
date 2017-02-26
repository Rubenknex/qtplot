import sys

from PyQt4 import QtGui

from .model import Model
from .view import MainView


class Controller:
    def __init__(self):
        self.model = Model()

        self.main_view = MainView()
        # LinecutView etc

        self.setup_view_to_controller()
        self.setup_model_to_controller()

        self.main_view.show()

    def setup_view_to_controller(self):
        self.main_view.b_load.clicked.connect(self.on_load)

    def setup_model_to_controller(self):
        self.model.data_file_changed.connect(self.on_data_file_changed)
        self.model.data2d_changed.connect(self.on_data2d_changed)

    def on_load(self):
        #open_directory = self.profile_settings['open_directory']
        #filename = str(QtGui.QFileDialog.getOpenFileName(directory=open_directory,
        #                                                 filter='*.dat'))

        filename = str(QtGui.QFileDialog.getOpenFileName(filter='*.dat'))

        self.model.load_data_file(filename)

    def on_data_file_changed(self):
        for cb in [self.main_view.cb_x, self.main_view.cb_y, self.main_view.cb_z]:
            cb.clear()
            cb.addItems(self.model.data_file.ids)
            # set index

    def on_data2d_changed(self):
        print('data changed!')


def main():
    app = QtGui.QApplication(sys.argv)

    if len(sys.argv) > 1:
        c = Controller(filename=sys.argv[1])
    else:
        c = Controller()

    sys.exit(app.exec_())
