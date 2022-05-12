"""
Created on 12/05/2022
@author jdh
"""

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.ptime as ptime


class Plot:

    def __init__(self, data, fps=30, name='jaxplot'):

        self.app = QtGui.QApplication([name])

        ## Create window with GraphicsView widget

        # can't create this without qapplication
        self.win = pg.GraphicsLayoutWidget()

        self.win.show()  ## show widget alone in its own window
        self.view = self.win.addViewBox()

        ## lock the aspect ratio so pixels are always square
        self.view.setAspectLocked(True)

        ## Create image item
        self.image = pg.ImageItem(border='w')
        self.view.addItem(self.image)

        # data should be array of frames with shape (number of frames, pixel x, pixel y)
        self.data = data
        self.number_of_frames = self.data.shape[0]
        self.fps = fps

        self.milliseconds_per_frame = int(1e3 / self.fps)

        # starting frame
        self.i = 0

        self.run_live()
        self.app.exec_()

    def run_live(self):

        # set the data frame to be the image
        self.image.setImage(self.data[self.i])

        # make it loop through data
        self.i = (self.i + 1) % self.number_of_frames

        QtCore.QTimer.singleShot(
            self.milliseconds_per_frame,
            self.run_live
        )






