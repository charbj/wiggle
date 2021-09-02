# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

""" Import all dependencies """
from chimerax.core.tools import ToolInstance
from chimerax.core import logger
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
from functools import partial, lru_cache
import os.path
import time
import subprocess

class Wiggle(ToolInstance):

    SESSION_ENDURING = False    # Does this instance persist when session closes
    SESSION_SAVE = True         # We do save/restore in sessions
    help = "help:user/tools/tutorial.html"
                                # Let ChimeraX know about our help page
    def __init__(self, session, tool_name):
        # 'session'   - chimerax.core.session.Session instance
        # 'tool_name' - string

        # Initialize base class.
        super().__init__(session, tool_name)

        self.display_name = "WIGGLE - Widget for Interactive and Graphics Guided Landscape Exploration"

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        self.tool_window.fill_context_menu = self.fill_context_menu

        from .deps.wiggle import Main
        ui = Main(session)
        layout = ui.setup()

        # Set the layout as the contents of our window
        self.tool_window.ui_area.setLayout(layout)

        # Show the window on the user-preferred side of the ChimeraX
        # main window
        self.tool_window.manage(None, fixed_size=True)

    def fill_context_menu(self, menu, x, y):
        # Add any tool-specific items to the given context menu (a QMenu instance).
        # The menu will then be automatically filled out with generic tool-related actions
        # (e.g. Hide Tool, Help, Dockable Tool, etc.)
        #
        # The x,y args are the x() and y() values of QContextMenuEvent, in the rare case
        # where the items put in the menu depends on where in the tool interface the menu
        # was raised.
        from Qt.QtWidgets import QAction
        clear_action = QAction("Clear", menu)
        clear_action.triggered.connect(lambda *args: self.line_edit.clear())
        menu.addAction(clear_action)

    def take_snapshot(self, session, flags):
        return {
            'version': 1,
            'current text': self.line_edit.text()
        }

    @classmethod
    def restore_snapshot(class_obj, session, data):
        # Instead of using a fixed string when calling the constructor below, we could
        # have saved the tool name during take_snapshot() (from self.tool_name, inherited
        # from ToolInstance) and used that saved tool name.  There are pros and cons to
        # both approaches.
        inst = class_obj(session, "Wiggle")
        inst.line_edit.setText(data['current text'])
        return inst


#Orphan code / outdated

# ''' Set global parameters '''
# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')
# lastClicked = []

# class WorkerOld(QtCore.QObject):
#     finished = QtCore.pyqtSignal()
#     progress = QtCore.pyqtSignal(float, float, bool, bool, bool)
#     msg = QtCore.pyqtSignal(str)
#     status = QtCore.pyqtSignal(bool)
#
#     def load_data(self, path):
#         if os.path.isfile(path):
#             d = np.load(path, allow_pickle=True)
#             return d
#         else:
#             return np.random.randint(50, size=(64, 64))
#
#     def run_clustering(self):
#         self.status.emit(False)
#         '''
#         self.comboBox_4.setItemText(0, _translate("MainWindow", "KMeans (fast)"))
#         self.comboBox_4.setItemText(1, _translate("MainWindow", "Affinity Propagation (slow)"))
#         self.comboBox_4.setItemText(2, _translate("MainWindow", "MeanShift (slow)"))
#         self.comboBox_4.setItemText(3, _translate("MainWindow", "Spectral Clustering (slow)"))
#         self.comboBox_4.setItemText(4, _translate("MainWindow", "Ward (fast)"))
#         self.comboBox_4.setItemText(5, _translate("MainWindow", "Agglomerative Clustering (fast)"))
#         self.comboBox_4.setItemText(6, _translate("MainWindow", "DBSCAN (fast)"))
#         self.comboBox_4.setItemText(7, _translate("MainWindow", "OPTICS (slow)"))
#         self.comboBox_4.setItemText(8, _translate("MainWindow", "BIRCH (fast)"))
#         self.comboBox_4.setItemText(9, _translate("MainWindow", "Gaussian Mixture (fast)"))
#         '''
#         global labels
#
#         from sklearn import cluster, mixture
#         from sklearn.neighbors import kneighbors_graph
#         self.msg.emit('Begin clustering ...')
#
#         global data
#         data = self.load_data(ui.lineEdit_4.text())
#         clusters = ui.spinBox_2.value()
#
#         #Subset for ETA calculation - at the expense of a small amount of time
#         fraction = 0.01
#         subset = int(len(data) * fraction)
#         idx = np.random.randint(data.shape[0], size=subset)
#         small_data = data[idx]
#
#
#         # ============
#         # Create cluster objects from sklearn clustering example
#         # ============
#
#         # connectivity matrix for structured Ward
#         if ui.comboBox_4.currentIndex() == 4 or ui.comboBox_4.currentIndex() == 5:
#             connectivity = kneighbors_graph(data, n_neighbors=5, include_self=False)
#             # make connectivity symmetric
#             connectivity = 0.5 * (connectivity + connectivity.T)
#         else:
#             connectivity = []
#
#         # estimate bandwidth for mean shift
#         if ui.comboBox_4.currentIndex() == 2:
#             bandwidth = cluster.estimate_bandwidth(data, quantile=0.3)
#         else:
#             bandwidth = []
#
#
#         ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
#
#         two_means = cluster.MiniBatchKMeans(n_clusters=clusters)
#
#         ward = cluster.AgglomerativeClustering(n_clusters=clusters, linkage='ward',
#             connectivity=connectivity)
#
#         spectral = cluster.SpectralClustering(n_clusters=clusters, eigen_solver='arpack',
#             affinity="nearest_neighbors")
#
#         dbscan = cluster.DBSCAN(eps=0.5)
#
#         optics = cluster.OPTICS(min_samples=20, xi=0.05,
#                                 min_cluster_size=0.1)
#
#         affinity_propagation = cluster.AffinityPropagation(damping=0.9,
#                                                            preference=-200)
#
#         average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",
#                                                           n_clusters=clusters, connectivity=connectivity)
#
#         birch = cluster.Birch(n_clusters=clusters)
#
#         gmm = mixture.GaussianMixture(n_components=clusters, covariance_type='full')
#
#         clustering_algorithms = (
#             ('MiniBatch KMeans', two_means),
#             ('Affinity Propagation', affinity_propagation),
#             ('MeanShift', ms),
#             ('Spectral Clustering', spectral),
#             ('Ward', ward),
#             ('Agglomerative Clustering', average_linkage),
#             ('DBSCAN', dbscan),
#             ('OPTICS', optics),
#             ('BIRCH', birch),
#             ('Gaussian Mixture', gmm)
#         )
#
#         name, operator = clustering_algorithms[ui.comboBox_4.currentIndex()]
#
#         self.msg.emit('Calculating ' + name + ' ...')
#
#         '''
#         THIS NEED FIXING...
#             - Check size of data
#             - Warn user for too large data
#             - Estimate memory requirements... ???
#             - Estimate time (??? linear, exponential, etc based on O() of algorithm)
#         '''
#         ### TIMING TEST
#         t0 = time.time()
#         #dummy = operator.fit(small_data)
#         time.sleep(1)
#         ETA = (1 / fraction) * (time.time() - t0)
#         self.progress.emit(ETA, t0, True, True, False)
#
#
#         #Fit full dataset
#         operator.fit(data)
#
#         #Generate labels list
#         if hasattr(operator, 'labels_'):
#             labels = operator.labels_.astype(int)
#         else:
#             labels = operator.predict(data)
#
#         #Finished close thread and emit complete signal
#         self.msg.emit('Finished ' + name + ' ! \n -------------------------')
#         self.progress.emit(1, 1, False, False, True)
#         self.finished.emit()
#         self.status.emit(True)
#
#     def run_embedding(self): ###Re implement as above, too much redundancy.
#         self.status.emit(False)
#         input = ui.lineEdit_4.text()
#         technique = ui.comboBox_3.currentIndex()
#         '''
#             Perform a range of dimensionality reduction analyses.
#         '''
#         global embedding
#         fraction = 0.005
#         global data
#         data = self.load_data(input)
#         subset = int(len(data)*fraction)
#         idx = np.random.randint(data.shape[0], size=subset)
#         small_data = data[idx]
#
#         if technique == 0:
#             self.msg.emit("Running umap... ")
#             import umap
#             operator = umap.UMAP(random_state=42, verbose=1)
#
#             ### TIMING TEST
#             t0 = time.time()
#             dummy = operator.fit_transform(small_data)
#             ETA = 50 * (time.time() - t0)
#             self.progress.emit(ETA, t0, True, True, False)
#             ### END TIMING TES
#
#             embedding = operator.fit_transform(data)
#             self.msg.emit("Finished umap! \n -------------------------")
#
#         if technique == 1:
#             self.msg.emit("Running PCA... ")
#             from sklearn.decomposition import PCA
#             operator = PCA(n_components=2)
#
#             ### TIMING TEST
#             t0 = time.time()
#             operator.fit(small_data)
#             ETA = (1 / fraction) * (time.time() - t0)
#             self.progress.emit(ETA, t0, True, True, False)
#             ### END TIMING TEST
#
#             operator.fit(data)
#             embedding = operator.transform(data)
#             self.msg.emit("Finished PCA! \n -------------------------")
#
#         if technique == 2:
#             self.msg.emit("Running tSNE...")
#             from sklearn.manifold import TSNE
#             operator = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1)
#
#             ### TIMING TEST
#             t0 = time.time()
#             dummy = operator.fit_transform(small_data)
#             ETA = 1.5 * (1/fraction)*(time.time() - t0)
#             self.progress.emit(ETA, t0, True, True, False)
#             ### END TIMING TEST
#
#             embedding = operator.fit_transform(data)
#             self.msg.emit("Finished tSNE! \n -------------------------")
#
#         if technique == 3:
#             self.msg.emit("Running PHATE... ")
#             QtWidgets.qApp.processEvents()
#             import phate
#             operator = phate.PHATE(n_jobs=-2)
#
#             ### TIMING TEST
#             t0 = time.time()
#             dummy = operator.fit_transform(small_data)
#             ETA = 5 * (time.time() - t0)
#             self.progress.emit(ETA, t0, True, True, False)
#             ### END TIMING TEST
#
#             embedding = operator.fit_transform(data)
#             self.msg.emit("Finished PHATE! \n -------------------------")
#
#         if technique == 4:
#             self.msg.emit("Running cVAE... ")
#             QtWidgets.qApp.processEvents()
#             from cvae import cvae
#
#             ### TIMING TEST
#             t0 = time.time()
#             operator = cvae.CompressionVAE(small_data)
#             operator.train()
#             dummy = operator.embed(small_data)
#             ETA = 5 * (time.time() - t0)
#             self.progress.emit(ETA, t0, True, True, False)
#             ### END TIMING TEST
#
#             operator = cvae.CompressionVAE(data)
#             operator.train()
#             embedding = operator.embed(data)
#             self.msg.emit("Finished cVAE! \n -------------------------")
#
#         self.progress.emit(ETA, t0, False, False, True)
#         self.finished.emit()
#         self.status.emit(True)
#
# class Ui_MainWindowOld(object):
#
#     def browse_dir(self, comment, button):
#         dirName = str(QtWidgets.QFileDialog.getExistingDirectory(None, comment, ""))
#         if dirName:
#             self.string = dirName + '/'
#             if button == self.pushButton_3:
#                 self.lineEdit.setText(self.string)
#
#     def browse_file(self, comment, button, type):
#         options = QtWidgets.QFileDialog.Options()
#         options |= QtWidgets.QFileDialog.DontUseNativeDialog
#         fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, comment, "", type, options=options)
#
#         if fileName:
#             if button == self.pushButton_3:
#                 self.lineEdit.setText(fileName)
#             elif button == self.pushButton_4:
#                 self.lineEdit_2.setText(fileName)
#             elif button == self.pushButton_6:
#                 self.lineEdit_6.setText(fileName)
#             elif button == self.pushButton_7:
#                 self.lineEdit_5.setText(fileName)
#             elif button == self.pushButton_8:
#                 self.lineEdit_4.setText(fileName)
#                 self.comboBox_3.setEnabled(True)
#
#     def status(self, kill: bool):
#         try:
#             self.state
#         except:
#             self.state = 'ying'
#
#         if self.state == 'ying':
#             self.statusBar.showMessage("Status: ")
#             self.statusBar.setStyleSheet("background-color : pink")
#             self.state = 'yang'
#             self.statusTimer.start(700)
#         elif self.state == 'yang':
#             self.statusBar.showMessage("Status: BUSY")
#             self.statusBar.setStyleSheet("background-color : pink")
#             self.state = 'ying'
#             self.statusTimer.start(700)
#         elif self.state == 'idle':
#             self.state == 'ying'
#             self.statusTimer.start(700)
#
#         if kill:
#             self.statusBar.showMessage("Status: IDLE")
#             self.statusBar.setStyleSheet("background-color: rgb(239, 239, 239);")
#             self.statusTimer.stop()
#             self.state == 'idle'
#
#     def reportProgress(self, ETA, t0, init: bool, ping: bool, kill: bool):
#         if init:
#             self.ETA = ETA
#             self.t0 = t0
#             progText = 'Running... Estimated time for completion: ' + str(self.ETA/60)[0:5] + ' minutes.'
#             self.textBrowser.append(progText)
#
#         if ping:
#             try:
#                 ETA = self.ETA
#                 t0 = self.t0
#                 if (time.time() - t0 < ETA):
#                     progress = 100 * ((time.time() - t0) / ETA)
#                     self.progressBar_7.setValue(int(progress))
#                     self.timer.start(50)
#                     QtWidgets.qApp.processEvents()
#             except:
#                 print('no good')
#
#         if kill:
#             self.progressBar_7.setValue(int(100))
#             self.timer.stop()
#
#     def launchEmbedding(self):
#         self.progressBar_7.reset()
#         self.pushButton_19.setEnabled(False)
#         # Step 2: Create a QThread object
#         self.thread = QtCore.QThread()
#         # Step 3: Create a worker object
#         self.worker = Worker()
#         # Step 4: Move worker to the thread
#         self.worker.moveToThread(self.thread)
#         # Step 5: Connect signals and slots
#         self.thread.started.connect(self.worker.run_embedding)
#         self.worker.finished.connect(self.thread.quit)
#         self.worker.finished.connect(self.worker.deleteLater)
#         self.thread.finished.connect(self.thread.deleteLater)
#         self.worker.progress.connect(self.reportProgress)
#         self.worker.status.connect(self.status)
#         self.worker.msg.connect(self.textBrowser.append)
#         # Step 6: Start the thread
#         self.thread.start()
#
#         # Final resets
#         self.worker.finished.connect(
#             lambda: self.comboBox_2.setEnabled(True)
#         )
#
#         self.worker.finished.connect(self.update_plot)
#
#         self.worker.finished.connect(
#             lambda : self.progressBar_7.setValue(100))
#
#         self.worker.finished.connect(
#             lambda: self.pushButton_19.setEnabled(True)
#         )
#
#         self.worker.finished.connect(self.plotArea.autoRange)
#
#     def launchKmeans(self):
#         self.progressBar_7.reset()
#         self.pushButton_23.setEnabled(False)
#         # Step 2: Create a QThread object
#         self.thread = QtCore.QThread()
#         # Step 3: Create a worker object
#         self.worker = Worker()
#         # Step 4: Move worker to the thread
#         self.worker.moveToThread(self.thread)
#         # Step 5: Connect signals and slots
#         self.thread.started.connect(self.worker.run_clustering)
#         self.worker.finished.connect(self.thread.quit)
#         self.worker.finished.connect(self.worker.deleteLater)
#         self.thread.finished.connect(self.thread.deleteLater)
#         self.worker.progress.connect(self.reportProgress)
#         self.worker.msg.connect(self.textBrowser.append)
#         self.worker.status.connect(self.status)
#         # Step 6: Start the thread
#         self.thread.start()
#
#         # Final resets
#         self.worker.finished.connect(
#             lambda: self.checkBox_5.setEnabled(True)
#         )
#
#         self.worker.finished.connect(self.update_plot)
#
#         self.worker.finished.connect(
#             lambda: self.progressBar_7.setValue(100)
#         )
#
#         self.worker.finished.connect(
#             lambda: self.pushButton_23.setEnabled(True)
#         )
#
#     def user_anchor_points(self, index, outName, save: bool, reset: bool, query: bool):
#
#         if query:
#             print('Query is triggered, return list')
#             print(self.current_indices)
#             flat_list = [item for sublist in self.current_indices for item in sublist]
#             print('Selected indices: ' + ' '.join([str(elem) for elem in flat_list]))
#         else:
#             #Check whether index is a group or individual point
#             if len(index) > 1:
#                 index = [index[0]]
#
#             try:
#                 self.current_indices
#             except:
#                 self.current_indices = []
#                 self.current_indices.append(index)
#                 self.textBrowser_4.setText(str(len(self.current_indices)))
#             else:
#                 self.current_indices.append(index)
#                 self.textBrowser_4.setText(str(len(self.current_indices)))
#
#         if reset:
#             self.current_indices = []
#             self.textBrowser_4.setText(str(len(self.current_indices)))
#
#         if save:
#             outFile = str(os.getcwd()) + '/' + outName
#             np.savetext(outFile, self.current_indices, delimiter=",")
#
#     def plot_clicked(self, plot, points, event):
#         clickedPen = pg.mkPen((0, 0, 0), width=3)
#         global lastClicked
#         global lastPlot
#         try:
#             lastPlot
#         except:
#             pass
#         else:
#             if lastPlot is not plot:
#                 lastClicked = []
#         finally:
#             for pt in lastClicked:
#                 pt.resetPen()
#
#         for pt in points:
#             pt.setPen(clickedPen)
#         indices = []
#         for pt in points:
#             pt = pt.pos()
#             x, y = pt.x(), pt.y()
#             lx = np.argwhere(embedding[:, 0] == x)
#             ly = np.argwhere(embedding[:, 1] == y)
#             #lx = np.argwhere(plot.data['x'] == x)
#             #ly = np.argwhere(plot.data['y'] == y)
#             i = np.intersect1d(lx, ly).tolist()
#             indices += i
#         indices = list(set(indices))  #Indicies are returned in order, undecided if this is necessarily OK...
#         lastClicked = points
#         lastPlot = plot
#         #print('Selected indices: ' + ' '.join([str(elem) for elem in indices]))
#         if self.radioButton_2.isChecked():
#             self.user_anchor_points(indices, '', save=False, reset=False, query=False)
#
#         self.currentZind = indices[0]
#         print('Will generate vol for index %s' % self.currentZind)
#
#     def update_plot(self): ###Try break this up or move to a seperate thread for smoothness and speed.
#
#         def map_to_colour(slice):
#             '''
#             ###########################################################################################################
#             Returns a colour object that is indexed by a value between 0 and 1
#
#                 e.g. cm[0.1] corresponds to a particular colour object
#
#             ###########################################################################################################
#             '''
#             userColourOptions = (
#                 'viridis',
#                 'plasma',
#                 'inferno',
#                 'magma',
#                 'cividis',
#                 'twilight',
#                 'hsv',
#                 'seismic_r',
#                 'coolwarm',
#                 'Spectral_r',
#                 'PiYG_r',
#                 'PRGn_r',
#                 'RdGy_r',
#                 'bwr_r'
#             )
#             cm = pg.colormap.get(userColourOptions[ui.comboBox_5.currentIndex()], source='matplotlib')
#             return cm[slice]
#
#         def plot_trajectories():
#             '''
#             ###########################################################################################################
#             Plot (pseudo)trajectories if required
#             ###########################################################################################################
#             '''
#             # Define trajectories by index
#             traj = pg.mkPen((181, 34, 34), width=3, style=QtCore.Qt.DotLine)
#             shadowPen = None  # or pg.mkPen((255, 255, 255), width=3)
#             index_dic = ([4640, 13515, 3635, 8262, 7310, 12757],
#                          [4608, 15361, 10880, 15873, 6660, 5004, 9344, 15362])
#             try:
#                 self.curvePlots
#             except:
#                 self.curvePlots = []
#             else:
#                 for curvePlot in self.curvePlots:
#                     self.plotArea.removeItem(curvePlot)
#
#             # Iterate over trajectories and plot these.
#             for n, index in enumerate(index_dic):
#                 curvePlot = pg.PlotCurveItem(
#                     pen=traj,
#                     shadowPen=shadowPen,
#                     name="trajectory_" + str(n)
#                 )
#
#                 curvePlot.setData(x=embedding[index][:, 0], y=embedding[index][:, 1])
#                 self.curvePlots.append(curvePlot)
#                 self.plotArea.addItem(curvePlot)
#
#         def plot_rois():
#             '''
#             ###########################################################################################################
#             Plot regions of interest (ROIs) if applicable
#             ###########################################################################################################
#             '''
#             axX = self.plotArea.getAxis('bottom')
#             x_min, x_max = axX.range  # <------- get range of x axis
#             axY = self.plotArea.getAxis('left')
#             y_min, y_max = axY.range  # <------- get range of y axis
#
#             try:
#                 self.ROIs
#             except:
#                 com_x, com_y = (x_max + x_min) / 2, (y_max + y_min) / 2
#
#                 roiPen = pg.mkPen((22, 208, 115), width=3, style=QtCore.Qt.SolidLine)
#                 HoverPen = pg.mkPen((255, 255, 0), width=3, style=QtCore.Qt.DashLine)
#
#                 self.ROIs = pg.PolyLineROI(
#                     [[0.2 * x_min + com_x, com_y],
#                      [com_x, 0.2 * y_max + com_y],
#                      [0.2 * x_max + com_x, com_y],
#                      [com_x, 0.2 * y_min + com_y]],
#                     closed=True, pen=roiPen, handlePen=(153, 51, 255), hoverPen=HoverPen)
#
#                 ## handles rotating around center
#                 #self.ROIs.addRotateHandle([com_x, com_y + 0.1 * y_max], [com_x, com_y])
#                 #self.ROIs.addScaleHandle([com_x + 0.05 * x_max, com_y - 0.05 * y_max], [com_x, com_y])
#             finally:
#                 self.plotArea.addItem(self.ROIs)
#
#         def plot_scatter(colour: bool):
#             '''
#             ###########################################################################################################
#             Plot latent space as scatter plot
#             ###########################################################################################################
#             '''
#             self.plotArea.clear()
#             self.plotArea.legend.clear()
#             self.scatterPlots = []
#             interactPlot = pg.ScatterPlotItem(
#                 pen=None,  # pg.mkPen((0, 0, 0), width=1),
#                 symbol='o',
#                 symbolPen=None,
#                 symbolSize=5,
#                 symbolBrush=(65, 105, 225, 100),
#                 downsample=True,
#                 downsampleMethod='mean',
#                 hoverable=True,
#                 hoverSymbol='s',
#                 hoverSize=15,
#                 hoverPen=pg.mkPen((0, 0, 0), width=3),
#                 hoverBrush=pg.mkBrush(250, 128, 114),
#                 brush=pg.mkBrush(65, 105, 225, 100),
#                 name='obs A'
#             )
#             interactPlot.setData(
#                 pos=embedding,
#                 data=range(0, len(embedding)))
#             interactPlot.sigClicked.connect(self.plot_clicked)
#             self.scatterPlots.append(interactPlot)
#             self.plotArea.addItem(interactPlot)
#             self.plotArea.disableAutoRange()
#
#             if colour:
#                 try:
#                     labels
#                 except:
#                     colours = pg.mkBrush(65, 105, 225, 100)
#                 else:
#                     # if len(labels) == len(embedding):
#                     unique = set(labels)
#                     unique_scaled = [item / (len(unique) - 1) for item in unique]
#                     colour_set = list(map(map_to_colour, unique_scaled))
#
#                     cluster_list = []
#                     for label in set(labels):
#                         cluster = [pt for subPlot in self.scatterPlots for pt in subPlot.points() if
#                                    labels[pt.data()] == label]
#                         cluster_list.append(cluster)
#
#                     legend_items = ['obs; clstr_' + str(label) for label in set(labels)]
#
#                     ###########################################################################################################
#                     # Plot latent space after embedding.
#
#                     self.plotArea.clear()
#                     self.plotArea.legend.clear()
#                     for subPlot in self.scatterPlots:
#                         self.plotArea.removeItem(subPlot)
#
#                     self.scatterPlots = []
#                     for cluster_n, cluster in enumerate(cluster_list):
#                         interactPlot = pg.ScatterPlotItem(
#                             pen=pg.mkPen((0, 0, 0), width=1),
#                             symbol='o',
#                             symbolPen=None,  # pg.mkPen((0, 0, 0), width=3),
#                             symbolSize=5,
#                             symbolBrush=(65, 105, 225, 100),
#                             downsample=True,
#                             downsampleMethod='mean',
#                             hoverable=True,
#                             hoverSymbol='s',
#                             hoverSize=15,
#                             hoverPen=pg.mkPen((0, 0, 0), width=3),
#                             hoverBrush=pg.mkBrush(250, 128, 114),
#                             brush=colour_set[cluster_n],
#                             name=legend_items[cluster_n]
#                         )
#                         interactPlot.setData(
#                             pos=[pt.pos() for pt in cluster],
#                             data=[pt.index() for pt in cluster])
#                         interactPlot.sigClicked.connect(self.plot_clicked)
#
#                         self.scatterPlots.append(interactPlot)
#                         self.plotArea.addItem(interactPlot)
#
#                     for plot in self.scatterPlots:
#                         self.plotArea.addItem(plot)
#
#         def heat_map():
#             x_min, y_min = np.amin(embedding, axis=0) #self.plotArea.getAxis('bottom')
#             #x_min, y_min = axX.range  # <------- get range of x axis
#             x_max, y_max = np.amax(embedding, axis=0) #self.plotArea.getAxis('left')
#             #x_max, y_max = axY.range  # <------- get range of y axis
#
#
#             pltRange = (x_max - x_min), (y_max - y_min)
#             pixel = int(pltRange[0]) / 100
#             x_range = int(pltRange[0] / pixel)
#             y_range = int(pltRange[1] / pixel)
#
#             x_r = np.arange(x_min, x_min + pltRange[0] - pixel, pixel)
#             x = np.repeat(x_r, y_range)
#             x = x.reshape(x_range, y_range)
#
#             y_r = np.arange(y_min, y_min + pltRange[1] - pixel, pixel)
#             y = np.tile(y_r, x_range)
#             y = y.reshape(x_range, y_range)
#
#             histo, _, _ = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=[x_range, y_range])
#
#             densityColourMesh = pg.PColorMeshItem(edgecolors=None, antialiasing=False, cmap='grey')
#             densityColourMesh.setZValue(-1)
#             self.plotArea.addItem(densityColourMesh)
#
#             densityColourMesh.setData(x, y, histo[:-1, :-1])
#
#
#
#         '''
#         ###########################################################################################################
#         Process user input and update plotArea accordingly
#         ###########################################################################################################
#         '''
#
#         if self.checkBox_5.checkState() != QtCore.Qt.Checked:
#             plot_scatter(colour=False)
#         else:
#             plot_scatter(colour=True)
#
#         if self.checkBox_8.checkState() == QtCore.Qt.Checked:
#             plot_trajectories()
#         else:
#             try:
#                 self.curvePlots
#             except:
#                 self.curvePlots = []
#             else:
#                 for curvePlot in self.curvePlots:
#                     self.plotArea.removeItem(curvePlot)
#
#         if self.radioButton_3.isChecked():
#             plot_rois()
#         else:
#             try:
#                 self.ROIs
#             except:
#                 pass
#             else:
#                 self.plotArea.removeItem(self.ROIs)
#
#         heat_map()
#
#     def export_roi(self):
#         if self.radioButton_3.isChecked():
#             try:
#                 self.scatterPlots
#             except:
#                 self.scatterPlots = [self.interactPlot]
#             finally:
#                 if self.checkBox_7.isChecked():
#                     select = [pt for subPlot in self.scatterPlots
#                               for pt in subPlot.points()
#                               if not self.ROIs.mapToItem(subPlot, self.ROIs.shape()).contains(pt.pos())]
#                 else:
#                     select = [pt for subPlot in self.scatterPlots
#                               for pt in subPlot.points()
#                               if self.ROIs.mapToItem(subPlot, self.ROIs.shape()).contains(pt.pos())]
#
#             indices_ROI = []
#             for pt in select:
#                 indices_ROI.append(pt.index())
#
#             #DO something with the indicies!
#
#             self.textBrowser.append("A total of " + str(len(indices_ROI)) + " particles were exported and converted to RELION and cryoSPARC formats."
#                                                                             "\n -------------------------")
#
#     def export_clusters(self):
#         if self.radioButton_4.isChecked():
#             try:
#                 labels
#             except:
#                 self.textBrowser.append("You must run clustering before you can export observation clusters "
#                                         "\n -------------------------")
#             else:
#                 try:
#                     self.scatterPlots
#                 except:
#                     self.scatterPlots = [self.interactPlot]
#                 finally:
#                     cluster_list = []
#                     for label in set(labels):
#                         cluster = [pt for subPlot in self.scatterPlots for pt in subPlot.points() if labels[pt.data()] == label]
#                         cluster_list.append(cluster)
#
#                     self.textBrowser.append("Exporting particles as cryoSPARC and RELION formats... \n")
#
#                     for num, cluster in enumerate(cluster_list):
#                         indices_cluster = []
#                         for pt in cluster:
#                             indices_cluster.append(pt.index())
#
#                     ###DO something with the indicies!
#
#                         self.textBrowser.append("\t Done 'cluster_" + str(num) + "'. Contains " + str(len(indices_cluster)) + " particles.")
#                 self.textBrowser.append("\n -------------------------")
#
#     # def trajectory_inference(self):
#     #     import pyVIA.core as via
#     #     v0 = via.VIA(data, coloursIndex, jac_std_global=0.15, dist_std_local=1, knn=20, too_big_factor=v0_too_big,
#     #                  root_user=[1], dataset='', random_seed=42, is_coarse=True, preserve_disconnected=True)
#     #     v0.run_VIA()
#
#     def return_pressed(self):
#         t0 = time.time()
#         self.apix = 1
#
#         # The use has pressed the Return key; log the current text as HTML
#         from .deps.cryodrgn_minimal import miniDRGN
#         from chimerax.map import volume_from_grid_data
#         from chimerax.map_data import ArrayGridData
#
#         # z = [1.541084408760070801e+00, -1.007112026214599609e+00,
#         #      -1.474042654037475586e+00, 1.732866287231445312e+00,
#         #      -1.431217432022094727e+00, 2.217696428298950195e+00,
#         #      1.486245989799499512e+00, -6.664801239967346191e-01]
#
#         Z = data[self.currentZind]
#
#         # ToolInstance has a 'session' attribute...
#         miniDRGN = miniDRGN(
#             ui.lineEdit_6.displayText(),
#             Z,
#             ui.lineEdit_5.displayText(),
#             apix=self.apix)
#
#         vol = volume_from_grid_data(ArrayGridData(miniDRGN.vol, name='cryoDRGN vol'), wiggle)
#         vol.show()
#
#         print(time.time() - t0)
#
#     def setupUi(self):
#         '''
#         Build the main window and define all attributes for Qt
#         '''
#         global layout
#         layout = QtWidgets.QFormLayout()
#         MainWindow = QtWidgets.QMainWindow()
#         layout.addWidget(MainWindow)
#
#         self.worker = Worker()
#
#         MainWindow.setObjectName("MainWindow")
#         MainWindow.resize(1008, 872)
#         sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
#         sizePolicy.setHorizontalStretch(0)
#         sizePolicy.setVerticalStretch(0)
#         sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
#         MainWindow.setSizePolicy(sizePolicy)
#         MainWindow.setMinimumSize(QtCore.QSize(1008, 872))
#         MainWindow.setMaximumSize(QtCore.QSize(1008, 872))
#         MainWindow.setWindowOpacity(1.0)
#         MainWindow.setWhatsThis("")
#         MainWindow.setStyleSheet("")
#         MainWindow.setDocumentMode(False)
#         self.timer = QtCore.QTimer()
#         self.statusTimer = QtCore.QTimer()
#         self.centralwidget = QtWidgets.QWidget(MainWindow)
#         self.centralwidget.setObjectName("centralwidget")
#         self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
#         self.textBrowser.setGeometry(QtCore.QRect(0, 750, 796, 96))
#         self.textBrowser.setObjectName("textBrowser")
#         self.textBrowser_3 = QtWidgets.QTextBrowser(self.centralwidget)
#         self.textBrowser_3.setEnabled(False)
#         self.textBrowser_3.setGeometry(QtCore.QRect(800, 780, 206, 66))
#         self.textBrowser_3.setAcceptDrops(True)
#         self.textBrowser_3.setStyleSheet("background-color: rgb(255, 255, 255);")
#         self.textBrowser_3.setObjectName("textBrowser_3")
#         self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
#         self.stackedWidget.setGeometry(QtCore.QRect(5, 0, 801, 91))
#         self.stackedWidget.setObjectName("stackedWidget")
#         self.page_2 = QtWidgets.QWidget()
#         self.page_2.setObjectName("page_2")
#         self.pushButton_8 = QtWidgets.QPushButton(self.page_2)
#         self.pushButton_8.setGeometry(QtCore.QRect(0, 30, 196, 21))
#         self.pushButton_8.setObjectName("pushButton_8")
#         self.pushButton_6 = QtWidgets.QPushButton(self.page_2)
#         self.pushButton_6.setGeometry(QtCore.QRect(0, 5, 196, 21))
#         self.pushButton_6.setObjectName("pushButton_6")
#         self.lineEdit_5 = QtWidgets.QLineEdit(self.page_2)
#         self.lineEdit_5.setGeometry(QtCore.QRect(200, 55, 596, 21))
#         self.lineEdit_5.setObjectName("lineEdit_5")
#         self.pushButton_7 = QtWidgets.QPushButton(self.page_2)
#         self.pushButton_7.setGeometry(QtCore.QRect(0, 55, 196, 21))
#         self.pushButton_7.setObjectName("pushButton_7")
#         self.lineEdit_6 = QtWidgets.QLineEdit(self.page_2)
#         self.lineEdit_6.setGeometry(QtCore.QRect(200, 5, 596, 21))
#         self.lineEdit_6.setObjectName("lineEdit_6")
#         self.lineEdit_4 = QtWidgets.QLineEdit(self.page_2)
#         self.lineEdit_4.setGeometry(QtCore.QRect(200, 30, 596, 21))
#         self.lineEdit_4.setObjectName("lineEdit_4")
#         self.stackedWidget.addWidget(self.page_2)
#         self.page = QtWidgets.QWidget()
#         self.page.setObjectName("page")
#         self.pushButton_3 = QtWidgets.QPushButton(self.page)
#         self.pushButton_3.setGeometry(QtCore.QRect(0, 5, 196, 21))
#         self.pushButton_3.setObjectName("pushButton_3")
#         self.pushButton_4 = QtWidgets.QPushButton(self.page)
#         self.pushButton_4.setGeometry(QtCore.QRect(0, 30, 196, 21))
#         self.pushButton_4.setObjectName("pushButton_4")
#         self.lineEdit = QtWidgets.QLineEdit(self.page)
#         self.lineEdit.setGeometry(QtCore.QRect(200, 5, 596, 21))
#         self.lineEdit.setObjectName("lineEdit")
#         self.lineEdit_2 = QtWidgets.QLineEdit(self.page)
#         self.lineEdit_2.setGeometry(QtCore.QRect(200, 30, 596, 21))
#         self.lineEdit_2.setObjectName("lineEdit_2")
#         self.stackedWidget.addWidget(self.page)
#         self.comboBox = QtWidgets.QComboBox(self.centralwidget)
#         self.comboBox.setGeometry(QtCore.QRect(820, 110, 166, 26))
#         self.comboBox.setObjectName("comboBox")
#         self.comboBox.addItem("")
#         self.comboBox.addItem("")
#         self.comboBox_3 = QtWidgets.QComboBox(self.centralwidget)
#         self.comboBox_3.setEnabled(False)
#         self.comboBox_3.setGeometry(QtCore.QRect(820, 165, 166, 21))
#         self.comboBox_3.setObjectName("comboBox_3")
#         self.comboBox_3.addItem("")
#         self.comboBox_3.addItem("")
#         self.comboBox_3.addItem("")
#         self.comboBox_3.addItem("")
#         self.comboBox_3.addItem("")
#         self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
#         self.comboBox_2.setEnabled(False)
#         self.comboBox_2.setGeometry(QtCore.QRect(820, 215, 166, 21))
#         self.comboBox_2.setObjectName("comboBox_2")
#         self.comboBox_2.addItem("")
#         self.comboBox_2.addItem("")
#         self.comboBox_2.addItem("")
#         self.comboBox_2.addItem("")
#         self.comboBox_2.addItem("")
#         self.stackedWidget_3 = QtWidgets.QStackedWidget(self.centralwidget)
#         self.stackedWidget_3.setEnabled(True)
#         self.stackedWidget_3.setGeometry(QtCore.QRect(795, 260, 221, 381))
#         self.stackedWidget_3.setObjectName("stackedWidget_3")
#         self.page_12 = QtWidgets.QWidget()
#         self.page_12.setObjectName("page_12")
#         self.checkBox = QtWidgets.QCheckBox(self.page_12)
#         self.checkBox.setGeometry(QtCore.QRect(10, 25, 181, 21))
#         self.checkBox.setObjectName("checkBox")
#         self.label_8 = QtWidgets.QLabel(self.page_12)
#         self.label_8.setGeometry(QtCore.QRect(10, -10, 221, 31))
#         font = QtGui.QFont()
#         font.setPointSize(8)
#         font.setBold(True)
#         font.setWeight(75)
#         self.label_8.setFont(font)
#         self.label_8.setObjectName("label_8")
#         self.line_4 = QtWidgets.QFrame(self.page_12)
#         self.line_4.setGeometry(QtCore.QRect(10, 10, 201, 20))
#         self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
#         self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
#         self.line_4.setObjectName("line_4")
#         self.pushButton_27 = QtWidgets.QPushButton(self.page_12)
#         self.pushButton_27.setEnabled(True)
#         self.pushButton_27.setGeometry(QtCore.QRect(35, 45, 161, 26))
#         self.pushButton_27.setObjectName("pushButton_27")
#         self.stackedWidget_3.addWidget(self.page_12)
#         self.page_16 = QtWidgets.QWidget()
#         self.page_16.setObjectName("page_16")
#         self.label_7 = QtWidgets.QLabel(self.page_16)
#         self.label_7.setGeometry(QtCore.QRect(10, -10, 171, 31))
#         font = QtGui.QFont()
#         font.setPointSize(8)
#         font.setBold(True)
#         font.setWeight(75)
#         self.label_7.setFont(font)
#         self.label_7.setObjectName("label_7")
#         self.label_11 = QtWidgets.QLabel(self.page_16)
#         self.label_11.setGeometry(QtCore.QRect(15, 60, 121, 31))
#         font = QtGui.QFont()
#         font.setPointSize(8)
#         font.setBold(True)
#         font.setWeight(75)
#         self.label_11.setFont(font)
#         self.label_11.setObjectName("label_11")
#         self.spinBox_2 = QtWidgets.QSpinBox(self.page_16)
#         self.spinBox_2.setGeometry(QtCore.QRect(140, 60, 61, 31))
#         self.spinBox_2.setProperty("value", 10)
#         self.spinBox_2.setObjectName("spinBox_2")
#         self.pushButton_23 = QtWidgets.QPushButton(self.page_16)
#         self.pushButton_23.setGeometry(QtCore.QRect(55, 235, 75, 23))
#         self.pushButton_23.setObjectName("pushButton_23")
#         self.pushButton_24 = QtWidgets.QPushButton(self.page_16)
#         self.pushButton_24.setGeometry(QtCore.QRect(135, 235, 75, 23))
#         self.pushButton_24.setObjectName("pushButton_24")
#         self.line_3 = QtWidgets.QFrame(self.page_16)
#         self.line_3.setGeometry(QtCore.QRect(10, 10, 201, 20))
#         self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
#         self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
#         self.line_3.setObjectName("line_3")
#         self.line_5 = QtWidgets.QFrame(self.page_16)
#         self.line_5.setGeometry(QtCore.QRect(5, 215, 201, 20))
#         self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
#         self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
#         self.line_5.setObjectName("line_5")
#         self.checkBox_5 = QtWidgets.QCheckBox(self.page_16)
#         self.checkBox_5.setGeometry(QtCore.QRect(5, 95, 176, 21))
#         self.checkBox_5.setObjectName("checkBox_5")
#         self.checkBox_6 = QtWidgets.QCheckBox(self.page_16)
#         self.checkBox_6.setGeometry(QtCore.QRect(5, 115, 231, 31))
#         self.checkBox_6.setObjectName("checkBox_6")
#         self.comboBox_4 = QtWidgets.QComboBox(self.page_16)
#         self.comboBox_4.setEnabled(True)
#         self.comboBox_4.setGeometry(QtCore.QRect(5, 25, 201, 26))
#         self.comboBox_4.setObjectName("comboBox_4")
#         self.comboBox_4.addItem("")
#         self.comboBox_4.addItem("")
#         self.comboBox_4.addItem("")
#         self.comboBox_4.addItem("")
#         self.comboBox_4.addItem("")
#         self.comboBox_4.addItem("")
#         self.comboBox_4.addItem("")
#         self.comboBox_4.addItem("")
#         self.comboBox_4.addItem("")
#         self.comboBox_4.addItem("")
#         self.comboBox_5 = QtWidgets.QComboBox(self.page_16)
#         self.comboBox_5.setEnabled(True)
#         self.comboBox_5.setGeometry(QtCore.QRect(5, 170, 201, 26))
#         self.comboBox_5.setObjectName("comboBox_5")
#         self.comboBox_5.addItem("")
#         self.comboBox_5.addItem("")
#         self.comboBox_5.addItem("")
#         self.comboBox_5.addItem("")
#         self.comboBox_5.addItem("")
#         self.comboBox_5.addItem("")
#         self.comboBox_5.addItem("")
#         self.comboBox_5.addItem("")
#         self.comboBox_5.addItem("")
#         self.comboBox_5.addItem("")
#         self.comboBox_5.addItem("")
#         self.comboBox_5.addItem("")
#         self.comboBox_5.addItem("")
#         self.comboBox_5.addItem("")
#         self.checkBox_2 = QtWidgets.QCheckBox(self.page_16)
#         self.checkBox_2.setEnabled(False)
#         self.checkBox_2.setGeometry(QtCore.QRect(5, 145, 221, 21))
#         self.checkBox_2.setWhatsThis("")
#         self.checkBox_2.setAccessibleDescription("")
#         self.checkBox_2.setObjectName("checkBox_2")
#         self.stackedWidget_3.addWidget(self.page_16)
#         self.page_13 = QtWidgets.QWidget()
#         self.page_13.setObjectName("page_13")
#         self.pushButton = QtWidgets.QPushButton(self.page_13)
#         self.pushButton.setGeometry(QtCore.QRect(55, 235, 75, 23))
#         self.pushButton.setObjectName("pushButton")
#         self.label_5 = QtWidgets.QLabel(self.page_13)
#         self.label_5.setGeometry(QtCore.QRect(10, -10, 221, 31))
#         font = QtGui.QFont()
#         font.setPointSize(8)
#         font.setBold(True)
#         font.setWeight(75)
#         self.label_5.setFont(font)
#         self.label_5.setObjectName("label_5")
#         self.pushButton_2 = QtWidgets.QPushButton(self.page_13)
#         self.pushButton_2.setGeometry(QtCore.QRect(135, 235, 75, 23))
#         self.pushButton_2.setObjectName("pushButton_2")
#         self.textBrowser_4 = QtWidgets.QTextBrowser(self.page_13)
#         self.textBrowser_4.setGeometry(QtCore.QRect(145, 120, 56, 31))
#         self.textBrowser_4.setObjectName("textBrowser_4")
#         self.label_6 = QtWidgets.QLabel(self.page_13)
#         self.label_6.setGeometry(QtCore.QRect(20, 120, 131, 31))
#         font = QtGui.QFont()
#         font.setPointSize(8)
#         font.setBold(True)
#         font.setWeight(75)
#         self.label_6.setFont(font)
#         self.label_6.setObjectName("label_6")
#         self.label_10 = QtWidgets.QLabel(self.page_13)
#         self.label_10.setGeometry(QtCore.QRect(110, 70, 41, 31))
#         font = QtGui.QFont()
#         font.setPointSize(8)
#         font.setBold(True)
#         font.setWeight(75)
#         self.label_10.setFont(font)
#         self.label_10.setObjectName("label_10")
#         self.line = QtWidgets.QFrame(self.page_13)
#         self.line.setGeometry(QtCore.QRect(15, 215, 201, 20))
#         self.line.setFrameShape(QtWidgets.QFrame.HLine)
#         self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
#         self.line.setObjectName("line")
#         self.line_2 = QtWidgets.QFrame(self.page_13)
#         self.line_2.setGeometry(QtCore.QRect(10, 10, 201, 20))
#         self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
#         self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
#         self.line_2.setObjectName("line_2")
#         self.spinBox_4 = QtWidgets.QSpinBox(self.page_13)
#         self.spinBox_4.setGeometry(QtCore.QRect(145, 45, 61, 31))
#         self.spinBox_4.setObjectName("spinBox_4")
#         self.radioButton = QtWidgets.QRadioButton(self.page_13)
#         self.radioButton.setGeometry(QtCore.QRect(5, 25, 211, 22))
#         self.radioButton.setChecked(True)
#         self.radioButton.setAutoExclusive(True)
#         self.radioButton.setObjectName("radioButton")
#         self.radioButton_2 = QtWidgets.QRadioButton(self.page_13)
#         self.radioButton_2.setGeometry(QtCore.QRect(5, 100, 166, 22))
#         self.radioButton_2.setObjectName("radioButton_2")
#         self.checkBox_8 = QtWidgets.QCheckBox(self.page_13)
#         self.checkBox_8.setGeometry(QtCore.QRect(5, 155, 156, 22))
#         self.checkBox_8.setObjectName("checkBox_8")
#         self.pushButton_13 = QtWidgets.QPushButton(self.page_13)
#         self.pushButton_13.setGeometry(QtCore.QRect(55, 260, 156, 23))
#         self.pushButton_13.setObjectName("pushButton_13")
#         self.stackedWidget_3.addWidget(self.page_13)
#         self.page_3 = QtWidgets.QWidget()
#         self.page_3.setObjectName("page_3")
#         self.label_9 = QtWidgets.QLabel(self.page_3)
#         self.label_9.setGeometry(QtCore.QRect(10, -10, 181, 31))
#         font = QtGui.QFont()
#         font.setPointSize(8)
#         font.setBold(True)
#         font.setWeight(75)
#         self.label_9.setFont(font)
#         self.label_9.setObjectName("label_9")
#         self.line_6 = QtWidgets.QFrame(self.page_3)
#         self.line_6.setGeometry(QtCore.QRect(5, 10, 201, 20))
#         self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
#         self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
#         self.line_6.setObjectName("line_6")
#         self.line_12 = QtWidgets.QFrame(self.page_3)
#         self.line_12.setGeometry(QtCore.QRect(10, 220, 201, 20))
#         self.line_12.setFrameShape(QtWidgets.QFrame.HLine)
#         self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
#         self.line_12.setObjectName("line_12")
#         self.pushButton_10 = QtWidgets.QPushButton(self.page_3)
#         self.pushButton_10.setGeometry(QtCore.QRect(55, 235, 75, 23))
#         self.pushButton_10.setObjectName("pushButton_10")
#         self.pushButton_11 = QtWidgets.QPushButton(self.page_3)
#         self.pushButton_11.setGeometry(QtCore.QRect(135, 235, 75, 23))
#         self.pushButton_11.setObjectName("pushButton_11")
#         self.comboBox_6 = QtWidgets.QComboBox(self.page_3)
#         self.comboBox_6.setEnabled(True)
#         self.comboBox_6.setGeometry(QtCore.QRect(5, 285, 201, 26))
#         self.comboBox_6.setObjectName("comboBox_6")
#         self.comboBox_6.addItem("")
#         self.comboBox_6.addItem("")
#         self.comboBox_6.addItem("")
#         self.comboBox_6.addItem("")
#         self.comboBox_6.addItem("")
#         self.line_13 = QtWidgets.QFrame(self.page_3)
#         self.line_13.setGeometry(QtCore.QRect(10, 255, 201, 20))
#         self.line_13.setFrameShape(QtWidgets.QFrame.HLine)
#         self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
#         self.line_13.setObjectName("line_13")
#         self.comboBox_7 = QtWidgets.QComboBox(self.page_3)
#         self.comboBox_7.setEnabled(True)
#         self.comboBox_7.setGeometry(QtCore.QRect(155, 25, 51, 21))
#         self.comboBox_7.setObjectName("comboBox_7")
#         self.comboBox_7.addItem("")
#         self.comboBox_7.addItem("")
#         self.comboBox_7.addItem("")
#         self.comboBox_7.addItem("")
#         self.comboBox_8 = QtWidgets.QComboBox(self.page_3)
#         self.comboBox_8.setEnabled(True)
#         self.comboBox_8.setGeometry(QtCore.QRect(155, 50, 51, 21))
#         self.comboBox_8.setObjectName("comboBox_8")
#         self.comboBox_8.addItem("")
#         self.comboBox_8.addItem("")
#         self.comboBox_8.addItem("")
#         self.comboBox_8.addItem("")
#         self.comboBox_8.addItem("")
#         self.comboBox_8.addItem("")
#         self.comboBox_8.addItem("")
#         self.comboBox_8.addItem("")
#         self.comboBox_8.addItem("")
#         self.comboBox_8.addItem("")
#         self.comboBox_9 = QtWidgets.QComboBox(self.page_3)
#         self.comboBox_9.setEnabled(True)
#         self.comboBox_9.setGeometry(QtCore.QRect(90, 75, 116, 21))
#         self.comboBox_9.setObjectName("comboBox_9")
#         self.comboBox_9.addItem("")
#         self.comboBox_9.addItem("")
#         self.comboBox_9.addItem("")
#         self.comboBox_9.addItem("")
#         self.comboBox_9.addItem("")
#         self.label_3 = QtWidgets.QLabel(self.page_3)
#         self.label_3.setGeometry(QtCore.QRect(10, 25, 126, 21))
#         self.label_3.setObjectName("label_3")
#         self.label_4 = QtWidgets.QLabel(self.page_3)
#         self.label_4.setGeometry(QtCore.QRect(10, 50, 126, 21))
#         self.label_4.setObjectName("label_4")
#         self.label_13 = QtWidgets.QLabel(self.page_3)
#         self.label_13.setGeometry(QtCore.QRect(10, 75, 71, 21))
#         self.label_13.setObjectName("label_13")
#         self.label_14 = QtWidgets.QLabel(self.page_3)
#         self.label_14.setGeometry(QtCore.QRect(10, 100, 136, 21))
#         self.label_14.setObjectName("label_14")
#         self.lineEdit_7 = QtWidgets.QLineEdit(self.page_3)
#         self.lineEdit_7.setEnabled(True)
#         self.lineEdit_7.setGeometry(QtCore.QRect(145, 100, 61, 21))
#         self.lineEdit_7.setObjectName("lineEdit_7")
#         self.lineEdit_8 = QtWidgets.QLineEdit(self.page_3)
#         self.lineEdit_8.setEnabled(True)
#         self.lineEdit_8.setGeometry(QtCore.QRect(145, 120, 61, 21))
#         self.lineEdit_8.setObjectName("lineEdit_8")
#         self.label_15 = QtWidgets.QLabel(self.page_3)
#         self.label_15.setGeometry(QtCore.QRect(10, 140, 86, 21))
#         self.label_15.setObjectName("label_15")
#         self.lineEdit_9 = QtWidgets.QLineEdit(self.page_3)
#         self.lineEdit_9.setEnabled(True)
#         self.lineEdit_9.setGeometry(QtCore.QRect(145, 140, 61, 21))
#         self.lineEdit_9.setObjectName("lineEdit_9")
#         self.label_16 = QtWidgets.QLabel(self.page_3)
#         self.label_16.setGeometry(QtCore.QRect(10, 120, 86, 21))
#         self.label_16.setObjectName("label_16")
#         self.lineEdit_10 = QtWidgets.QLineEdit(self.page_3)
#         self.lineEdit_10.setEnabled(True)
#         self.lineEdit_10.setGeometry(QtCore.QRect(145, 160, 61, 21))
#         self.lineEdit_10.setObjectName("lineEdit_10")
#         self.label_17 = QtWidgets.QLabel(self.page_3)
#         self.label_17.setGeometry(QtCore.QRect(10, 160, 111, 21))
#         self.label_17.setObjectName("label_17")
#         self.lineEdit_11 = QtWidgets.QLineEdit(self.page_3)
#         self.lineEdit_11.setEnabled(True)
#         self.lineEdit_11.setGeometry(QtCore.QRect(145, 180, 61, 21))
#         self.lineEdit_11.setObjectName("lineEdit_11")
#         self.label_18 = QtWidgets.QLabel(self.page_3)
#         self.label_18.setGeometry(QtCore.QRect(10, 180, 86, 21))
#         self.label_18.setObjectName("label_18")
#         self.label_19 = QtWidgets.QLabel(self.page_3)
#         self.label_19.setGeometry(QtCore.QRect(10, 265, 181, 21))
#         font = QtGui.QFont()
#         font.setPointSize(8)
#         font.setBold(True)
#         font.setWeight(75)
#         self.label_19.setFont(font)
#         self.label_19.setObjectName("label_19")
#         self.checkBox_13 = QtWidgets.QCheckBox(self.page_3)
#         self.checkBox_13.setGeometry(QtCore.QRect(5, 310, 156, 22))
#         self.checkBox_13.setObjectName("checkBox_13")
#         self.pushButton_12 = QtWidgets.QPushButton(self.page_3)
#         self.pushButton_12.setGeometry(QtCore.QRect(60, 330, 146, 23))
#         self.pushButton_12.setObjectName("pushButton_12")
#         self.label_40 = QtWidgets.QLabel(self.page_3)
#         self.label_40.setGeometry(QtCore.QRect(10, 200, 116, 21))
#         self.label_40.setObjectName("label_40")
#         self.lineEdit_17 = QtWidgets.QLineEdit(self.page_3)
#         self.lineEdit_17.setEnabled(True)
#         self.lineEdit_17.setGeometry(QtCore.QRect(145, 200, 61, 21))
#         self.lineEdit_17.setObjectName("lineEdit_17")
#         self.checkBox_18 = QtWidgets.QCheckBox(self.page_3)
#         self.checkBox_18.setGeometry(QtCore.QRect(5, 355, 196, 22))
#         self.checkBox_18.setObjectName("checkBox_18")
#         self.stackedWidget_3.addWidget(self.page_3)
#         self.page_15 = QtWidgets.QWidget()
#         self.page_15.setObjectName("page_15")
#         self.label_12 = QtWidgets.QLabel(self.page_15)
#         self.label_12.setGeometry(QtCore.QRect(10, -10, 156, 31))
#         font = QtGui.QFont()
#         font.setPointSize(8)
#         font.setBold(True)
#         font.setWeight(75)
#         self.label_12.setFont(font)
#         self.label_12.setObjectName("label_12")
#         self.line_11 = QtWidgets.QFrame(self.page_15)
#         self.line_11.setGeometry(QtCore.QRect(5, 10, 201, 20))
#         self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
#         self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
#         self.line_11.setObjectName("line_11")
#         self.pushButton_5 = QtWidgets.QPushButton(self.page_15)
#         self.pushButton_5.setEnabled(False)
#         self.pushButton_5.setGeometry(QtCore.QRect(125, 230, 85, 27))
#         self.pushButton_5.setObjectName("pushButton_5")
#         self.pushButton_9 = QtWidgets.QPushButton(self.page_15)
#         self.pushButton_9.setEnabled(False)
#         self.pushButton_9.setGeometry(QtCore.QRect(5, 230, 116, 27))
#         self.pushButton_9.setObjectName("pushButton_9")
#         self.radioButton_3 = QtWidgets.QRadioButton(self.page_15)
#         self.radioButton_3.setGeometry(QtCore.QRect(10, 30, 101, 22))
#         self.radioButton_3.setObjectName("radioButton_3")
#         self.radioButton_4 = QtWidgets.QRadioButton(self.page_15)
#         self.radioButton_4.setGeometry(QtCore.QRect(10, 75, 136, 22))
#         self.radioButton_4.setObjectName("radioButton_4")
#         self.checkBox_7 = QtWidgets.QCheckBox(self.page_15)
#         self.checkBox_7.setEnabled(False)
#         self.checkBox_7.setGeometry(QtCore.QRect(55, 50, 121, 22))
#         self.checkBox_7.setObjectName("checkBox_7")
#         self.stackedWidget_3.addWidget(self.page_15)
#         # self.graphicsView = pg.PlotWidget(self.centralwidget)
#         # self.graphicsView.setGeometry(QtCore.QRect(0, 80, 796, 666))
#         # self.graphicsView.setObjectName("graphicsView")
#         self.graphicsView = pg.GraphicsLayoutWidget(self.centralwidget)
#         self.graphicsView.setGeometry(QtCore.QRect(0, 80, 796, 666))
#         self.graphicsView.setObjectName("graphicsView")
#         self.progressBar_7 = QtWidgets.QProgressBar(self.centralwidget)
#         self.progressBar_7.setGeometry(QtCore.QRect(800, 750, 201, 23))
#         self.progressBar_7.setProperty("value", 67)
#         self.progressBar_7.setObjectName("progressBar_7")
#         self.pushButton_19 = QtWidgets.QPushButton(self.centralwidget)
#         self.pushButton_19.setGeometry(QtCore.QRect(945, 190, 41, 21))
#         self.pushButton_19.setObjectName("pushButton_19")
#         self.checkBox_9 = QtWidgets.QCheckBox(self.centralwidget)
#         self.checkBox_9.setEnabled(True)
#         self.checkBox_9.setGeometry(QtCore.QRect(800, 720, 176, 21))
#         self.checkBox_9.setObjectName("checkBox_9")
#         self.checkBox_10 = QtWidgets.QCheckBox(self.centralwidget)
#         self.checkBox_10.setEnabled(True)
#         self.checkBox_10.setGeometry(QtCore.QRect(800, 695, 176, 21))
#         self.checkBox_10.setObjectName("checkBox_10")
#         self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
#         self.lineEdit_3.setEnabled(False)
#         self.lineEdit_3.setGeometry(QtCore.QRect(870, 665, 126, 26))
#         self.lineEdit_3.setObjectName("lineEdit_3")
#         self.checkBox_12 = QtWidgets.QCheckBox(self.centralwidget)
#         self.checkBox_12.setGeometry(QtCore.QRect(800, 645, 196, 21))
#         self.checkBox_12.setObjectName("checkBox_12")
#         self.label_2 = QtWidgets.QLabel(self.centralwidget)
#         self.label_2.setGeometry(QtCore.QRect(805, 670, 61, 17))
#         self.label_2.setObjectName("label_2")
#         self.label_37 = QtWidgets.QLabel(self.centralwidget)
#         self.label_37.setGeometry(QtCore.QRect(805, 195, 41, 21))
#         font = QtGui.QFont()
#         font.setPointSize(8)
#         font.setBold(True)
#         font.setWeight(75)
#         self.label_37.setFont(font)
#         self.label_37.setObjectName("label_37")
#         self.label_38 = QtWidgets.QLabel(self.centralwidget)
#         self.label_38.setGeometry(QtCore.QRect(805, 145, 131, 21))
#         font = QtGui.QFont()
#         font.setPointSize(8)
#         font.setBold(True)
#         font.setWeight(75)
#         self.label_38.setFont(font)
#         self.label_38.setObjectName("label_38")
#         self.label_39 = QtWidgets.QLabel(self.centralwidget)
#         self.label_39.setGeometry(QtCore.QRect(805, 85, 141, 21))
#         font = QtGui.QFont()
#         font.setPointSize(8)
#         font.setBold(True)
#         font.setWeight(75)
#         self.label_39.setFont(font)
#         self.label_39.setObjectName("label_39")
#         self.label = QtWidgets.QLabel(self.centralwidget)
#         self.label.setGeometry(QtCore.QRect(805, 0, 196, 76))
#         self.label.setText("")
#         self.label.setPixmap(QtGui.QPixmap("./Wiggle.PNG"))
#         self.label.setScaledContents(True)
#         self.label.setObjectName("label")
#         MainWindow.setCentralWidget(self.centralwidget)
#         self.statusBar = QtWidgets.QStatusBar(MainWindow)
#         self.statusBar.setObjectName("statusBar")
#         self.statusBar.showMessage("Status: IDLE")
#         MainWindow.setStatusBar(self.statusBar)
#         self.retranslateUi(MainWindow)
#
#         # Set up the pyqtgraph
#         self.plotArea = self.graphicsView.addPlot() #title="Interactive Latent Coordinate Space"
#         self.plotArea.addLegend()
#         self.interactPlot = pg.ScatterPlotItem(
#             pen=None, #pg.mkPen((0, 0, 0), width=0.5), #Colour and style of edge/outline
#             symbol='o',
#             # symbolPen=None, #Colour and style of connecting line
#             # symbolSize=5,
#             # symbolBrush=(65, 105, 225, 100),
#             Downsample=True,
#             hoverable=True,
#             hoverSymbol='s', #Hover symbol is shape of hovered point
#             hoverSize=15,
#             hoverPen=pg.mkPen((0, 0, 0), width=3), #Hover pen is outline
#             hoverBrush=pg.mkBrush(250, 128, 114), #Hover brush is colour of fill
#             name='Observations (obs)'
#         )
#         self.plotArea.addItem(self.interactPlot)
#
#         '''
#         Connections, signals and actions
#         '''
#         self.interactPlot.sigClicked.connect(self.plot_clicked)
#         self.stackedWidget.setCurrentIndex(0)
#         self.stackedWidget_3.setCurrentIndex(5)
#         self.comboBox_9.setCurrentIndex(2)
#         self.comboBox_7.setCurrentIndex(2)
#         self.comboBox_8.setCurrentIndex(3)
#
#         self.checkBox_5.setEnabled(False)
#         self.comboBox_5.currentIndexChanged['int'].connect(self.update_plot)
#         self.comboBox.currentIndexChanged['int'].connect(self.stackedWidget.setCurrentIndex)
#         self.comboBox.activated['int'].connect(self.stackedWidget.setCurrentIndex)
#         self.comboBox_2.activated['int'].connect(self.stackedWidget_3.setCurrentIndex)
#
#         self.checkBox.toggled['bool'].connect(self.pushButton_27.setDisabled)
#         self.radioButton_4.toggled['bool'].connect(self.checkBox_7.setDisabled)
#         self.radioButton_4.clicked['bool'].connect(self.pushButton_9.setEnabled)
#         self.radioButton_4.clicked['bool'].connect(self.pushButton_5.setEnabled)
#
#         self.radioButton_3.clicked['bool'].connect(self.pushButton_5.setEnabled)
#         self.radioButton_3.clicked['bool'].connect(self.pushButton_9.setEnabled)
#         self.radioButton_3.toggled['bool'].connect(self.checkBox_7.setEnabled)
#         self.radioButton_3.toggled['bool'].connect(self.update_plot)
#         self.checkBox_12.toggled['bool'].connect(self.lineEdit_3.setEnabled)
#         self.checkBox_12.clicked['bool'].connect(self.lineEdit_3.setEnabled)
#         self.pushButton_5.setEnabled(False)
#         self.pushButton_9.setEnabled(False)
#         self.pushButton_3.clicked.connect(
#             partial(self.browse_file, "Select canonical map", self.pushButton_3, "Volumes (*.mrc)"))
#         self.pushButton_4.clicked.connect(
#             partial(self.browse_file, "Select latent space", self.pushButton_4, "Pickle (*.pkl)"))
#         self.pushButton_6.clicked.connect(
#             partial(self.browse_file, "Select consensus map", self.pushButton_6, "Pickle (*.pkl)"))
#         self.pushButton_7.clicked.connect(
#             partial(self.browse_file, "Select network weights", self.pushButton_7, "Pickle (*.pkl)"))
#         self.pushButton_8.clicked.connect(
#             partial(self.browse_file, "Select latent space (z.pkl)", self.pushButton_8, "Pickle (*.pkl)"))
#         self.pushButton_19.clicked.connect(self.launchEmbedding)
#         self.pushButton_23.clicked.connect(self.launchKmeans)
#         self.checkBox_8.stateChanged.connect(self.update_plot)
#         self.checkBox_5.stateChanged.connect(self.update_plot)
#         self.timer.timeout.connect(
#             lambda: self.reportProgress('', '', False, True, False))
#         self.statusTimer.timeout.connect(
#             lambda: self.status(False))
#         self.pushButton_2.clicked.connect(
#             lambda : self.user_anchor_points('', '', save=False, query=False, reset=True))
#         self.pushButton.clicked.connect(
#             lambda : self.user_anchor_points('', '', save=False, query=True, reset=False))
#
#         self.pushButton_5.clicked.connect(
#                 lambda : self.export_roi()
#             )
#         self.pushButton_5.clicked.connect(
#                 lambda : self.export_clusters()
#             )
#
#         self.pushButton_27.clicked.connect(self.return_pressed)
#
#         QtCore.QMetaObject.connectSlotsByName(MainWindow)
#
#     def retranslateUi(self, MainWindow):
#         _translate = QtCore.QCoreApplication.translate
#         MainWindow.setWindowTitle(_translate("MainWindow", "WIGGLE - 0.0.5"))
#         self.textBrowser.setHtml(_translate("MainWindow",
#                                             "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
#                                             "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
#                                             "p, li { white-space: pre-wrap; }\n"
#                                             "</style></head><body style=\" font-family:\'Noto Sans\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
#                                             "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:10pt; font-weight:600;\">Command line output will appear here</span></p></body></html>"))
#         self.textBrowser_3.setHtml(_translate("MainWindow",
#                                               "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
#                                               "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
#                                               "p, li { white-space: pre-wrap; }\n"
#                                               "</style></head><body style=\" font-family:\'Noto Sans\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
#                                               "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:7pt; font-weight:600;\">Charles Bayly-Jones 2021</span></p>\n"
#                                               "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:7pt; font-weight:600;\"><br /></p>\n"
#                                               "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:7pt; font-weight:600;\">wiggle.help@gmail.com</span></p>\n"
#                                               "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:7pt;\"> </span></p>\n"
#                                               "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:7pt; font-weight:600;\">Monash University, Australia</span></p></body></html>"))
#         self.pushButton_8.setText(_translate("MainWindow", "Load latent space (z.pkl)"))
#         self.pushButton_6.setText(_translate("MainWindow", "Load config (.pkl)"))
#         self.pushButton_7.setText(_translate("MainWindow", "Load network weights (.pkl)"))
#         self.pushButton_3.setText(_translate("MainWindow", "Load canonical map"))
#         self.pushButton_4.setText(_translate("MainWindow", "Load deformation field"))
#         self.comboBox.setItemText(0, _translate("MainWindow", "cryoDRGN"))
#         self.comboBox.setItemText(1, _translate("MainWindow", "cryoSPARC"))
#         self.comboBox_3.setToolTip(_translate("MainWindow", "To enable, first import a latent space."))
#         self.comboBox_3.setItemText(0, _translate("MainWindow", "UMAP"))
#         self.comboBox_3.setItemText(1, _translate("MainWindow", "PCA"))
#         self.comboBox_3.setItemText(2, _translate("MainWindow", "tSNE"))
#         self.comboBox_3.setItemText(3, _translate("MainWindow", "PHATE"))
#         self.comboBox_3.setItemText(4, _translate("MainWindow", "cVAE"))
#         self.comboBox_2.setToolTip(_translate("MainWindow", "To enable, first run dimensionality reduction."))
#         self.comboBox_2.setItemText(0, _translate("MainWindow", "Interactive"))
#         self.comboBox_2.setItemText(1, _translate("MainWindow", "Clustering"))
#         self.comboBox_2.setItemText(2, _translate("MainWindow", "Graph traversal"))
#         self.comboBox_2.setItemText(3, _translate("MainWindow", "MEP analysis"))
#         self.comboBox_2.setItemText(4, _translate("MainWindow", "Misc"))
#         self.checkBox.setToolTip(_translate("MainWindow",
#                                             "Automatically replace the displayed map in ChimeraX with the selected configurational state."))
#         self.checkBox.setText(_translate("MainWindow", "Update map continually"))
#         self.label_8.setText(_translate("MainWindow", "Interactive mode"))
#         self.pushButton_27.setToolTip(_translate("MainWindow",
#                                                  "Manually update the displayed map in ChimeraX with the configurational state currently selected."))
#         self.pushButton_27.setText(_translate("MainWindow", "Update"))
#         self.label_7.setText(_translate("MainWindow", "Cluster analysis"))
#         self.label_11.setToolTip(
#             _translate("MainWindow", "Determines the number of classes, groups or clusters generated by kmeans"))
#         self.label_11.setText(
#             _translate("MainWindow", "<html><head/><body><p align=\"right\">Number of clusters? :</p></body></html>"))
#         self.pushButton_23.setToolTip(_translate("MainWindow", "Start analysis."))
#         self.pushButton_23.setText(_translate("MainWindow", "Go!"))
#         self.pushButton_24.setToolTip(_translate("MainWindow", "Clear previous results."))
#         self.pushButton_24.setText(_translate("MainWindow", "Reset"))
#         self.checkBox_5.setToolTip(_translate("MainWindow", "Colour scatterplot by k means cluster indicies"))
#         self.checkBox_5.setText(_translate("MainWindow", "Colour points by clusters"))
#         self.checkBox_6.setToolTip(
#             _translate("MainWindow", "For each cluster center, generate a volume and display it in ChimeraX."))
#         self.checkBox_6.setText(_translate("MainWindow", "Volumes at cluster centers?"))
#         self.comboBox_4.setToolTip(_translate("MainWindow", "To enable, first import a latent space."))
#         self.comboBox_4.setItemText(0, _translate("MainWindow", "KMeans (fast)"))
#         self.comboBox_4.setItemText(1, _translate("MainWindow", "Affinity Propagation (slow)"))
#         self.comboBox_4.setItemText(2, _translate("MainWindow", "MeanShift (slow)"))
#         self.comboBox_4.setItemText(3, _translate("MainWindow", "Spectral Clustering (slow)"))
#         self.comboBox_4.setItemText(4, _translate("MainWindow", "Ward (fast)"))
#         self.comboBox_4.setItemText(5, _translate("MainWindow", "Agglomerative Clustering (fast)"))
#         self.comboBox_4.setItemText(6, _translate("MainWindow", "DBSCAN (fast)"))
#         self.comboBox_4.setItemText(7, _translate("MainWindow", "OPTICS (slow)"))
#         self.comboBox_4.setItemText(8, _translate("MainWindow", "BIRCH (fast)"))
#         self.comboBox_4.setItemText(9, _translate("MainWindow", "Gaussian Mixture (fast)"))
#         self.comboBox_5.setToolTip(_translate("MainWindow", "To enable, first import a latent space."))
#         self.comboBox_5.setItemText(0, _translate("MainWindow", "viridis (u)"))
#         self.comboBox_5.setItemText(1, _translate("MainWindow", "plasma (u)"))
#         self.comboBox_5.setItemText(2, _translate("MainWindow", "inferno (u)"))
#         self.comboBox_5.setItemText(3, _translate("MainWindow", "magma (u)"))
#         self.comboBox_5.setItemText(4, _translate("MainWindow", "cividis (u)"))
#         self.comboBox_5.setItemText(5, _translate("MainWindow", "twilight (c)"))
#         self.comboBox_5.setItemText(6, _translate("MainWindow", "hsv (c)"))
#         self.comboBox_5.setItemText(7, _translate("MainWindow", "seismic (d)"))
#         self.comboBox_5.setItemText(8, _translate("MainWindow", "coolwarm (d)"))
#         self.comboBox_5.setItemText(9, _translate("MainWindow", "Spectral (d)"))
#         self.comboBox_5.setItemText(10, _translate("MainWindow", "PiYG (d)"))
#         self.comboBox_5.setItemText(11, _translate("MainWindow", "PRGn (d)"))
#         self.comboBox_5.setItemText(12, _translate("MainWindow", "RdGy (d)"))
#         self.comboBox_5.setItemText(13, _translate("MainWindow", "bwr (d)"))
#         self.checkBox_2.setToolTip(_translate("MainWindow",
#                                               "Colour reconstruction within ChimeraX based on cluster label. To enable, first run kmeans."))
#         self.checkBox_2.setText(_translate("MainWindow", "Colour volume by cluster"))
#         self.pushButton.setToolTip(_translate("MainWindow", "Start analysis."))
#         self.pushButton.setText(_translate("MainWindow", "Go!"))
#         self.label_5.setText(_translate("MainWindow", "Path traversal"))
#         self.pushButton_2.setToolTip(_translate("MainWindow", "Clear previous results."))
#         self.pushButton_2.setText(_translate("MainWindow", "Reset"))
#         self.textBrowser_4.setToolTip(_translate("MainWindow",
#                                                  "The number of currently selected anchors is displayed here. Reset will clear if you need to start again."))
#         self.label_6.setToolTip(_translate("MainWindow",
#                                            "The number of currently selected anchors is displayed here. Reset will clear if you need to start again."))
#         self.label_6.setText(_translate("MainWindow", "Current anchor count:"))
#         self.label_10.setText(_translate("MainWindow", "OR"))
#         self.radioButton.setToolTip(_translate("MainWindow",
#                                                "Traverse the latent space based on the principal component below (e.g. 1 will traverse the latent space along PC1)"))
#         self.radioButton.setText(_translate("MainWindow", "Traverse principal component:"))
#         self.radioButton_2.setToolTip(_translate("MainWindow",
#                                                  "Select multiple anchor points in the latent space and then attempt to traverse the space crossing these anchors."))
#         self.radioButton_2.setText(_translate("MainWindow", "Select anchor points:"))
#         self.checkBox_8.setText(_translate("MainWindow", "Plot trajectories?"))
#         self.pushButton_13.setToolTip(_translate("MainWindow", "Start analysis."))
#         self.pushButton_13.setText(_translate("MainWindow", "Make volume series"))
#         self.label_9.setText(_translate("MainWindow", "Minimum energy path analysis"))
#         self.pushButton_10.setToolTip(_translate("MainWindow", "Start analysis."))
#         self.pushButton_10.setText(_translate("MainWindow", "Go!"))
#         self.pushButton_11.setToolTip(_translate("MainWindow", "Clear previous results."))
#         self.pushButton_11.setText(_translate("MainWindow", "Reset"))
#         self.comboBox_6.setToolTip(_translate("MainWindow", "To enable, first import a latent space."))
#         self.comboBox_7.setToolTip(_translate("MainWindow", "To enable, first import a latent space."))
#         self.comboBox_7.setItemText(0, _translate("MainWindow", "3"))
#         self.comboBox_7.setItemText(1, _translate("MainWindow", "5"))
#         self.comboBox_7.setItemText(2, _translate("MainWindow", "7"))
#         self.comboBox_7.setItemText(3, _translate("MainWindow", "9"))
#         self.comboBox_8.setToolTip(_translate("MainWindow", "To enable, first import a latent space."))
#         self.comboBox_8.setItemText(0, _translate("MainWindow", "3"))
#         self.comboBox_8.setItemText(1, _translate("MainWindow", "5"))
#         self.comboBox_8.setItemText(2, _translate("MainWindow", "7"))
#         self.comboBox_8.setItemText(3, _translate("MainWindow", "9"))
#         self.comboBox_8.setItemText(4, _translate("MainWindow", "11"))
#         self.comboBox_8.setItemText(5, _translate("MainWindow", "13"))
#         self.comboBox_8.setItemText(6, _translate("MainWindow", "15"))
#         self.comboBox_8.setItemText(7, _translate("MainWindow", "17"))
#         self.comboBox_8.setItemText(8, _translate("MainWindow", "19"))
#         self.comboBox_8.setItemText(9, _translate("MainWindow", "21"))
#         self.comboBox_9.setToolTip(_translate("MainWindow", "To enable, first import a latent space."))
#         self.comboBox_9.setItemText(0, _translate("MainWindow", "Coarse (96)"))
#         self.comboBox_9.setItemText(1, _translate("MainWindow", "Medium (128)"))
#         self.comboBox_9.setItemText(2, _translate("MainWindow", "Fine (156)"))
#         self.comboBox_9.setItemText(3, _translate("MainWindow", "Ultrafine (196)"))
#         self.comboBox_9.setItemText(4, _translate("MainWindow", "Extreme (256)"))
#         self.comboBox_6.setItemText(0, _translate("MainWindow", "Energy -36.76: Rank 1"))
#         self.comboBox_6.setItemText(1, _translate("MainWindow", "Energy 6.34: Rank 1"))
#         self.comboBox_6.setItemText(2, _translate("MainWindow", "Energy 443.9: Rank 1"))
#         self.comboBox_6.setItemText(3, _translate("MainWindow", "Energy -76.6: Rank 1"))
#         self.comboBox_6.setItemText(3, _translate("MainWindow", "Energy 0.38: Rank 1"))
#         self.label_3.setText(_translate("MainWindow", "Width of weiner filter"))
#         self.label_4.setText(_translate("MainWindow", "Width of median filter"))
#         self.label_13.setText(_translate("MainWindow", "Resolution"))
#         self.label_14.setText(_translate("MainWindow", "No. searches to spawn:"))
#         self.lineEdit_7.setText(_translate("MainWindow", "1200"))
#         self.lineEdit_8.setText(_translate("MainWindow", "0.05"))
#         self.label_15.setText(_translate("MainWindow", "Sampling"))
#         self.lineEdit_9.setText(_translate("MainWindow", "40"))
#         self.label_16.setText(_translate("MainWindow", "Learning rate"))
#         self.lineEdit_10.setText(_translate("MainWindow", "0.2"))
#         self.label_17.setText(_translate("MainWindow", "Coupling constant"))
#         self.lineEdit_11.setText(_translate("MainWindow", "2.5"))
#         self.label_18.setText(_translate("MainWindow", "String tension"))
#         self.label_19.setText(_translate("MainWindow", "Select path below"))
#         self.checkBox_13.setText(_translate("MainWindow", "Plot trajectory?"))
#         self.pushButton_12.setToolTip(_translate("MainWindow", "Start analysis."))
#         self.pushButton_12.setText(_translate("MainWindow", "Make volume series"))
#         self.label_40.setText(_translate("MainWindow", "Minima to consider"))
#         self.lineEdit_17.setText(_translate("MainWindow", "10"))
#         self.checkBox_18.setText(_translate("MainWindow", "Generate diagnostic plots?"))
#         self.label_12.setText(_translate("MainWindow", "Particle curation"))
#         self.pushButton_5.setText(_translate("MainWindow", "Done!"))
#         self.pushButton_9.setText(_translate("MainWindow", "Reset selection"))
#         self.radioButton_3.setText(_translate("MainWindow", "Lasso tool"))
#         self.radioButton_4.setText(_translate("MainWindow", "Export by cluster"))
#         self.checkBox_7.setText(_translate("MainWindow", "Invert selection"))
#         self.pushButton_19.setText(_translate("MainWindow", "Go!"))
#         self.checkBox_9.setToolTip(_translate("MainWindow", "Colour scatterplot by k means cluster indicies"))
#         self.checkBox_9.setText(_translate("MainWindow", "Hide legend"))
#         self.checkBox_10.setToolTip(_translate("MainWindow", "Colour scatterplot by k means cluster indicies"))
#         self.checkBox_10.setText(_translate("MainWindow", "Save volumes to disk"))
#         self.checkBox_12.setToolTip(_translate("MainWindow", "Randomly sample points from the latent space to speed up visualisation for large data sets"))
#         self.checkBox_12.setText(_translate("MainWindow", "Subsample the latent space"))
#         self.label_2.setText(_translate("MainWindow", "Num. Obs."))
#         self.label_37.setText(_translate("MainWindow", "Mode"))
#         self.label_38.setText(_translate("MainWindow", "Dimensional reduction"))
#         self.label_39.setText(_translate("MainWindow", "Variation analysis input"))