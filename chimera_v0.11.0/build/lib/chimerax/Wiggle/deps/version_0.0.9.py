

import numpy as np
import time
import pyqtgraph as pg

from PyQt5 import QtCore, QtGui, QtWidgets
from functools import partial
from math import floor

np.seterr(divide='ignore', invalid='ignore')
lastClicked = []

class Embedder(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    exit = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(float, float, bool, bool, bool)
    msg = QtCore.pyqtSignal(str)
    status = QtCore.pyqtSignal(bool)
    def __init__(self, technique_index, data_path, frac, subset: bool, prev_changed=None, prev_data=None):
        self.technique = technique_index
        self.data_path = data_path
        self.fraction = frac
        self.do_subset = subset
        self.previous_changed = prev_changed
        self.previous_data = prev_data
        super().__init__()

    def estimate_time(self, operator):
        ### TIMING TEST
        fraction = 0.05
        # subset = int(len(self.data) * fraction)
        # idx = np.random.randint(self.data.shape[0], size=subset)
        # small_data = self.data[idx]
        t0 = time.time()
        # operator.fit_transform(small_data)
        time.sleep(5)
        ETA = (1/fraction) * (time.time() - t0)
        self.progress.emit(ETA, t0, True, True, False)
        return ETA, t0
        ### END TIMING TES

    def run_embedding(self): ###Re implement as above, too much redundancy.
        from .data_handler import DataManager
        self.status.emit(False)
        #technique = self.comboBox_3.currentIndex()
        '''
            Perform a range of dimensionality reduction analyses.
        '''

        if self.do_subset and self.previous_changed is not None:
            if self.previous_changed:
                self.msg.emit("Subset size has changed...")
                self.msg.emit("Subsampling latent space to %s" % int(self.fraction))
                DataManager = DataManager(input=self.data_path, mode=ui.inputType[ui.comboBox.currentIndex()],
                                            subset=True, fraction=int(self.fraction))
                *self.data, = DataManager.load_data()
            else:
                self.msg.emit("Subset size hasn't changed, using previous subset...")
                self.data[2] = self.previous_data
        elif self.do_subset and self.previous_changed is None:
                self.msg.emit("Subsampling latent space to %s" % int(self.fraction))
                DataManager = DataManager(input=self.data_path, mode=ui.inputType[ui.comboBox.currentIndex()],
                                            subset=True, fraction=int(self.fraction))
                *self.data, = DataManager.load_data()
        else:
            self.msg.emit("Using whole dataset. Subsampling can improve interactive experience. Try 25k data points...")
            DataManager = DataManager(input=self.data_path, mode=ui.inputType[ui.comboBox.currentIndex()],
                                      subset=False)
            *self.data, = DataManager.load_data()

        z_space = self.data[2]

        # Exit if data is not defined, no user input.
        if type(self.data) is int:
            print("Exiting embedding, path not found or none")
            self.progress.emit(1, 1, False, False, True)
            self.exit.emit()
            self.status.emit(True)
            return

        # global embedding
        if self.technique == 0:
            self.msg.emit("Running umap... ")
            import umap
            operator = umap.UMAP(random_state=42, verbose=1, densmap=True)
            ETA, t0 = self.estimate_time(operator)
            self.embedding = operator.fit_transform(z_space)
            self.msg.emit("Finished umap! \n -------------------------")

        if self.technique == 1:
            self.msg.emit("Running PCA... ")
            from sklearn.decomposition import PCA
            operator = PCA(n_components=2)
            operator.fit(z_space)
            ETA, t0 = self.estimate_time(operator)
            self.embedding = operator.transform(z_space)
            self.msg.emit("Finished PCA! \n -------------------------")

        if self.technique == 2:
            self.msg.emit("Running tSNE...")
            from sklearn.manifold import TSNE
            operator = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1)
            ETA, t0 = self.estimate_time(operator)
            self.embedding = operator.fit_transform(z_space)
            self.msg.emit("Finished tSNE! \n -------------------------")

        if self.technique == 3:
            self.msg.emit("Running PHATE... ")
            QtWidgets.qApp.processEvents()
            import phate
            operator = phate.PHATE(n_components=2,
                                   decay=10,
                                   knn=5,
                                   knn_max=15,
                                   t=30,
                                   mds="classic",
                                   knn_dist="euclidean",
                                   mds_dist="euclidean",
                                   n_jobs=-2,
                                   n_landmark=None,
                                   verbose=True)
            ETA, t0 = self.estimate_time(operator)
            self.embedding = operator.fit_transform(z_space)
            self.msg.emit("Finished PHATE! \n -------------------------")

        if self.technique == 4:
            self.msg.emit("Running cVAE... ")
            QtWidgets.qApp.processEvents()
            from cvae import cvae
            operator = cvae.CompressionVAE(z_space)
            operator.train()
            ETA, t0 = self.estimate_time(operator)
            self.embedding = operator.embed(z_space)
            self.msg.emit("Finished cVAE! \n -------------------------")

        self.progress.emit(ETA, t0, False, False, True)
        self.finished.emit()
        self.status.emit(True)

class Clusterer(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(float, float, bool, bool, bool)
    msg = QtCore.pyqtSignal(str)
    status = QtCore.pyqtSignal(bool)
    def __init__(self, data, clusters, method):
        self.data = data
        self.clusters = clusters
        self.method = method
        self.labels = ''
        super().__init__()

    def run_clustering(self):
        self.status.emit(False)
        '''
        self.comboBox_4.setItemText(0, _translate("MainWindow", "KMeans (fast)"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "Affinity Propagation (slow)"))
        self.comboBox_4.setItemText(2, _translate("MainWindow", "MeanShift (slow)"))
        self.comboBox_4.setItemText(3, _translate("MainWindow", "Spectral Clustering (slow)"))
        self.comboBox_4.setItemText(4, _translate("MainWindow", "Ward (fast)"))
        self.comboBox_4.setItemText(5, _translate("MainWindow", "Agglomerative Clustering (fast)"))
        self.comboBox_4.setItemText(6, _translate("MainWindow", "DBSCAN (fast)"))
        self.comboBox_4.setItemText(7, _translate("MainWindow", "OPTICS (slow)"))
        self.comboBox_4.setItemText(8, _translate("MainWindow", "BIRCH (fast)"))
        self.comboBox_4.setItemText(9, _translate("MainWindow", "Gaussian Mixture (fast)"))
        '''
        # global labels

        from sklearn import cluster, mixture
        from sklearn.neighbors import kneighbors_graph
        self.msg.emit('Begin clustering ...')

        #Subset for ETA calculation - at the expense of a small amount of time
        fraction = 0.01
        subset = int(len(self.data) * fraction)
        idx = np.random.randint(self.data.shape[0], size=subset)
        small_data = self.data[idx]

        # ============
        # Create cluster objects from sklearn clustering example
        # ============

        # connectivity matrix for structured Ward
        if self.method == 4 or self.method == 5:
            connectivity = kneighbors_graph(self.data, n_neighbors=5, include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)
        else:
            connectivity = []

        # estimate bandwidth for mean shift
        if self.method == 2:
            bandwidth = cluster.estimate_bandwidth(self.data, quantile=0.3)
        else:
            bandwidth = []


        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

        two_means = cluster.MiniBatchKMeans(n_clusters=self.clusters)

        ward = cluster.AgglomerativeClustering(n_clusters=self.clusters, linkage='ward', connectivity=connectivity)

        spectral = cluster.SpectralClustering(n_clusters=self.clusters, eigen_solver='arpack',
            affinity="nearest_neighbors")

        dbscan = cluster.DBSCAN(eps=2, min_samples=50, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)

        optics = cluster.OPTICS(min_samples=20, xi=0.05,
                                min_cluster_size=100)

        affinity_propagation = cluster.AffinityPropagation(damping=0.6,
                                                           preference=200)

        average_linkage = cluster.AgglomerativeClustering(linkage="complete", n_clusters=self.clusters)

        birch = cluster.Birch(n_clusters=self.clusters)

        gmm = mixture.GaussianMixture(n_components=self.clusters, covariance_type='full')

        clustering_algorithms = (
            ('MiniBatch KMeans', two_means),
            ('Affinity Propagation', affinity_propagation),
            ('MeanShift', ms),
            ('Spectral Clustering', spectral),
            ('Ward', ward),
            ('Agglomerative Clustering', average_linkage),
            ('DBSCAN', dbscan),
            ('OPTICS', optics),
            ('BIRCH', birch),
            ('Gaussian Mixture', gmm)
        )

        name, operator = clustering_algorithms[self.method]

        self.msg.emit('Calculating ' + name + ' ...')

        '''
        THIS NEED FIXING...
            - Check size of data
            - Warn user for too large data
            - Estimate memory requirements... ???
            - Estimate time (??? linear, exponential, etc based on O() of algorithm)
        '''
        ### TIMING TEST
        t0 = time.time()
        #dummy = operator.fit(small_data)
        time.sleep(1)
        ETA = (1 / fraction) * (time.time() - t0)
        self.progress.emit(ETA, t0, True, True, False)


        #Fit full dataset
        operator.fit(self.data)

        #Generate labels list
        if hasattr(operator, 'labels_'):
            self.labels = operator.labels_.astype(int)
        else:
            self.labels = operator.predict(self.data)

        #Finished close thread and emit complete signal
        self.msg.emit('Finished ' + name + ' ! \n -------------------------')
        self.progress.emit(1, 1, False, False, True)
        self.finished.emit()
        self.status.emit(True)

class Ui_MainWindow(object):
    def __init__(self, session):
        self.state = 'idle'
        self.inputType = {
            0 : 'cryodrgn',
            1 : 'cryosparc_3dva',
            2 : 'cryosparc_3dflx'
        }
        self.resolution = {
            0: 96,
            1: 128,
            2: 156,
            3: 196,
            4: 256
        }
        self.data_path = ''
        self.subset_state = None
        self.previous_data = None
        self.wiggle = session

    def browse_dir(self, comment, button):
        dirName = str(QtWidgets.QFileDialog.getExistingDirectory(None, comment, ""))
        if dirName:
            self.string = dirName + '/'
            if button == self.pushButton_3:
                self.lineEdit.setText(self.string)

    def input_type(self):
        type = self.inputType[self.comboBox.currentIndex()]
        single_input = self.checkBox_14.isChecked()

        if single_input:
            if type == 'cryodrgn':
                self.stackedWidget.setCurrentIndex(4)
                self.data_path = [self.lineEdit_18.text()]
            elif type == 'cryosparc_3dva':
                self.stackedWidget.setCurrentIndex(3)
                self.data_path = [self.lineEdit_16.text()]
            elif type == 'cryosparc_3dflx':
                self.stackedWidget.setCurrentIndex(5)
        else:
            self.stackedWidget.setCurrentIndex(self.comboBox.currentIndex())
            if type == 'cryodrgn':
                self.data_path = [self.lineEdit_6.text(), self.lineEdit_5.text(), self.lineEdit_4.text()]
            elif type == 'cryosparc_3dva':
                self.data_path = [self.lineEdit_15.text(), self.lineEdit_13.text(), self.lineEdit_14.text()]
            elif type == 'cryosparc_3dflx':
                pass

    def browse_file(self, comment, button, type):
        if button == self.pushButton_9:
            fileName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, comment, "", type,
                                                                options=QtWidgets.QFileDialog.DontUseNativeDialog)
        else:
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, comment, "", type,
                                                                    options=QtWidgets.QFileDialog.DontUseNativeDialog)

        if fileName:
            #cryoSPARC 3D flex
            if button == self.pushButton_3:
                self.lineEdit.setText(fileName)
            elif button == self.pushButton_4:
                self.lineEdit_2.setText(fileName)

            #cryoSPARC 3DVA
            elif button == self.pushButton_5:
                self.lineEdit_13.setText(fileName)
            elif button == self.pushButton_9:
                self.lineEdit_14.setText(','.join(fileName))
            elif button == self.pushButton_13:
                self.lineEdit_15.setText(fileName)
                self.comboBox_3.setEnabled(True)

            #cryoSPARC single file format
            elif button == self.pushButton_16:
                self.lineEdit_16.setText(fileName)
                self.comboBox_3.setEnabled(True)

            #cryoDRGN inputs
            elif button == self.pushButton_6: #To do: make this a requirement for volumiser()
                self.lineEdit_6.setText(fileName)
                # self.w.config = fileName
            elif button == self.pushButton_7: #To do: make a requirement for volumiser()
                self.lineEdit_5.setText(fileName)
                # self.w.weights = fileName
            elif button == self.pushButton_8: #z_space minimal requirement for dimensionality reduction.
                self.lineEdit_4.setText(fileName)
                self.comboBox_3.setEnabled(True)

            # cryoDRGN single file format
            elif button == self.pushButton_21:
                self.lineEdit_18.setText(fileName)
                self.comboBox_3.setEnabled(True)

            self.input_type()

    def status(self, kill: bool):
        if self.state == 'busy':
            self.state = ''
        elif self.state == '':
            self.state = 'busy'
        elif self.state == 'idle':
            self.state = 'busy'

        if not kill:
            self.statusTimer.start(700)
            self.statusBar.setStyleSheet("background-color : pink")
        else:
            self.state = 'idle'
            self.statusBar.setStyleSheet("background-color: rgb(239, 239, 239);")
            self.statusTimer.stop()

        self.statusBar.showMessage("Status: " + str(self.state))

    def reportProgress(self, ETA, t0, init: bool, ping: bool, kill: bool):
        if init:
            self.ETA = ETA
            self.t0 = t0
            progText = 'Estimated time for completion: ' + str(self.ETA/60)[0:5] + ' minutes.'
            self.textBrowser.append(progText)

        if ping:
            try:
                ETA = self.ETA
                t0 = self.t0
                if (time.time() - t0 < ETA):
                    progress = 100 * ((time.time() - t0) / ETA)
                    self.progressBar_7.setValue(int(progress))
                    self.timer.start(50)
                    QtWidgets.qApp.processEvents()
            except:
                print('no good - problem in reportProgress function')

        if kill:
            self.progressBar_7.setValue(int(100))
            self.timer.stop()

    def launchEmbedding(self):
        self.progressBar_7.reset()
        self.pushButton_19.setEnabled(False)
        # Step 2: Create a QThread object
        self.thread = QtCore.QThread()
        # Step 3: Create a worker object
        if self.checkBox_12.checkState() == QtCore.Qt.Checked and self.lineEdit_3.text().isnumeric():
            self.e = Embedder(
                self.comboBox_3.currentIndex(),
                self.data_path,
                self.lineEdit_3.text(),
                True,
                self.subset_state,
                self.previous_data
            )
        else:
            self.e = Embedder(
                self.comboBox_3.currentIndex(),
                self.data_path,
                self.lineEdit_3.text(),
                False
            )

        # Step 4: Move worker to the thread
        self.e.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.e.run_embedding)
        self.e.exit.connect(self.thread.quit)
        self.e.exit.connect(self.e.deleteLater)
        self.e.finished.connect(self.thread.quit)
        self.e.finished.connect(self.e.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.e.progress.connect(self.reportProgress)
        self.e.status.connect(self.status)
        self.e.msg.connect(self.textBrowser.append)
        # Step 6: Start the thread
        self.thread.start()

        # Final resets
        def update_interactive_window():
            self.w.embedding = self.e.embedding
            self.state = False
            self.previous_data = self.e.data[2]
            self.w.data = self.e.data

        self.e.finished.connect(
            lambda : update_interactive_window()
        )

        self.e.finished.connect(
            lambda: self.comboBox_2.setEnabled(True)
        )

        self.e.finished.connect(
            lambda : self.progressBar_7.setValue(100))

        self.e.finished.connect(
            lambda: self.pushButton_19.setEnabled(True)
        )

        self.e.exit.connect(
            lambda: self.pushButton_19.setEnabled(True)
        )

        self.e.finished.connect(
            lambda: self.w.modify_state('space', True)
        )

        self.e.finished.connect(
            lambda: self.w.initialise_volumiser()
        )

        self.e.finished.connect(self.w.plotArea.autoRange)

    def launchKmeans(self):
        self.progressBar_7.reset()
        self.pushButton_23.setEnabled(False)
        # Step 2: Create a QThread object
        self.thread2 = QtCore.QThread()
        # Step 3: Create a worker object
        self.c = Clusterer(self.e.data[2], self.spinBox_2.value(), self.comboBox_4.currentIndex())
        # Step 4: Move worker to the thread
        self.c.moveToThread(self.thread2)
        # Step 5: Connect signals and slots
        self.thread2.started.connect(self.c.run_clustering)
        self.c.finished.connect(self.thread2.quit)
        self.c.finished.connect(self.c.deleteLater)
        self.thread2.finished.connect(self.thread2.deleteLater)
        self.c.progress.connect(self.reportProgress)
        self.c.msg.connect(self.textBrowser.append)
        self.c.status.connect(self.status)
        # Step 6: Start the thread
        self.thread2.start()

        # Final resets
        def update_interactive_window():
            self.w.labels = self.c.labels

        self.c.finished.connect(
            lambda : update_interactive_window()
        )

        self.c.finished.connect(
            lambda: self.checkBox_5.setEnabled(True)
        )


        self.c.finished.connect(
            lambda : self.w.modify_state('space', self.checkBox_5.isChecked())
        )

        self.c.finished.connect(
            lambda: self.progressBar_7.setValue(100)
        )

        self.c.finished.connect(
            lambda: self.pushButton_23.setEnabled(True)
        )

    def launchMEP(self):
        from pthinf.path_inference_dijkstra import Path
        self.progressBar_7.reset()
        self.pushButton_23.setEnabled(False)
        # Step 2: Create a QThread object
        self.thread3 = QtCore.QThread()
        # Step 3: Create a worker object
        self.p = Path(
            self.w.embedding,
            float(self.lineEdit_8.text()),
            int(self.comboBox_7.currentText()),
            int(self.comboBox_8.currentText()),
            self.resolution[self.comboBox_9.currentIndex()],
            int(self.lineEdit_7.text()),
            int(self.lineEdit_17.text()),
            float(self.lineEdit_10.text()),
            float(self.lineEdit_11.text()),
        )
        # Step 4: Move worker to the thread
        self.p.moveToThread(self.thread3)
        # Step 5: Connect signals and slots
        self.p.finished.connect(self.thread3.quit)
        self.p.finished.connect(self.p.deleteLater)
        self.thread3.finished.connect(self.thread3.deleteLater)
        self.p.progress.connect(self.reportProgress)
        self.p.msg.connect(self.textBrowser.append)
        self.p.status.connect(self.status)

        self.thread3.started.connect(self.p.costFuncApproach)
        # Step 6: Start the thread
        self.thread3.start()

        # Final resets
        def update_interactive_window():
            self.w.MEP_TrajLists = self.p.trajectories

        self.p.finished.connect(
            lambda : update_interactive_window()
        )

        self.p.finished.connect(
            lambda: self.MEP_path_display()
        )

        self.p.finished.connect(
            lambda : self.w.modify_state('MEP', self.checkBox_13.isChecked())
        )

        self.p.finished.connect(
            lambda: self.progressBar_7.setValue(100)
        )

        self.p.finished.connect(
            lambda: self.pushButton_23.setEnabled(True)
        )

    def MEP_path_display(self):
        self.w.MEP_legendItemsTraj = []
        for _, item in enumerate(self.w.MEP_TrajLists):
            self.w.MEP_legendItemsTraj.append("path_" + str(_))
        self.w.currentMEPlegenddisplay = self.w.MEP_legendItemsTraj

        self.comboBox_6.clear()
        self.comboBox_6.addItem('all')
        [self.comboBox_6.addItem(item) for item in self.w.MEP_legendItemsTraj]
        self.comboBox_6.setEnabled(True)
        self.checkBox_11.setEnabled(True)
        self.checkBox_13.setEnabled(True)
        self.pushButton_12.setEnabled(True)

    def change_current_MEP_view(self):
        if self.comboBox_6.currentIndex() == 0:
            self.w.currentMEPdisplay = self.w.MEP_TrajLists
            self.w.modify_state('MEP', self.checkBox_13.isChecked())
            self.w.currentMEPlegenddisplay = self.w.MEP_legendItemsTraj
        else:
            self.w.currentMEPlegenddisplay = [self.w.MEP_legendItemsTraj[self.comboBox_6.currentIndex() - 1]]
            self.w.currentMEPdisplay = [self.w.MEP_TrajLists[self.comboBox_6.currentIndex() - 1]]
            self.w.modify_state('MEP', self.checkBox_13.isChecked())

    def user_anchor_points(self, query: bool, save: bool, reset: bool, clear: bool):
        if query:
            flat_list = [item for sublist in self.w.current_anchors for item in sublist]
            if len(flat_list) > 0:
                self.w.TrajLists.append(flat_list)
                self.w.legendItemsTraj = []
                for _, item in enumerate(self.w.TrajLists):
                    self.w.legendItemsTraj.append("trajectory_" + str(_))
            self.w.current_anchors = []
            print('User selection of anchor points: ' + ' '.join([str(elem) for elem in flat_list]))

        if clear:
            self.w.current_anchors = []

        if reset:
            for curvePlot, curvePlot_pt in zip(self.w.curvePlotsUSER, self.w.curvePlotsUSER_pts):
                self.w.plotArea.removeItem(curvePlot)
                self.w.plotArea.removeItem(curvePlot_pt)
            for morph_slider in self.w.slider_list:
                morph_slider.delete()
            self.w.TrajLists = []
            self.w.slider_list = []

        if save:
            filename = QtWidgets.QFileDialog.getSaveFileName(None, "Save file", "", ".txt", options=QtWidgets.QFileDialog.DontUseNativeDialog)
            _filename = filename[0]+filename[1]
            with open(_filename, 'w') as f:
                for name_index,trajectory in enumerate(self.w.TrajLists):
                    for coord in trajectory:
                        f.write(str(coord) + ',')
                    f.write(str(self.w.legendItemsTraj[name_index]))
                    f.write('\n')

        self.w.modify_state('trajectories', self.checkBox_8.isChecked())
        self.lineEdit_12.setText(str(len(self.w.current_anchors)))

        if query and len(self.w.legendItemsTraj) > 0:
            self.comboBox_10.clear()
            [self.comboBox_10.addItem(item) for item in self.w.legendItemsTraj]
            self.comboBox_10.setEnabled(True)
            self.pushButton_20.setEnabled(True)
            self.checkBox_19.setEnabled(True)
            self.checkBox_20.setEnabled(True)
        elif query and len(self.w.legendItemsTraj) == 0 or reset:
            self.comboBox_10.clear()
            self.comboBox_10.setEnabled(False)
            self.pushButton_20.setEnabled(False)
            self.checkBox_19.setEnabled(False)
            self.checkBox_20.setEnabled(False)

    # NEEDS ADDRESSING # CBJ
    def write_output_file(self, indices, type):
        pass

    # NEEDS ADDRESSING # CBJ
    def export_roi(self):
        if self.radioButton_3.isChecked():
            if self.checkBox_7.isChecked():
                select = [pt for subPlot in self.w.scatterPlots
                          if subPlot.isVisible()
                          for pt in subPlot.points()
                          if not self.w.ROIs.mapToItem(subPlot, self.w.ROIs.shape()).contains(pt.pos())]
            else:
                select = [pt for subPlot in self.w.scatterPlots
                          if subPlot.isVisible()
                          for pt in subPlot.points()
                          if self.w.ROIs.mapToItem(subPlot, self.w.ROIs.shape()).contains(pt.pos())]

            indices_ROI = []
            for pt in select:
                indices_ROI.append(pt.index())

            #DO something with the indicies!

            self.textBrowser.append("A total of " + str(len(indices_ROI)) + " particles were exported and converted to RELION and cryoSPARC formats."
                                                                            "\n -------------------------")

    # NEEDS ADDRESSING # CBJ
    def export_clusters(self):
        if self.radioButton_4.isChecked():
            try:
                self.w.labels
            except:
                self.textBrowser.append("You must run clustering before you can export observation clusters "
                                        "\n -------------------------")
            else:
                cluster_list = []
                for label in set(self.w.labels):
                    cluster = [pt for subPlot in self.w.scatterPlots for pt in subPlot.points() if self.w.labels[pt.data()] == label]
                    cluster_list.append(cluster)

                self.textBrowser.append("Exporting particles as cryoSPARC and RELION formats...")

                for num, cluster in enumerate(cluster_list):
                    indices_cluster = []
                    for pt in cluster:
                        indices_cluster.append(pt.index())

                ###DO something with the indicies!

                    self.textBrowser.append("\t Done 'cluster_" + str(num) + "'. Contains " + str(len(indices_cluster)) + " particles.")
            self.textBrowser.append("-------------------------")

    # def return_pressed(self):
    #     t0 = time.time()
    #     self.apix = 1
    #
    #     # The use has pressed the Return key; log the current text as HTML
    #     from .cryodrgn_minimal import miniDRGN
    #     from chimerax.map import volume_from_grid_data
    #     from chimerax.map_data import ArrayGridData
    #     from chimerax.core.models import Model
    #
    #     Z = data[self.currentZind]
    #
    #     # ToolInstance has a 'session' attribute...
    #     miniDRGN = miniDRGN(
    #         self.lineEdit_6.displayText(),
    #         Z,
    #         self.lineEdit_5.displayText(),
    #         apix=self.apix)
    #
    #     mod = []
    #     if self.checkBox_3.checkState() == QtCore.Qt.Checked:
    #         for mid in self.wiggle.models:
    #             print(mid.id)
    #             mod.append(mid)
    #         self.wiggle.models.close(mod)
    #     self.vol = volume_from_grid_data(ArrayGridData(miniDRGN.vol, name='cryoDRGN vol'), self.wiggle,
    #                                      open_model=True)

    def setupUi(self, MainWindow):

        def _night_mode():
            if self.checkBox_17.isChecked():
                self.w.graphicsView.setBackground(background='k')
                palette = QtGui.QPalette()
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
                brush = QtGui.QBrush(QtGui.QColor(76, 76, 76))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
                brush = QtGui.QBrush(QtGui.QColor(63, 63, 63))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
                brush = QtGui.QBrush(QtGui.QColor(34, 34, 34))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
                brush = QtGui.QBrush(QtGui.QColor(189, 194, 185))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Highlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
                brush = QtGui.QBrush(QtGui.QColor(76, 76, 76))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
                brush = QtGui.QBrush(QtGui.QColor(63, 63, 63))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
                brush = QtGui.QBrush(QtGui.QColor(34, 34, 34))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
                brush = QtGui.QBrush(QtGui.QColor(189, 194, 185))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Highlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
                brush = QtGui.QBrush(QtGui.QColor(76, 76, 76))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
                brush = QtGui.QBrush(QtGui.QColor(63, 63, 63))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
                brush = QtGui.QBrush(QtGui.QColor(34, 34, 34))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
                brush = QtGui.QBrush(QtGui.QColor(25, 25, 25))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
                brush = QtGui.QBrush(QtGui.QColor(174, 174, 174))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Highlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(51, 51, 51))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
            else:
                self.w.graphicsView.setBackground(background='w')
                palette = QtGui.QPalette()
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
                brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(112, 112, 112))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
                brush = QtGui.QBrush(QtGui.QColor(150, 150, 150))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
                brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
                brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(112, 112, 112))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
                brush = QtGui.QBrush(QtGui.QColor(150, 150, 150))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
                brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
                brush = QtGui.QBrush(QtGui.QColor(112, 112, 112))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
                brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
                brush = QtGui.QBrush(QtGui.QColor(112, 112, 112))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
                brush = QtGui.QBrush(QtGui.QColor(150, 150, 150))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
                brush = QtGui.QBrush(QtGui.QColor(112, 112, 112))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
                brush = QtGui.QBrush(QtGui.QColor(112, 112, 112))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
                brush = QtGui.QBrush(QtGui.QColor(225, 225, 225))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
                brush.setStyle(QtCore.Qt.SolidPattern)
                palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
            self.palette = palette
            MainWindow.setPalette(self.palette)
            MainWindow.setCentralWidget(self.centralwidget)

        '''
        ###########################################################################################################
        Build the main window and define all attributes for Qt
        :param MainWindow:
        :return:
        ###########################################################################################################
        '''
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1008, 872)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1008, 872))
        MainWindow.setMaximumSize(QtCore.QSize(1008, 872))

        MainWindow.setWindowOpacity(1.0)
        MainWindow.setWhatsThis("")
        MainWindow.setStyleSheet("")
        MainWindow.setDocumentMode(False)

        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)

        self.timer = QtCore.QTimer()
        self.onlyInt = QtGui.QIntValidator()
        self.onlyFloat = QtGui.QDoubleValidator()
        self.statusTimer = QtCore.QTimer()
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(0, 750, 796, 96))
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_3.setEnabled(False)
        self.textBrowser_3.setGeometry(QtCore.QRect(800, 780, 206, 66))
        self.textBrowser_3.setAcceptDrops(True)
        self.textBrowser_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 0, 801, 91))
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.pushButton_8 = QtWidgets.QPushButton(self.page_2)
        self.pushButton_8.setGeometry(QtCore.QRect(0, 30, 196, 21))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_6 = QtWidgets.QPushButton(self.page_2)
        self.pushButton_6.setGeometry(QtCore.QRect(0, 5, 196, 21))
        self.pushButton_6.setObjectName("pushButton_6")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_5.setGeometry(QtCore.QRect(200, 55, 596, 21))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.pushButton_7 = QtWidgets.QPushButton(self.page_2)
        self.pushButton_7.setGeometry(QtCore.QRect(0, 55, 196, 21))
        self.pushButton_7.setObjectName("pushButton_7")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_6.setGeometry(QtCore.QRect(200, 5, 596, 21))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_4.setGeometry(QtCore.QRect(200, 30, 596, 21))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.stackedWidget.addWidget(self.page_2)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.lineEdit_13 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_13.setGeometry(QtCore.QRect(200, 5, 596, 21))
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.pushButton_5 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_5.setGeometry(QtCore.QRect(0, 5, 196, 21))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_9 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_9.setGeometry(QtCore.QRect(0, 30, 196, 21))
        self.pushButton_9.setObjectName("pushButton_9")
        self.lineEdit_14 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_14.setGeometry(QtCore.QRect(200, 30, 596, 21))
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.pushButton_13 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_13.setGeometry(QtCore.QRect(0, 55, 196, 21))
        self.pushButton_13.setObjectName("pushButton_13")
        self.lineEdit_15 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_15.setGeometry(QtCore.QRect(200, 55, 596, 21))
        self.lineEdit_15.setObjectName("lineEdit_15")
        self.stackedWidget.addWidget(self.page_4)
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.pushButton_3 = QtWidgets.QPushButton(self.page)
        self.pushButton_3.setEnabled(False)
        self.pushButton_3.setGeometry(QtCore.QRect(0, 5, 196, 21))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.page)
        self.pushButton_4.setEnabled(False)
        self.pushButton_4.setGeometry(QtCore.QRect(0, 30, 196, 21))
        self.pushButton_4.setObjectName("pushButton_4")
        self.lineEdit = QtWidgets.QLineEdit(self.page)
        self.lineEdit.setEnabled(False)
        self.lineEdit.setGeometry(QtCore.QRect(200, 5, 596, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_2.setEnabled(False)
        self.lineEdit_2.setGeometry(QtCore.QRect(200, 30, 596, 21))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_41 = QtWidgets.QLabel(self.page)
        self.label_41.setGeometry(QtCore.QRect(5, 55, 276, 21))
        self.label_41.setFont(font)
        self.label_41.setObjectName("label_41")
        self.stackedWidget.addWidget(self.page)
        self.page_5 = QtWidgets.QWidget()
        self.page_5.setObjectName("page_5")
        self.pushButton_16 = QtWidgets.QPushButton(self.page_5)
        self.pushButton_16.setGeometry(QtCore.QRect(0, 5, 196, 21))
        self.pushButton_16.setObjectName("pushButton_16")
        self.lineEdit_16 = QtWidgets.QLineEdit(self.page_5)
        self.lineEdit_16.setGeometry(QtCore.QRect(200, 5, 596, 21))
        self.lineEdit_16.setObjectName("lineEdit_16")
        self.stackedWidget.addWidget(self.page_5)
        self.page_6 = QtWidgets.QWidget()
        self.page_6.setObjectName("page_6")
        self.pushButton_21 = QtWidgets.QPushButton(self.page_6)
        self.pushButton_21.setGeometry(QtCore.QRect(0, 5, 196, 21))
        self.pushButton_21.setObjectName("pushButton_21")
        self.lineEdit_18 = QtWidgets.QLineEdit(self.page_6)
        self.lineEdit_18.setGeometry(QtCore.QRect(200, 5, 596, 21))
        self.lineEdit_18.setObjectName("lineEdit_18")
        self.stackedWidget.addWidget(self.page_6)

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(805, 120, 191, 23))
        self.comboBox.setObjectName("comboBox")
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(176, 154, 225))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        self.comboBox.setPalette(palette)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")

        self.comboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_3.setEnabled(False)
        self.comboBox_3.setGeometry(QtCore.QRect(805, 182, 146, 23))
        self.comboBox_3.setObjectName("comboBox_3")
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(243, 129, 129))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(243, 129, 129))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        self.comboBox_3.setPalette(palette)
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")

        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setEnabled(False)
        self.comboBox_2.setGeometry(QtCore.QRect(805, 232, 191, 23))
        self.comboBox_2.setToolTip("")
        self.comboBox_2.setObjectName("comboBox_2")
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(142, 218, 248))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        self.comboBox_2.setPalette(palette)
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")

        self.stackedWidget_3 = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget_3.setEnabled(True)
        self.stackedWidget_3.setGeometry(QtCore.QRect(795, 260, 221, 381))
        self.stackedWidget_3.setObjectName("stackedWidget_3")
        self.page_12 = QtWidgets.QWidget()
        self.page_12.setObjectName("page_12")
        self.checkBox = QtWidgets.QCheckBox(self.page_12)
        self.checkBox.setGeometry(QtCore.QRect(10, 25, 146, 21))
        self.checkBox.setChecked(True)
        self.checkBox.setObjectName("checkBox")
        self.label_8 = QtWidgets.QLabel(self.page_12)
        self.label_8.setGeometry(QtCore.QRect(10, -10, 221, 31))
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.line_4 = QtWidgets.QFrame(self.page_12)
        self.line_4.setGeometry(QtCore.QRect(10, 10, 201, 20))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.pushButton_27 = QtWidgets.QPushButton(self.page_12)
        self.pushButton_27.setEnabled(False)
        self.pushButton_27.setGeometry(QtCore.QRect(110, 45, 86, 26))
        self.pushButton_27.setObjectName("pushButton_27")
        self.checkBox_3 = QtWidgets.QCheckBox(self.page_12)
        self.checkBox_3.setGeometry(QtCore.QRect(10, 75, 181, 21))
        self.checkBox_3.setChecked(False)
        self.checkBox_3.setObjectName("checkBox_3")
        self.stackedWidget_3.addWidget(self.page_12)
        self.page_16 = QtWidgets.QWidget()
        self.page_16.setObjectName("page_16")
        self.label_7 = QtWidgets.QLabel(self.page_16)
        self.label_7.setGeometry(QtCore.QRect(10, -10, 171, 31))
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_11 = QtWidgets.QLabel(self.page_16)
        self.label_11.setGeometry(QtCore.QRect(15, 60, 121, 31))
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.spinBox_2 = QtWidgets.QSpinBox(self.page_16)
        self.spinBox_2.setGeometry(QtCore.QRect(140, 60, 61, 31))
        self.spinBox_2.setProperty("value", 10)
        self.spinBox_2.setObjectName("spinBox_2")
        self.pushButton_23 = QtWidgets.QPushButton(self.page_16)
        self.pushButton_23.setGeometry(QtCore.QRect(55, 235, 75, 23))
        self.pushButton_23.setObjectName("pushButton_23")
        self.pushButton_24 = QtWidgets.QPushButton(self.page_16)
        self.pushButton_24.setGeometry(QtCore.QRect(135, 235, 75, 23))
        self.pushButton_24.setObjectName("pushButton_24")
        self.line_3 = QtWidgets.QFrame(self.page_16)
        self.line_3.setGeometry(QtCore.QRect(10, 10, 201, 20))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_5 = QtWidgets.QFrame(self.page_16)
        self.line_5.setGeometry(QtCore.QRect(5, 220, 201, 20))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.checkBox_5 = QtWidgets.QCheckBox(self.page_16)
        self.checkBox_5.setGeometry(QtCore.QRect(5, 95, 176, 21))
        self.checkBox_5.setObjectName("checkBox_5")
        self.checkBox_6 = QtWidgets.QCheckBox(self.page_16)
        self.checkBox_6.setGeometry(QtCore.QRect(5, 115, 231, 31))
        self.checkBox_6.setObjectName("checkBox_6")
        self.comboBox_4 = QtWidgets.QComboBox(self.page_16)
        self.comboBox_4.setEnabled(True)
        self.comboBox_4.setGeometry(QtCore.QRect(5, 25, 201, 26))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_5 = QtWidgets.QComboBox(self.page_16)
        self.comboBox_5.setEnabled(True)
        self.comboBox_5.setGeometry(QtCore.QRect(5, 170, 201, 26))
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.checkBox_2 = QtWidgets.QCheckBox(self.page_16)
        self.checkBox_2.setEnabled(False)
        self.checkBox_2.setGeometry(QtCore.QRect(5, 145, 221, 21))
        self.checkBox_2.setWhatsThis("")
        self.checkBox_2.setAccessibleDescription("")
        self.checkBox_2.setObjectName("checkBox_2")
        self.stackedWidget_3.addWidget(self.page_16)
        self.page_13 = QtWidgets.QWidget()
        self.page_13.setObjectName("page_13")
        self.pushButton = QtWidgets.QPushButton(self.page_13)
        self.pushButton.setGeometry(QtCore.QRect(55, 235, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.label_5 = QtWidgets.QLabel(self.page_13)
        self.label_5.setGeometry(QtCore.QRect(10, -10, 221, 31))
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.pushButton_2 = QtWidgets.QPushButton(self.page_13)
        self.pushButton_2.setGeometry(QtCore.QRect(135, 235, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_6 = QtWidgets.QLabel(self.page_13)
        self.label_6.setGeometry(QtCore.QRect(20, 150, 131, 21))
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_10 = QtWidgets.QLabel(self.page_13)
        self.label_10.setGeometry(QtCore.QRect(95, 95, 41, 31))
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.line = QtWidgets.QFrame(self.page_13)
        self.line.setGeometry(QtCore.QRect(10, 220, 196, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.page_13)
        self.line_2.setGeometry(QtCore.QRect(10, 10, 201, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.spinBox_4 = QtWidgets.QSpinBox(self.page_13)
        self.spinBox_4.setGeometry(QtCore.QRect(165, 25, 46, 26))
        self.spinBox_4.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_4.setMaximum(3)
        self.spinBox_4.setObjectName("spinBox_4")
        self.radioButton = QtWidgets.QRadioButton(self.page_13)
        self.radioButton.setGeometry(QtCore.QRect(5, 25, 166, 26))
        self.radioButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.radioButton.setChecked(True)
        self.radioButton.setAutoExclusive(True)
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.page_13)
        self.radioButton_2.setGeometry(QtCore.QRect(5, 125, 166, 22))
        self.radioButton_2.setObjectName("radioButton_2")
        self.checkBox_8 = QtWidgets.QCheckBox(self.page_13)
        self.checkBox_8.setGeometry(QtCore.QRect(5, 180, 156, 22))
        self.checkBox_8.setObjectName("checkBox_8")
        self.checkBox_10 = QtWidgets.QCheckBox(self.page_13)
        self.checkBox_10.setGeometry(QtCore.QRect(5, 200, 156, 22))
        self.checkBox_10.setObjectName("checkBox_10")
        self.line_7 = QtWidgets.QFrame(self.page_13)
        self.line_7.setGeometry(QtCore.QRect(5, 100, 81, 20))
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.line_8 = QtWidgets.QFrame(self.page_13)
        self.line_8.setGeometry(QtCore.QRect(120, 100, 81, 20))
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.lineEdit_12 = QtWidgets.QLineEdit(self.page_13)
        self.lineEdit_12.setEnabled(True)
        self.lineEdit_12.setGeometry(QtCore.QRect(145, 150, 61, 21))
        self.lineEdit_12.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_12.setReadOnly(True)
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.label_20 = QtWidgets.QLabel(self.page_13)
        self.label_20.setGeometry(QtCore.QRect(40, 55, 121, 26))
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.spinBox_3 = QtWidgets.QSpinBox(self.page_13)
        self.spinBox_3.setGeometry(QtCore.QRect(165, 55, 46, 26))
        self.spinBox_3.setAlignment(QtCore.Qt.AlignCenter)
        self.spinBox_3.setProperty("value", 8)
        self.spinBox_3.setObjectName("spinBox_3")
        self.pushButton_17 = QtWidgets.QPushButton(self.page_13)
        self.pushButton_17.setGeometry(QtCore.QRect(55, 265, 75, 23))
        self.pushButton_17.setObjectName("pushButton_17")
        self.pushButton_18 = QtWidgets.QPushButton(self.page_13)
        self.pushButton_18.setGeometry(QtCore.QRect(135, 265, 75, 23))
        self.pushButton_18.setObjectName("pushButton_18")
        self.label_21 = QtWidgets.QLabel(self.page_13)
        self.label_21.setGeometry(QtCore.QRect(10, 290, 181, 21))
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.comboBox_10 = QtWidgets.QComboBox(self.page_13)
        self.comboBox_10.setEnabled(False)
        self.comboBox_10.setGeometry(QtCore.QRect(5, 310, 166, 26))
        self.comboBox_10.setObjectName("comboBox_10")
        self.pushButton_20 = QtWidgets.QPushButton(self.page_13)
        self.pushButton_20.setEnabled(False)
        self.pushButton_20.setGeometry(QtCore.QRect(175, 310, 36, 26))
        self.pushButton_20.setObjectName("pushButton_20")
        self.checkBox_19 = QtWidgets.QCheckBox(self.page_13)
        self.checkBox_19.setEnabled(False)
        self.checkBox_19.setGeometry(QtCore.QRect(3, 335, 101, 22))
        self.checkBox_19.setChecked(True)
        self.checkBox_19.setObjectName("checkBox_19")
        self.checkBox_20 = QtWidgets.QCheckBox(self.page_13)
        self.checkBox_20.setEnabled(False)
        self.checkBox_20.setGeometry(QtCore.QRect(105, 335, 101, 22))
        self.checkBox_20.setChecked(False)
        self.checkBox_20.setObjectName("checkBox_20")
        self.stackedWidget_3.addWidget(self.page_13)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.label_9 = QtWidgets.QLabel(self.page_3)
        self.label_9.setGeometry(QtCore.QRect(10, -10, 181, 31))
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.line_6 = QtWidgets.QFrame(self.page_3)
        self.line_6.setGeometry(QtCore.QRect(5, 10, 201, 20))
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.line_12 = QtWidgets.QFrame(self.page_3)
        self.line_12.setGeometry(QtCore.QRect(10, 235, 201, 20))
        self.line_12.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        self.pushButton_10 = QtWidgets.QPushButton(self.page_3)
        self.pushButton_10.setGeometry(QtCore.QRect(55, 250, 75, 23))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.page_3)
        self.pushButton_11.setGeometry(QtCore.QRect(135, 250, 75, 23))
        self.pushButton_11.setObjectName("pushButton_11")
        self.comboBox_6 = QtWidgets.QComboBox(self.page_3)
        self.comboBox_6.setEnabled(False)
        self.comboBox_6.setGeometry(QtCore.QRect(5, 300, 166, 26))
        self.comboBox_6.setObjectName("comboBox_6")
        self.line_13 = QtWidgets.QFrame(self.page_3)
        self.line_13.setGeometry(QtCore.QRect(10, 270, 201, 20))
        self.line_13.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_13.setObjectName("line_13")
        self.comboBox_7 = QtWidgets.QComboBox(self.page_3)
        self.comboBox_7.setEnabled(True)
        self.comboBox_7.setGeometry(QtCore.QRect(155, 25, 51, 21))
        self.comboBox_7.setToolTip("")
        self.comboBox_7.setObjectName("comboBox_7")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_8 = QtWidgets.QComboBox(self.page_3)
        self.comboBox_8.setEnabled(True)
        self.comboBox_8.setGeometry(QtCore.QRect(155, 50, 51, 21))
        self.comboBox_8.setToolTip("")
        self.comboBox_8.setObjectName("comboBox_8")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_8.addItem("")
        self.comboBox_9 = QtWidgets.QComboBox(self.page_3)
        self.comboBox_9.setEnabled(True)
        self.comboBox_9.setGeometry(QtCore.QRect(90, 75, 116, 21))
        self.comboBox_9.setToolTip("")
        self.comboBox_9.setObjectName("comboBox_9")
        self.comboBox_9.addItem("")
        self.comboBox_9.addItem("")
        self.comboBox_9.addItem("")
        self.comboBox_9.addItem("")
        self.comboBox_9.addItem("")
        self.label_3 = QtWidgets.QLabel(self.page_3)
        self.label_3.setGeometry(QtCore.QRect(10, 25, 126, 21))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.page_3)
        self.label_4.setGeometry(QtCore.QRect(10, 50, 126, 21))
        self.label_4.setObjectName("label_4")
        self.label_13 = QtWidgets.QLabel(self.page_3)
        self.label_13.setGeometry(QtCore.QRect(10, 75, 71, 21))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.page_3)
        self.label_14.setGeometry(QtCore.QRect(10, 100, 136, 21))
        self.label_14.setObjectName("label_14")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_7.setEnabled(True)
        self.lineEdit_7.setGeometry(QtCore.QRect(145, 100, 61, 21))
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.lineEdit_7.setValidator(self.onlyInt)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_8.setEnabled(True)
        self.lineEdit_8.setGeometry(QtCore.QRect(145, 120, 61, 21))
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.lineEdit_8.setValidator(self.onlyFloat)
        self.label_15 = QtWidgets.QLabel(self.page_3)
        self.label_15.setGeometry(QtCore.QRect(10, 140, 86, 21))
        self.label_15.setObjectName("label_15")
        self.lineEdit_9 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_9.setEnabled(True)
        self.lineEdit_9.setGeometry(QtCore.QRect(145, 140, 61, 21))
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.label_16 = QtWidgets.QLabel(self.page_3)
        self.label_16.setGeometry(QtCore.QRect(10, 120, 86, 21))
        self.label_16.setObjectName("label_16")
        self.lineEdit_10 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_10.setEnabled(True)
        self.lineEdit_10.setGeometry(QtCore.QRect(145, 160, 61, 21))
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.lineEdit_10.setValidator(self.onlyFloat)
        self.label_17 = QtWidgets.QLabel(self.page_3)
        self.label_17.setGeometry(QtCore.QRect(10, 160, 111, 21))
        self.label_17.setObjectName("label_17")
        self.lineEdit_11 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_11.setEnabled(True)
        self.lineEdit_11.setGeometry(QtCore.QRect(145, 180, 61, 21))
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.lineEdit_11.setValidator(self.onlyFloat)
        self.label_18 = QtWidgets.QLabel(self.page_3)
        self.label_18.setGeometry(QtCore.QRect(10, 180, 86, 21))
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.page_3)
        self.label_19.setGeometry(QtCore.QRect(10, 280, 181, 21))
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.checkBox_13 = QtWidgets.QCheckBox(self.page_3)
        self.checkBox_13.setEnabled(False)
        self.checkBox_13.setGeometry(QtCore.QRect(10, 325, 156, 22))
        self.checkBox_13.setObjectName("checkBox_13")
        self.checkBox_13.setChecked(True)
        self.label_40 = QtWidgets.QLabel(self.page_3)
        self.label_40.setGeometry(QtCore.QRect(10, 200, 116, 21))
        self.label_40.setObjectName("label_40")
        self.lineEdit_17 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_17.setEnabled(True)
        self.lineEdit_17.setGeometry(QtCore.QRect(145, 200, 61, 21))
        self.lineEdit_17.setObjectName("lineEdit_17")
        self.lineEdit_17.setValidator(self.onlyInt)
        self.checkBox_11 = QtWidgets.QCheckBox(self.page_3)
        self.checkBox_11.setEnabled(False)
        self.checkBox_11.setGeometry(QtCore.QRect(10, 345, 156, 22))
        self.checkBox_11.setObjectName("checkBox_11")
        self.pushButton_12 = QtWidgets.QPushButton(self.page_3)
        self.pushButton_12.setEnabled(False)
        self.pushButton_12.setGeometry(QtCore.QRect(175, 300, 36, 26))
        self.pushButton_12.setObjectName("pushButton_12")
        self.checkBox_18 = QtWidgets.QCheckBox(self.page_3)
        self.checkBox_18.setGeometry(QtCore.QRect(10, 220, 196, 22))
        self.checkBox_18.setObjectName("checkBox_18")
        self.stackedWidget_3.addWidget(self.page_3)
        self.page_15 = QtWidgets.QWidget()
        self.page_15.setObjectName("page_15")
        self.label_12 = QtWidgets.QLabel(self.page_15)
        self.label_12.setGeometry(QtCore.QRect(10, -10, 156, 31))
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.line_11 = QtWidgets.QFrame(self.page_15)
        self.line_11.setGeometry(QtCore.QRect(5, 10, 201, 20))
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.radioButton_3 = QtWidgets.QRadioButton(self.page_15)
        self.radioButton_3.setGeometry(QtCore.QRect(10, 30, 101, 22))
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(self.page_15)
        self.radioButton_4.setGeometry(QtCore.QRect(10, 75, 136, 22))
        self.radioButton_4.setObjectName("radioButton_4")
        self.checkBox_7 = QtWidgets.QCheckBox(self.page_15)
        self.checkBox_7.setEnabled(False)
        self.checkBox_7.setGeometry(QtCore.QRect(55, 50, 121, 22))
        self.checkBox_7.setObjectName("checkBox_7")
        self.pushButton_14 = QtWidgets.QPushButton(self.page_15)
        self.pushButton_14.setGeometry(QtCore.QRect(135, 235, 75, 23))
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_15 = QtWidgets.QPushButton(self.page_15)
        self.pushButton_15.setGeometry(QtCore.QRect(55, 235, 75, 23))
        self.pushButton_15.setObjectName("pushButton_15")
        self.line_14 = QtWidgets.QFrame(self.page_15)
        self.line_14.setGeometry(QtCore.QRect(10, 220, 201, 20))
        self.line_14.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_14.setObjectName("line_14")
        self.stackedWidget_3.addWidget(self.page_15)
        self.progressBar_7 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_7.setGeometry(QtCore.QRect(800, 750, 201, 23))
        self.progressBar_7.setProperty("value", 100)
        self.progressBar_7.setObjectName("progressBar_7")
        self.pushButton_19 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_19.setGeometry(QtCore.QRect(955, 182, 41, 23))
        self.pushButton_19.setObjectName("pushButton_19")
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(244, 130, 130))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        self.pushButton_19.setPalette(palette)
        self.checkBox_9 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_9.setEnabled(False)
        self.checkBox_9.setGeometry(QtCore.QRect(805, 725, 176, 21))
        self.checkBox_9.setObjectName("checkBox_9")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setEnabled(False)
        self.lineEdit_3.setGeometry(QtCore.QRect(895, 80, 106, 21))
        self.lineEdit_3.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_3.setValidator(self.onlyInt)
        self.checkBox_12 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_12.setGeometry(QtCore.QRect(805, 80, 96, 21))
        self.checkBox_12.setObjectName("checkBox_12")
        self.label_37 = QtWidgets.QLabel(self.centralwidget)
        self.label_37.setGeometry(QtCore.QRect(805, 212, 41, 21))
        self.label_37.setFont(font)
        self.label_37.setObjectName("label_37")
        self.label_38 = QtWidgets.QLabel(self.centralwidget)
        self.label_38.setGeometry(QtCore.QRect(805, 162, 151, 21))
        self.label_38.setFont(font)
        self.label_38.setObjectName("label_38")
        self.label_39 = QtWidgets.QLabel(self.centralwidget)
        self.label_39.setGeometry(QtCore.QRect(805, 100, 156, 21))
        self.label_39.setFont(font)
        self.label_39.setObjectName("label_39")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(805, 0, 196, 76))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("Wiggle.PNG"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.checkBox_14 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_14.setEnabled(True)
        self.checkBox_14.setGeometry(QtCore.QRect(820, 146, 131, 16))
        self.checkBox_14.setObjectName("checkBox_14")


        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setEnabled(False)
        self.spinBox.setGeometry(QtCore.QRect(935, 640, 61, 21))
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(25)
        self.spinBox.setProperty("value", 5)
        self.spinBox.setObjectName("spinBox")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setEnabled(False)
        self.label_22.setGeometry(QtCore.QRect(805, 645, 86, 16))
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(805, 625, 181, 21))
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.checkBox_15 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_15.setEnabled(False)
        self.checkBox_15.setGeometry(QtCore.QRect(805, 705, 116, 21))
        self.checkBox_15.setObjectName("checkBox_15")
        self.spinBox_5 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_5.setEnabled(False)
        self.spinBox_5.setGeometry(QtCore.QRect(935, 705, 61, 21))
        self.spinBox_5.setMinimum(0)
        self.spinBox_5.setMaximum(255)
        self.spinBox_5.setSingleStep(10)
        self.spinBox_5.setProperty("value", 100)
        self.spinBox_5.setObjectName("spinBox_5")
        self.checkBox_16 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_16.setEnabled(False)
        self.checkBox_16.setGeometry(QtCore.QRect(805, 685, 176, 21))
        self.checkBox_16.setObjectName("checkBox_16")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        self.label_24.setEnabled(False)
        self.label_24.setGeometry(QtCore.QRect(805, 665, 71, 16))
        self.label_24.setObjectName("label_24")
        self.spinBox_6 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_6.setEnabled(False)
        self.spinBox_6.setGeometry(QtCore.QRect(885, 663, 36, 21))
        self.spinBox_6.setAcceptDrops(False)
        self.spinBox_6.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_6.setMinimum(0)
        self.spinBox_6.setMaximum(255)
        self.spinBox_6.setSingleStep(1)
        self.spinBox_6.setProperty("value", 65)
        self.spinBox_6.setObjectName("spinBox_6")
        self.spinBox_7 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_7.setEnabled(False)
        self.spinBox_7.setGeometry(QtCore.QRect(925, 663, 36, 21))
        self.spinBox_7.setAcceptDrops(False)
        self.spinBox_7.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_7.setMinimum(0)
        self.spinBox_7.setMaximum(255)
        self.spinBox_7.setSingleStep(1)
        self.spinBox_7.setProperty("value", 105)
        self.spinBox_7.setObjectName("spinBox_7")
        self.spinBox_8 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_8.setEnabled(False)
        self.spinBox_8.setGeometry(QtCore.QRect(964, 663, 36, 21))
        self.spinBox_8.setAcceptDrops(False)
        self.spinBox_8.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_8.setMinimum(0)
        self.spinBox_8.setMaximum(255)
        self.spinBox_8.setSingleStep(1)
        self.spinBox_8.setProperty("value", 225)
        self.spinBox_8.setObjectName("spinBox_8")
        self.checkBox_17 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_17.setEnabled(True)
        self.checkBox_17.setGeometry(QtCore.QRect(905, 725, 96, 21))
        self.checkBox_17.setObjectName("checkBox_17")

        ########################################################################################################################
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setStyleSheet("")
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        ########################################################################################################################
        self.retranslateUi(MainWindow)
        self.w = InteractiveWindow(self.centralwidget, None, None, self.wiggle)

        '''
        Visibility
        '''
        self.checkBox_5.setEnabled(False)
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_17.setEnabled(False)
        self.pushButton_18.setEnabled(False)

        '''
        Connections, signals and actions
        '''
        self.checkBox_17.toggled['bool'].connect(_night_mode)


        self.stackedWidget.setCurrentIndex(0)
        self.stackedWidget_3.setCurrentIndex(5)
        self.comboBox_9.setCurrentIndex(2)
        self.comboBox_7.setCurrentIndex(2)
        self.comboBox_8.setCurrentIndex(3)
        self.checkBox_14.stateChanged.connect(self.input_type)
        self.checkBox_15.toggled['bool'].connect(self.spinBox_5.setEnabled)


        self.comboBox.currentIndexChanged['int'].connect(self.stackedWidget.setCurrentIndex)
        self.comboBox.currentIndexChanged['int'].connect(lambda: self.checkBox_14.setChecked(False))

        self.comboBox_2.activated['int'].connect(self.stackedWidget_3.setCurrentIndex)


        self.checkBox.toggled['bool'].connect(self.pushButton_27.setDisabled)
        self.radioButton_4.toggled['bool'].connect(self.checkBox_7.setDisabled)
        self.radioButton_4.clicked['bool'].connect(self.pushButton_9.setEnabled)
        self.radioButton_4.clicked['bool'].connect(self.pushButton_5.setEnabled)

        self.radioButton_3.clicked['bool'].connect(self.pushButton_5.setEnabled)
        self.radioButton_3.clicked['bool'].connect(self.pushButton_9.setEnabled)
        self.radioButton_3.toggled['bool'].connect(self.checkBox_7.setEnabled)

        self.checkBox_12.toggled['bool'].connect(self.lineEdit_3.setEnabled)
        self.checkBox_12.clicked['bool'].connect(self.lineEdit_3.setEnabled)

        self.checkBox_12.stateChanged.connect(self.reset)
        self.lineEdit_3.textChanged.connect(self.reset)
        self.pushButton_19.clicked.connect(self.reset)

        # CryoSPARC multiple inputs
        self.pushButton_5.clicked.connect(
            partial(self.browse_file, "Select consensus map", self.pushButton_5, "Volumes (*.mrc)"))
        self.pushButton_9.clicked.connect(
            partial(self.browse_file, "Select ALL component maps", self.pushButton_9, "Volumes (*.mrc)"))
        self.pushButton_13.clicked.connect(
            partial(self.browse_file, "Select particle.cs file containing metadata", self.pushButton_13, "Particles (*.cs)"))

        # CryoSPARC single input
        self.pushButton_16.clicked.connect(
            partial(self.browse_file, "Select bundled cryoSPARC file.", self.pushButton_16, "Bundle (*.npz)"))

        #CryoDRGN multiple inputs
        self.pushButton_6.clicked.connect(
            partial(self.browse_file, "Select config file (e.g. config.pkl)", self.pushButton_6, "Pickle (*.pkl)"))
        self.pushButton_7.clicked.connect(
            partial(self.browse_file, "Select network weights (e.g. weights.49.pkl)", self.pushButton_7, "Pickle (*.pkl)"))
        self.pushButton_8.clicked.connect(
            partial(self.browse_file, "Select latent space (e.g. z.49.pkl)", self.pushButton_8, "Pickle (*.pkl)"))

        # CryoSPARC single input
        self.pushButton_21.clicked.connect(
            partial(self.browse_file, "Select bundled cryoDRGN file.", self.pushButton_21, "Bundle (*.npz)"))

        self.pushButton_19.clicked.connect(self.launchEmbedding)
        self.pushButton_23.clicked.connect(self.launchKmeans)
        self.pushButton_10.clicked.connect(self.launchMEP)

        self.radioButton_2.toggled['bool'].connect(self.pushButton.setEnabled)
        self.radioButton_2.toggled['bool'].connect(self.pushButton_2.setEnabled)
        self.radioButton_2.toggled['bool'].connect(self.pushButton_17.setEnabled)
        self.radioButton_2.toggled['bool'].connect(self.pushButton_18.setEnabled)
        self.radioButton_2.toggled['bool'].connect(self.comboBox_10.setEnabled)

        self.checkBox_6.toggled['bool'].connect(self.checkBox_2.setEnabled)
        self.comboBox_6.currentIndexChanged.connect(self.change_current_MEP_view)

        #SLOTS TO MODIFY THE PLOT / INTERACTIVE WINDOW
        self.comboBox_5.currentIndexChanged['int'].connect(
            lambda: self.w.modify_state('refresh', None))

        self.checkBox_8.stateChanged.connect(
            lambda: self.w.modify_state('trajectories',
                                        self.checkBox_8.isChecked())
        )

        self.checkBox_5.stateChanged.connect(
            lambda: self.w.modify_state('colour',
                                        self.checkBox_5.isChecked())
        )

        self.checkBox_9.stateChanged.connect(
            lambda: self.w.modify_state('legend',
                                        self.checkBox_9.isChecked())
        )

        self.radioButton_3.toggled['bool'].connect(
            lambda: self.w.modify_state('lasso',
                                        self.radioButton_3.isChecked())
        )

        self.radioButton_2.toggled['bool'].connect(
            lambda: self.w.modify_state('anchor',
                                        self.radioButton_2.isChecked()))

        self.timer.timeout.connect(
            lambda: self.reportProgress('', '', False, True, False)
        )

        self.statusTimer.timeout.connect(
            lambda: self.status(False)
        )

        self.pushButton_2.clicked.connect(
            lambda : self.user_anchor_points(query=False, save=False, reset=False, clear=True)
        )

        self.pushButton.clicked.connect(
            lambda : self.user_anchor_points(query=True, save=False, reset=False, clear=False)
        )

        self.pushButton_18.clicked.connect(
            lambda: self.user_anchor_points(query=False, save=True, reset=False, clear=False)
        )

        self.pushButton_17.clicked.connect(
            lambda: self.user_anchor_points(query=False, save=False, reset=True, clear=False)
        )

        self.pushButton_15.clicked.connect(
                lambda : self.export_roi()
        )

        self.pushButton_15.clicked.connect(
                lambda : self.export_clusters()
        )

        self.checkBox.stateChanged.connect(
            lambda : self.w.modify_state('map_update', self.checkBox.isChecked())
        )

        self.checkBox_13.stateChanged.connect(
            lambda : self.w.modify_state('MEP', self.checkBox_13.isChecked())
        )

        self.spinBox_6.valueChanged.connect(
            lambda : self.w.modify_state('refresh', None)
        )

        self.spinBox_7.valueChanged.connect(
            lambda : self.w.modify_state('refresh', None)
        )

        self.spinBox_8.valueChanged.connect(
            lambda : self.w.modify_state('refresh', None)
        )

        self.spinBox_5.valueChanged.connect(
            lambda : self.w.modify_state('refresh', None)
        )

        self.checkBox_15.stateChanged.connect(
            lambda : self.w.modify_state('refresh', None)
        )

        self.checkBox_16.stateChanged.connect(
            lambda : self.w.modify_state('refresh', None)
        )

        self.spinBox.valueChanged.connect(
            lambda : self.w.modify_state('refresh', None)
        )

        self.pushButton_27.clicked.connect(self.w.volumiser)
        self.pushButton_20.clicked.connect(self.w.volumiser_by_traj)

        self.checkBox_3.stateChanged.connect(
            lambda : self.w.modify_state('replace_map', self.checkBox_3.isChecked())
        )

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # NEEDS ADDRESSING # CBJ - "sometimes users change their minds"
    def reset(self):
        '''
        ###########################################################################################################
        Disable GUI until user re-runs dimensionality reduction OR changes their minds
        ###########################################################################################################
        '''
        if self.comboBox_2.isEnabled():
            self.comboBox_2.setDisabled(True)
            self.comboBox_2.setCurrentIndex(0)
            self.stackedWidget_3.setCurrentIndex(0)

        if self.checkBox_5.isEnabled():
            self.checkBox_5.setDisabled(True)
            self.checkBox_5.setChecked(False)

        if self.previous_data is not None:
            if len(self.previous_data) == int(self.lineEdit_3.text()):
                self.subset_state = False
            else:
                self.subset_state = True

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "WIGGLE - 1.0.0 alpha"))
        self.textBrowser.setHtml(_translate("MainWindow",
                                            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            "</style></head><body style=\" font-family:\'Noto Sans\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:10pt; font-weight:600;\">Command line output will appear here</span></p></body></html>"))
        self.textBrowser_3.setHtml(_translate("MainWindow",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "</style></head><body style=\" font-family:\'Noto Sans\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:7pt; font-weight:600;\">Charles Bayly-Jones 2021</span></p>\n"
                                              "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:7pt; font-weight:600;\"><br /></p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:7pt; font-weight:600;\">wiggle.help@gmail.com</span></p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:7pt;\"> </span></p>\n"
                                              "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:7pt; font-weight:600;\">Monash University, Australia</span></p></body></html>"))
        self.pushButton_8.setText(_translate("MainWindow", "Load latent (Z) space (.pkl)"))
        self.pushButton_6.setText(_translate("MainWindow", "Load cryoDRGN config (.pkl)"))
        self.pushButton_7.setText(_translate("MainWindow", "Load network weights (.pkl)"))
        self.pushButton_5.setToolTip(
            _translate("MainWindow", "Output basis map file from cryoSPARC 3D variational analysis"))
        self.pushButton_5.setText(_translate("MainWindow", "Load consensus map (.mrc)"))
        self.pushButton_9.setToolTip(
            _translate("MainWindow", "Select ALL variational component maps from cryoSPARC 3D variational analysis"))
        self.pushButton_9.setText(_translate("MainWindow", "Load component maps (.mrc)"))
        self.pushButton_13.setToolTip(_translate("MainWindow",
                                                 "Particle metadata file (cryoSPARC format, .CS). Looks like \"cryosparc_P0_J0_particles.cs\""))
        self.pushButton_13.setText(_translate("MainWindow", "Load particle metadata (.cs)"))
        self.pushButton_3.setText(_translate("MainWindow", "Load canonical map"))
        self.pushButton_4.setText(_translate("MainWindow", "Load deformation field"))
        self.label_41.setText(_translate("MainWindow", "Currently under development. Not yet available."))
        self.pushButton_16.setToolTip(_translate("MainWindow", "Compiled (single-file) binary format of 3DVA analysis"))
        self.pushButton_16.setStatusTip(_translate("MainWindow", "To generate, run \'/wiggle/scripts/compile_3DVA_output.py\' or instead swap dialog back to import individual files"))
        self.pushButton_16.setText(_translate("MainWindow", "Load cryoSPARC 3DVA (.npy)"))
        self.pushButton_21.setToolTip(_translate("MainWindow", "Compiled (single-file) binary format of cryoDRGN analysis"))
        self.pushButton_21.setStatusTip(_translate("MainWindow", "To generate, run \'/wiggle/scripts/compile_cryoDRGN_output.py\' or instead swap dialog back to import individual files"))
        self.pushButton_21.setText(_translate("MainWindow", "Load cryoDRGN (.npy)"))
        self.comboBox.setToolTip(_translate("MainWindow", "Mode"))
        self.comboBox.setItemText(0, _translate("MainWindow", "cryoDRGN"))
        self.comboBox.setItemText(1, _translate("MainWindow", "cryoSPARC 3DVA"))
        self.comboBox.setItemText(2, _translate("MainWindow", "cryoSPARC 3Dflex"))
        self.comboBox_3.setToolTip(_translate("MainWindow", "Type of dimensionality reduction"))
        self.comboBox_3.setStatusTip(_translate("MainWindow", "To enable, first import a latent space."))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "UMAP (slow)"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "PCA (very fast)"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "tSNE (fast)"))
        self.comboBox_3.setItemText(3, _translate("MainWindow", "PHATE (very slow)"))
        self.comboBox_3.setItemText(4, _translate("MainWindow", "cVAE (fast)"))
        self.comboBox_2.setStatusTip(_translate("MainWindow", "To enable, first run dimensionality reduction."))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "Interactive"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "Clustering"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "Graph Traversal"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "MEP Analysis"))
        self.comboBox_2.setItemText(4, _translate("MainWindow", "Particle Export"))
        self.checkBox.setToolTip(_translate("MainWindow",
                                            "Automatically replace the displayed map in ChimeraX with the selected configurational state."))
        self.checkBox.setText(_translate("MainWindow", "Continuous update"))
        self.label_8.setText(_translate("MainWindow", "Interactive mode"))
        self.pushButton_27.setToolTip(_translate("MainWindow",
                                                 "Manually update the displayed map in ChimeraX with the configurational state currently selected."))
        self.pushButton_27.setText(_translate("MainWindow", "Manual"))
        self.checkBox_3.setToolTip(_translate("MainWindow",
                                              "Whether to maintain a single map object (overwrite previous) or create multiple map objects"))
        self.checkBox_3.setText(_translate("MainWindow", "Replace map each time?"))
        self.label_7.setText(_translate("MainWindow", "Cluster analysis"))
        self.label_11.setToolTip(
            _translate("MainWindow", "Determines the number of classes, groups or clusters generated by kmeans"))
        self.label_11.setText(
            _translate("MainWindow", "<html><head/><body><p align=\"right\">Number of clusters? :</p></body></html>"))
        self.pushButton_23.setToolTip(_translate("MainWindow", "Start analysis."))
        self.pushButton_23.setText(_translate("MainWindow", "Go!"))
        self.pushButton_24.setToolTip(_translate("MainWindow", "Clear previous results."))
        self.pushButton_24.setText(_translate("MainWindow", "Reset"))
        self.checkBox_5.setToolTip(_translate("MainWindow", "Colour scatterplot by k means cluster indicies"))
        self.checkBox_5.setText(_translate("MainWindow", "Colour points by clusters"))
        self.checkBox_6.setToolTip(
            _translate("MainWindow", "For each cluster center, generate a volume and display it in ChimeraX."))
        self.checkBox_6.setText(_translate("MainWindow", "Generate volumes at centers?"))
        self.comboBox_4.setToolTip(_translate("MainWindow", "To enable, first import a latent space."))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "KMeans (fast)"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "Affinity Propagation (slow)"))
        self.comboBox_4.setItemText(2, _translate("MainWindow", "MeanShift (slow)"))
        self.comboBox_4.setItemText(3, _translate("MainWindow", "Spectral Clustering (slow)"))
        self.comboBox_4.setItemText(4, _translate("MainWindow", "Ward (fast)"))
        self.comboBox_4.setItemText(5, _translate("MainWindow", "Agglomerative Clustering (fast)"))
        self.comboBox_4.setItemText(6, _translate("MainWindow", "DBSCAN (fast)"))
        self.comboBox_4.setItemText(7, _translate("MainWindow", "OPTICS (slow)"))
        self.comboBox_4.setItemText(8, _translate("MainWindow", "BIRCH (fast)"))
        self.comboBox_4.setItemText(9, _translate("MainWindow", "Gaussian Mixture (fast)"))
        self.comboBox_5.setToolTip(_translate("MainWindow", "To enable, first import a latent space."))
        self.comboBox_5.setItemText(0, _translate("MainWindow", "viridis (u)"))
        self.comboBox_5.setItemText(1, _translate("MainWindow", "plasma (u)"))
        self.comboBox_5.setItemText(2, _translate("MainWindow", "inferno (u)"))
        self.comboBox_5.setItemText(3, _translate("MainWindow", "magma (u)"))
        self.comboBox_5.setItemText(4, _translate("MainWindow", "cividis (u)"))
        self.comboBox_5.setItemText(5, _translate("MainWindow", "twilight (c)"))
        self.comboBox_5.setItemText(6, _translate("MainWindow", "hsv (c)"))
        self.comboBox_5.setItemText(7, _translate("MainWindow", "seismic (d)"))
        self.comboBox_5.setItemText(8, _translate("MainWindow", "coolwarm (d)"))
        self.comboBox_5.setItemText(9, _translate("MainWindow", "Spectral (d)"))
        self.comboBox_5.setItemText(10, _translate("MainWindow", "PiYG (d)"))
        self.comboBox_5.setItemText(11, _translate("MainWindow", "PRGn (d)"))
        self.comboBox_5.setItemText(12, _translate("MainWindow", "RdGy (d)"))
        self.comboBox_5.setItemText(13, _translate("MainWindow", "bwr (d)"))
        self.checkBox_2.setToolTip(_translate("MainWindow",
                                              "Colour reconstruction within ChimeraX based on cluster label. To enable, first run kmeans."))
        self.checkBox_2.setText(_translate("MainWindow", "Colour volume by cluster"))
        self.pushButton.setToolTip(_translate("MainWindow", "Start analysis."))
        self.pushButton.setText(_translate("MainWindow", "Done"))
        self.label_5.setText(_translate("MainWindow", "Path traversal"))
        self.pushButton_2.setToolTip(_translate("MainWindow", "Clear previous results."))
        self.pushButton_2.setText(_translate("MainWindow", "Clear"))
        self.label_6.setToolTip(_translate("MainWindow",
                                           "The number of currently selected anchors is displayed here. Reset will clear if you need to start again."))
        self.label_6.setText(_translate("MainWindow", "Current anchor count:"))
        self.label_10.setText(_translate("MainWindow", "OR"))
        self.radioButton.setToolTip(_translate("MainWindow",
                                               "Traverse the latent space along a given variable (0: horizontal, 1: vertical, 2: +ve diagonal, 3: -ve diagonal)"))
        self.radioButton.setText(_translate("MainWindow", "Traverse latent variable:"))
        self.radioButton_2.setToolTip(_translate("MainWindow",
                                                 "Select multiple anchor points in the latent space and then attempt to traverse the space crossing these anchors."))
        self.radioButton_2.setText(_translate("MainWindow", "Select anchor points:"))
        self.checkBox_8.setText(_translate("MainWindow", "Plot trajectories?"))
        self.checkBox_10.setToolTip(_translate("MainWindow",
                                               "Highlight the latent coordinate corresponding to currently displayed volume."))
        self.checkBox_10.setText(_translate("MainWindow", "Link volume to plot?"))
        self.lineEdit_12.setText(_translate("MainWindow", "-1"))
        self.label_20.setToolTip(
            _translate("MainWindow", "Determines the number of classes, groups or clusters generated by kmeans"))
        self.label_20.setText(
            _translate("MainWindow", "<html><head/><body><p align=\"right\">Number of steps? :</p></body></html>"))
        self.pushButton_17.setToolTip(_translate("MainWindow", "Delete all trajectories and reset."))
        self.pushButton_17.setText(_translate("MainWindow", "Reset"))
        self.pushButton_18.setToolTip(_translate("MainWindow", "Clear previous results."))
        self.pushButton_18.setText(_translate("MainWindow", "Save"))
        self.label_21.setText(_translate("MainWindow", "Select trajectory below"))
        self.comboBox_10.setToolTip(_translate("MainWindow", "User defined paths for volume generation."))
        self.pushButton_20.setToolTip(_translate("MainWindow", "Compute volumes."))
        self.pushButton_20.setText(_translate("MainWindow", "Go!"))
        self.checkBox_19.setToolTip(_translate("MainWindow",
                                               "If selected, wiggle will generate a morph map corresponding to the trajectory. If unselected, individual volumes along the path with be rendered."))
        self.checkBox_19.setText(_translate("MainWindow", "Morph map?"))
        self.checkBox_20.setToolTip(_translate("MainWindow", "Change the direction of the path to the inverse."))
        self.checkBox_20.setText(_translate("MainWindow", "Reverse?"))
        self.label_9.setText(_translate("MainWindow", "Minimum energy path analysis"))
        self.pushButton_10.setToolTip(_translate("MainWindow", "Start analysis."))
        self.pushButton_10.setText(_translate("MainWindow", "Go!"))
        self.pushButton_11.setToolTip(_translate("MainWindow", "Clear previous results."))
        self.pushButton_11.setText(_translate("MainWindow", "Reset"))
        self.comboBox_6.setToolTip(_translate("MainWindow", "Ranked paths: least resistance to most resistance"))
        self.comboBox_7.setItemText(0, _translate("MainWindow", "3"))
        self.comboBox_7.setItemText(1, _translate("MainWindow", "5"))
        self.comboBox_7.setItemText(2, _translate("MainWindow", "7"))
        self.comboBox_7.setItemText(3, _translate("MainWindow", "9"))
        self.comboBox_8.setItemText(0, _translate("MainWindow", "3"))
        self.comboBox_8.setItemText(1, _translate("MainWindow", "5"))
        self.comboBox_8.setItemText(2, _translate("MainWindow", "7"))
        self.comboBox_8.setItemText(3, _translate("MainWindow", "9"))
        self.comboBox_8.setItemText(4, _translate("MainWindow", "11"))
        self.comboBox_8.setItemText(5, _translate("MainWindow", "13"))
        self.comboBox_8.setItemText(6, _translate("MainWindow", "15"))
        self.comboBox_8.setItemText(7, _translate("MainWindow", "17"))
        self.comboBox_8.setItemText(8, _translate("MainWindow", "19"))
        self.comboBox_8.setItemText(9, _translate("MainWindow", "21"))
        self.comboBox_9.setItemText(0, _translate("MainWindow", "Coarse (96)"))
        self.comboBox_9.setItemText(1, _translate("MainWindow", "Medium (128)"))
        self.comboBox_9.setItemText(2, _translate("MainWindow", "Fine (156)"))
        self.comboBox_9.setItemText(3, _translate("MainWindow", "Ultrafine (196)"))
        self.comboBox_9.setItemText(4, _translate("MainWindow", "Extreme (256)"))
        self.label_3.setText(_translate("MainWindow", "Width of weiner filter"))
        self.label_4.setText(_translate("MainWindow", "Width of median filter"))
        self.label_13.setText(_translate("MainWindow", "Resolution"))
        self.label_14.setText(_translate("MainWindow", "No. searches to spawn:"))
        self.lineEdit_7.setText(_translate("MainWindow", "1200"))
        self.lineEdit_8.setText(_translate("MainWindow", "0.05"))
        self.label_15.setText(_translate("MainWindow", "Sampling"))
        self.lineEdit_9.setText(_translate("MainWindow", "40"))
        self.label_16.setText(_translate("MainWindow", "Learning rate"))
        self.lineEdit_10.setText(_translate("MainWindow", "0.2"))
        self.label_17.setText(_translate("MainWindow", "Coupling constant"))
        self.lineEdit_11.setText(_translate("MainWindow", "2.5"))
        self.label_18.setText(_translate("MainWindow", "String tension"))
        self.label_19.setText(_translate("MainWindow", "Select path below"))
        self.checkBox_13.setToolTip(_translate("MainWindow", "Show trajectory as lines on the embedding."))
        self.checkBox_13.setText(_translate("MainWindow", "Plot trajectory?"))
        self.label_40.setText(_translate("MainWindow", "Minima to consider"))
        self.lineEdit_17.setText(_translate("MainWindow", "10"))
        self.checkBox_11.setToolTip(_translate("MainWindow",
                                               "Output all the volumes as a single volume series? If false, output individual volumes."))
        self.checkBox_11.setText(_translate("MainWindow", "Output volume series?"))
        self.pushButton_12.setToolTip(_translate("MainWindow", "Compute volumes."))
        self.pushButton_12.setText(_translate("MainWindow", "Go!"))
        self.checkBox_18.setToolTip(_translate("MainWindow", "Show intermedate plots for troubleshooting."))
        self.checkBox_18.setText(_translate("MainWindow", "Generate diagnostic plots?"))
        self.label_12.setText(_translate("MainWindow", "Particle curation"))
        self.radioButton_3.setText(_translate("MainWindow", "Lasso tool"))
        self.radioButton_4.setText(_translate("MainWindow", "Export by cluster"))
        self.checkBox_7.setText(_translate("MainWindow", "Invert selection"))
        self.pushButton_14.setToolTip(_translate("MainWindow", "Clear previous results."))
        self.pushButton_14.setText(_translate("MainWindow", "Reset"))
        self.pushButton_15.setToolTip(_translate("MainWindow", "Start analysis."))
        self.pushButton_15.setText(_translate("MainWindow", "Go!"))
        self.pushButton_19.setText(_translate("MainWindow", "Go!"))
        self.checkBox_9.setToolTip(_translate("MainWindow", "Colour scatterplot by k means cluster indicies"))
        self.checkBox_9.setText(_translate("MainWindow", "Hide legend"))
        self.lineEdit_3.setToolTip(_translate("MainWindow", "Subsample space to this many observations."))
        self.checkBox_12.setToolTip(_translate("MainWindow", "Large spaces can be slow or cause lag."))
        self.checkBox_12.setText(_translate("MainWindow", "Subsample"))
        self.label_37.setText(_translate("MainWindow", "Mode"))
        self.label_38.setText(_translate("MainWindow", "Dimensional reduction"))
        self.label_39.setText(_translate("MainWindow", "Variation analysis input"))
        self.checkBox_14.setToolTip(
            _translate("MainWindow", "Swap input dialog to accept single-file format for cryoSPARC \n"
                                     " 3DVA output. The single-file format simplifies sharing and \n"
                                     " can be compiled with \'/wiggle/scripts/compile_3DVA_output.py/\'"))
        self.checkBox_14.setText(_translate("MainWindow", "Single-file format"))
        self.label_22.setText(_translate("MainWindow", "Point size"))
        self.label_23.setText(_translate("MainWindow", "Display"))
        self.checkBox_16.setText(_translate("MainWindow", "Show silhouette"))
        self.label_24.setText(_translate("MainWindow", "Colour (rgb)"))
        self.checkBox_15.setText(_translate("MainWindow", "Transparency"))
        self.checkBox_17.setText(_translate("MainWindow", "Night mode"))

class InteractiveWindow(object):
    def __init__(self, centralWidget, embedding, labels, session):
        self.isColour = False
        self.isCluster = False
        self.isSubSample = False
        self.isTrajPlot = False
        self.isMEPTrajPlot = False
        self.isAnchors = False
        self.current_anchors = []
        self.isROI = False
        self.spaceChanged = True
        self.autoUpdate = True
        self.legend = True
        self.curvePlotsUSER = []
        self.curvePlotsUSER_pts = []
        self.curvePlotsMEP = []
        self.legendItemsTraj = []
        self.TrajLists = []
        self.MEP_TrajLists = []
        self.MEP_legendItemsTraj = []
        self.config = ''
        self.weights = ''
        self.apix = 3.4
        self.labels = labels
        self.embedding = embedding
        self.data = ''
        self.currentMEPdisplay = []
        self.currentMEPlegenddisplay = []
        self.replace_map = False
        self.wiggle = session
        self.flip = True

        # Set up the pyqtgraph
        self.graphicsView = pg.GraphicsLayoutWidget(centralWidget)
        self.graphicsView.setGeometry(QtCore.QRect(0, 80, 796, 666))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView.setBackground(background='w')
        self.plotArea = self.graphicsView.addPlot()
        self.plotArea.addLegend()
        self.plotArea.hideAxis('bottom')
        self.plotArea.hideAxis('left')

    def initialise_volumiser(self):
        self.mode = ui.inputType[ui.comboBox.currentIndex()]
        if self.mode == 'cryodrgn':
            from .cryodrgn_minimal import miniDRGN
            self.miniDRGN = miniDRGN(self.data[0], self.data[1], self.apix, self.flip)
        elif self.mode == 'cryosparc_3dva':
            from .miniSPARC import miniSPARC
            self.miniSPARC = miniSPARC(self.data[0], self.data[1], self.apix, self.flip)

    def plot_clicked(self, plot, points, event):
        clickedPen = pg.mkPen((0, 0, 0), width=3)
        global lastClicked
        global lastPlot
        try:
            lastPlot
        except:
            pass
        else:
            if lastPlot is not plot:
                lastClicked = []
        finally:
            for pt in lastClicked:
                pt.resetPen()

        for pt in points:
            pt.setPen(clickedPen)
        indices = []
        for pt in points:
            pt = pt.pos()
            x, y = pt.x(), pt.y()
            lx = np.argwhere(self.embedding[:, 0] == x)
            ly = np.argwhere(self.embedding[:, 1] == y)
            #lx = np.argwhere(plot.data['x'] == x)
            #ly = np.argwhere(plot.data['y'] == y)
            i = np.intersect1d(lx, ly).tolist()
            indices += i
        indices = list(set(indices))  #Indicies are returned in order, undecided if this is necessarily OK...
        lastClicked = points
        lastPlot = plot
        # print('Indices: ' + ' '.join([str(elem) for elem in indices]))

        if self.isAnchors:
            if len(indices) > 1:
                index = [indices[0]]
            else:
                index = indices
            self.current_anchors.append(index)

        #Save the Z index for miniDRGN
        self.currentZind = indices[0]

        if self.autoUpdate:
            self.volumiser()

        ui.lineEdit_12.setText(str(len(self.current_anchors)))

    def modify_state(self, property, active: bool, val=''):
        '''
        ###########################################################################################################
        Modify the plot options according to UI parameters
        ###########################################################################################################
        '''

        # Whether to replace the current map with a new one.
        if property == 'replace_map':
            self.replace_map = active
            return

        # Whether to trajectories should be shown
        if property == 'trajectories':
            self.isTrajPlot = active
            self.plot_trajectories()
            return

        # Whether to trajectories should be shown
        if property == 'MEP':
            self.isMEPTrajPlot = active
            self.plot_MEPtrajectories()
            return

        # Whether to show ROI
        if property == 'lasso':
            self.isROI = active
            self.plot_rois()
            return

        # Whether to hide legend
        if property == 'legend':
            self.legend = not active
            self.plot_legend()
            return

        #Whether anchor points should be returned from the InteractiveWindow
        if property == 'anchor':
            self.isAnchors = active
            return

        # Controller frequency of map update
        if property == 'map_update':
            self.autoUpdate = active
            return

        #Whether to colour plot or not
        if property == 'colour':
            self.isColour = active

        # In case the embedding landscape is modified
        if property == 'space':
            self.spaceChanged = True

        if property == 'refresh':
            self.refresh_plot()
        else:
            self.plot_scatter()

    def map_to_colour(self, slice):
        '''
        ###########################################################################################################
        Returns a colour object that is indexed by a value between 0 and 1
            e.g. cm[0.1] corresponds to a particular colour object
        ###########################################################################################################
        '''
        userColourOptions = (
            'viridis',
            'plasma',
            'inferno',
            'magma',
            'cividis',
            'twilight',
            'hsv',
            'seismic_r',
            'coolwarm',
            'Spectral_r',
            'PiYG_r',
            'PRGn_r',
            'RdGy_r',
            'bwr_r'
        )
        cm = pg.colormap.get(userColourOptions[ui.comboBox_5.currentIndex()], source='matplotlib')
        vals = cm.getColors()
        vals[:, -1] = self.alpha
        _cm = pg.colormap.ColorMap(pos=np.linspace(0.0, 1.0, vals.shape[0]), color=vals)
        return _cm[slice]

    def plot_trajectories(self):
        '''
        ###########################################################################################################
        Plot (pseudo)trajectories if required
        ###########################################################################################################
        '''
        for curvePlot,curvePlot_pt in zip(self.curvePlotsUSER,self.curvePlotsUSER_pts):
            self.plotArea.removeItem(curvePlot)
            self.plotArea.removeItem(curvePlot_pt)

        if self.isTrajPlot:
            self.curvePlotsUSER = []
            self.curvePlotsUSER_pts = []
            # Define trajectories by index
            traj = pg.mkPen((181, 34, 34), width=3, style=QtCore.Qt.DotLine)
            shadowPen = None  # or pg.mkPen((255, 255, 255), width=3)

            # Iterate over trajectories and plot these.
            for n, index in enumerate(self.TrajLists):
                curvePlot = pg.PlotCurveItem(
                    pen=traj,
                    shadowPen=shadowPen
                )
                curvePlot_pts = pg.ScatterPlotItem(
                    # pen=pg.mkPen((0, 0, 0), width=1),
                    pen=pg.mkPen(None),
                    size=ui.spinBox.value(),
                    downsample=True,
                    downsampleMethod='mean',
                    hoverable=True,
                    hoverSymbol='o',
                    hoverSize=30,
                    hoverBrush=pg.mkBrush(166, 229, 181, 150)
                )

                curvePlot.setData(x=self.embedding[index][:, 0], y=self.embedding[index][:, 1])
                curvePlot_pts.setData(x=self.embedding[index][:, 0], y=self.embedding[index][:, 1])
                curvePlot_pts.sigHovered.connect(self.point_hover_slider)
                self.curvePlotsUSER.append(curvePlot)
                self.curvePlotsUSER_pts.append(curvePlot_pts)
                self.plotArea.addItem(curvePlot)
                self.plotArea.addItem(curvePlot_pts)

            if self.legend:
                self.plot_legend()

    def refresh_trajectories(self, morph_slider):
        # morph_slider = self.slider_list[ui.comboBox_10.currentIndex()]
        slider_ind = floor(morph_slider.slider.value()/10)
        for ind, slider in enumerate(self.slider_list):
            if slider == morph_slider:
                _list = self.TrajLists[ind]
        pt_highlight = _list[slider_ind]
        spots = []
        for ind in _list:
            x = self.embedding[ind][0]
            y = self.embedding[ind][1]
            if ind == pt_highlight:
                s = {'pos': (x, y), 'size': 30, 'pen': None, 'brush': pg.mkBrush(166, 229, 181, 150), 'symbol': 'o'}
            else:
                s = {'pos': (x, y), 'size': ui.spinBox.value(), 'pen': None, 'symbol': 'o'}
            spots.append(s)

        for plot in self.curvePlotsUSER_pts:
            plot.setData(spots=spots)

    def point_hover_slider(self, plot, points):
        if ui.checkBox_10.isChecked():
            # _list = self.TrajLists[ui.comboBox_10.currentIndex()]
            for ind, curve in enumerate(self.curvePlotsUSER_pts):
                if plot == curve:
                    _list = self.TrajLists[ind]
                    for pt in points:
                        pt = pt.pos()
                        x, y = pt.x(), pt.y()
                        lx = np.argwhere(self.embedding[:, 0] == x)
                        ly = np.argwhere(self.embedding[:, 1] == y)

                        i = np.intersect1d(lx, ly).tolist()[0]
                        if i in _list:
                            morph_slider = self.slider_list[ind]
                            # slider.setValue(_list.index(i)*10)
                            morph_slider.set_slider(_list.index(i)*10)

    def plot_MEPtrajectories(self):
        '''
        ###########################################################################################################
        Plot (pseudo)trajectories if required
        ###########################################################################################################
        '''
        for curvePlot in self.curvePlotsMEP:
            self.plotArea.removeItem(curvePlot)

        if self.isMEPTrajPlot:
            self.curvePlotsMEP = []
            # Define trajectories by index
            traj = pg.mkPen((168, 109, 0), width=3, style=QtCore.Qt.DotLine)
            shadowPen = None  # or pg.mkPen((255, 255, 255), width=3)
            # Iterate over trajectories and plot these.
            for n, index in enumerate(self.currentMEPdisplay):
                curvePlot = pg.PlotCurveItem(
                    pen=traj,
                    shadowPen=shadowPen
                )

                curvePlot.setData(x=self.embedding[index][:, 0], y=self.embedding[index][:, 1])
                self.curvePlotsMEP.append(curvePlot)
                self.plotArea.addItem(curvePlot)

            if self.legend:
                self.plot_legend()

    def plot_rois(self):
        '''
        ###########################################################################################################
        Plot regions of interest (ROIs) if applicable
        ###########################################################################################################
        '''
        if self.spaceChanged:
            try:
                self.plotArea.removeItem(self.ROIs)
            except:
                pass

            axX = self.plotArea.getAxis('bottom')
            x_min, x_max = axX.range  # <------- get range of x axis
            axY = self.plotArea.getAxis('left')
            y_min, y_max = axY.range  # <------- get range of y axis

            com_x, com_y = (x_max + x_min) / 2, (y_max + y_min) / 2

            roiPen = pg.mkPen((22, 208, 115), width=3, style=QtCore.Qt.SolidLine)
            HoverPen = pg.mkPen((255, 255, 0), width=3, style=QtCore.Qt.DashLine)

            self.ROIs = pg.PolyLineROI(
                [[0.2 * x_min + com_x, com_y],
                 [com_x, 0.2 * y_max + com_y],
                 [0.2 * x_max + com_x, com_y],
                 [com_x, 0.2 * y_min + com_y]],
                closed=True, pen=roiPen, handlePen=(153, 51, 255), hoverPen=HoverPen)

            self.spaceChanged = False

        if self.isROI:
            self.plotArea.addItem(self.ROIs)
        else:
            self.plotArea.removeItem(self.ROIs)

    def plot_legend(self):
        if self.legend:
            self.plotArea.legend.clear()
            for legend, plot in zip(self.legendItems, self.scatterPlots):
                self.plotArea.legend.addItem(plot, legend)
            if self.isTrajPlot:
                for legend, plot in zip(self.legendItemsTraj, self.curvePlotsUSER):
                    self.plotArea.legend.addItem(plot, legend)
            if self.isMEPTrajPlot:
                for legend, plot in zip(self.currentMEPlegenddisplay, self.curvePlotsMEP):
                    self.plotArea.legend.addItem(plot, legend)
        else:
            self.plotArea.legend.clear()

    def refresh_plot(self):
        if ui.checkBox_15.isChecked():
            self.alpha = ui.spinBox_5.value()
        else:
            self.alpha = 255

        if ui.checkBox_16.isChecked():
            pen_type = pg.mkPen((0, 0, 0), width=1)
        else:
            pen_type = None

        if self.isColour:
            try:
                self.labels
            except:
                print("BUG: Cluster labels were not found, but colour was somehow triggered.")

            #Find non-redundant list of labels
            unique = set(self.labels)
            unique_scaled = [item / (len(unique) - 1) for item in unique]
            self.colour_set = list(map(self.map_to_colour, unique_scaled))
            self.legendItems = ['obs; clstr_' + str(label) for label in set(self.labels)]

        else:
            self.colour_set = [pg.mkBrush(ui.spinBox_6.value(),
                                          ui.spinBox_7.value(),
                                          ui.spinBox_8.value(),
                                          self.alpha)]
            self.legendItems = ['obs']


        for l, plot in enumerate(self.scatterPlots):
            plot.setPen(pen_type)
            plot.setSize(ui.spinBox.value())
            plot.setBrush(self.colour_set[l])

    def plot_scatter(self):
        '''
        ###########################################################################################################
        Plot 2D embedding of latent space as scatter plot
        Each point is an observation of the state represented by the position in the embedding
        ###########################################################################################################
        '''
        if ui.checkBox_15.isChecked():
            self.alpha = ui.spinBox_5.value()
        else:
            self.alpha = 255

        if ui.checkBox_16.isChecked():
            pen_type = pg.mkPen((0, 0, 0), width=1)
        else:
            pen_type = None


        if self.isColour:
            try:
                self.labels
            except:
                print("BUG: Cluster labels were not found, but colour was somehow triggered.")

            #Find non-redundant list of labels
            unique = set(self.labels)
            unique_scaled = [item / (len(unique) - 1) for item in unique]
            self.colour_set = list(map(self.map_to_colour, unique_scaled))
            self.legendItems = ['obs; clstr_' + str(label) for label in set(self.labels)]

            cluster_list = []
            for label in set(self.labels):
                cluster = [pt for subPlot in self.scatterPlots for pt in subPlot.points() if
                           self.labels[pt.data()] == label]
                cluster_list.append(cluster)
            self.hard_reset = True
        else:
            cluster_list = ['']
            self.colour_set = [pg.mkBrush(ui.spinBox_6.value(),
                                          ui.spinBox_7.value(),
                                          ui.spinBox_8.value(),
                                          self.alpha)]
            self.legendItems = ['obs']

        ###########################################################################################################
        # Plot latent space after embedding.

        try:
            self.scatterPlots
        except:
            pass
        else:
            for subPlot in self.scatterPlots:
                self.plotArea.removeItem(subPlot)

        self.scatterPlots = []
        for cluster_n, cluster in enumerate(cluster_list):
            interactPlot = pg.ScatterPlotItem(
                pen=pen_type,
                size=ui.spinBox.value(),
                # symbol='o',
                # symbolPen=None,  # pg.mkPen((0, 0, 0), width=3),
                # symbolSize=10,
                # symbolBrush=colour_style,
                downsample=True,
                downsampleMethod='mean',
                hoverable=True,
                hoverSymbol='s',
                hoverSize=12,
                hoverPen=pg.mkPen((0, 0, 0), width=2),
                hoverBrush=pg.mkBrush(250, 128, 114)
                # brush=self.colour_set[cluster_n]
            )

            if self.isColour:
                interactPlot.setData(
                    pos=[pt.pos() for pt in cluster],
                    data=[pt.index() for pt in cluster],
                    brush=self.colour_set[cluster_n])
            else:
                interactPlot.setData(
                    pos=self.embedding,
                    data=range(0, len(self.embedding)))

            interactPlot.sigClicked.connect(self.plot_clicked)
            self.scatterPlots.append(interactPlot)

        for l,plot in enumerate(self.scatterPlots):
            plot.setZValue(-l)
            self.plotArea.addItem(plot)

        self.refresh_plot()
        if self.legend:
            self.plot_legend()

        ui.checkBox_9.setEnabled(True)
        ui.checkBox_15.setEnabled(True)
        ui.checkBox_16.setEnabled(True)
        ui.label_24.setEnabled(True)
        ui.label_22.setEnabled(True)
        ui.spinBox.setEnabled(True)
        ui.spinBox_6.setEnabled(True)
        ui.spinBox_7.setEnabled(True)
        ui.spinBox_8.setEnabled(True)

    def heat_map(self):
        x_min, y_min = np.amin(self.embedding, axis=0) #self.plotArea.getAxis('bottom')
        #x_min, y_min = axX.range  # <------- get range of x axis
        x_max, y_max = np.amax(self.embedding, axis=0) #self.plotArea.getAxis('left')
        #x_max, y_max = axY.range  # <------- get range of y axis


        pltRange = (x_max - x_min), (y_max - y_min)
        pixel = int(pltRange[0]) / 100
        x_range = int(pltRange[0] / pixel)
        y_range = int(pltRange[1] / pixel)

        x_r = np.arange(x_min, x_min + pltRange[0] - pixel, pixel)
        x = np.repeat(x_r, y_range)
        x = x.reshape(x_range, y_range)

        y_r = np.arange(y_min, y_min + pltRange[1] - pixel, pixel)
        y = np.tile(y_r, x_range)
        y = y.reshape(x_range, y_range)

        histo, _, _ = np.histogram2d(self.embedding[:, 0], self.embedding[:, 1], bins=[x_range, y_range])

        densityColourMesh = pg.PColorMeshItem(edgecolors=None, antialiasing=False, cmap='grey')
        densityColourMesh.setZValue(-1)
        self.plotArea.addItem(densityColourMesh)

        densityColourMesh.setData(x, y, histo[:-1, :-1])

    def volumiser(self):
        try:
            self._last_vol
        except:
            self._last_vol = None

        from chimerax.map import volume_from_grid_data
        from chimerax.map_data import ArrayGridData

        if self.mode == 'cryodrgn':
            grid = ArrayGridData(self.miniDRGN.generate(self.data[2][self.currentZind]), name='cryodrgn_vol_i')
        elif self.mode == 'cryosparc_3dva':
            grid = ArrayGridData(self.miniSPARC.generate(self.data[2][self.currentZind]), name='cryosparc_vol_i')

        vol = volume_from_grid_data(grid, self.wiggle, open_model=True, show_dialog=False)
        vol.set_display_style('surface')

        if self.replace_map and not self._last_vol.deleted:
            self._last_vol.delete()

        self._last_vol = vol

    def volumiser_by_cluster(self):
        pass

    def volumiser_by_traj(self):
        try:
            self.slider_list
        except:
            self.slider_list = []

        from chimerax.map import volume_from_grid_data
        from chimerax.map_data import ArrayGridData
        from chimerax.map_filter.morph import morph_maps
        from chimerax.map_filter.morph_gui import MorphMapSlider

        Z_trajectory = self.TrajLists[ui.comboBox_10.currentIndex()]
        volumes = []
        for coord in Z_trajectory:
            if self.mode == 'cryodrgn':
                grid = ArrayGridData(self.miniDRGN.generate(self.data[2][coord]), name='cryodrgn_vol_i')
            elif self.mode == 'cryosparc_3dva':
                grid = ArrayGridData(self.miniSPARC.generate(self.data[2][coord]), name='cryosparc_vol_i')
            vol = volume_from_grid_data(grid, self.wiggle, open_model=True, show_dialog=False)
            volumes.append(vol)

        frames = len(volumes) * 10 - 1
        step = 1 / frames
        volTraj = morph_maps(volumes, frames, 0, step, 1, (0.0, 1.0), False, False, None, True, False, 'all', 1, None)

        morph_slider = MorphMapSlider(self.wiggle, volTraj)
        morph_slider.slider.valueChanged.connect(lambda : self.refresh_trajectories(morph_slider))
        self.slider_list.append(morph_slider)

class Wiggle(object):
    def __init__(self, session):
        self.wiggle = session
        super().__init__()

    def chimera_initialise(self):
        layout = QtWidgets.QGridLayout()
        MainWindow = QtWidgets.QMainWindow()
        layout.addWidget(MainWindow)
        layout.setContentsMargins(0,0,0,0)

        #Initialise the Ui_MainWindow class
        global ui
        ui = Ui_MainWindow(self.wiggle)

        #Run UI setup method of Ui_MainWindow class
        ui.setupUi(MainWindow)
        return layout