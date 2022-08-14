## Charles Bayly-Jones 2020 ##

## Import dependencies ##
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
from functools import partial
import os.path
import time
from cryodrgn_minimal import miniDRGN

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
lastClicked = []

class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(float, float, bool, bool, bool)
    msg = QtCore.pyqtSignal(str)
    status = QtCore.pyqtSignal(bool)

    def load_data(self, path, do_subset: bool, fraction=-1):
        if os.path.isfile(path):
            d = np.load(path, allow_pickle=True)
            if do_subset:
                idx = np.random.randint(d.shape[0], size=fraction)
                return d[idx]
            else:
                return d
        else:
            return np.random.randint(50, size=(64, 64))

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
        global labels

        from sklearn import cluster, mixture
        from sklearn.neighbors import kneighbors_graph
        self.msg.emit('Begin clustering ...')


        clusters = ui.spinBox_2.value()

        #Subset for ETA calculation - at the expense of a small amount of time
        fraction = 0.01
        subset = int(len(data) * fraction)
        idx = np.random.randint(data.shape[0], size=subset)
        small_data = data[idx]


        # ============
        # Create cluster objects from sklearn clustering example
        # ============

        # connectivity matrix for structured Ward
        if ui.comboBox_4.currentIndex() == 4 or ui.comboBox_4.currentIndex() == 5:
            connectivity = kneighbors_graph(data, n_neighbors=5, include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)
        else:
            connectivity = []

        # estimate bandwidth for mean shift
        if ui.comboBox_4.currentIndex() == 2:
            bandwidth = cluster.estimate_bandwidth(data, quantile=0.3)
        else:
            bandwidth = []


        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

        two_means = cluster.MiniBatchKMeans(n_clusters=clusters)

        ward = cluster.AgglomerativeClustering(n_clusters=clusters, linkage='ward',
            connectivity=connectivity)

        spectral = cluster.SpectralClustering(n_clusters=clusters, eigen_solver='arpack',
            affinity="nearest_neighbors")

        dbscan = cluster.DBSCAN(eps=0.5)

        optics = cluster.OPTICS(min_samples=20, xi=0.05,
                                min_cluster_size=0.1)

        affinity_propagation = cluster.AffinityPropagation(damping=0.6,
                                                           preference=200)

        average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",
                                                          n_clusters=clusters, connectivity=connectivity)

        birch = cluster.Birch(n_clusters=clusters)

        gmm = mixture.GaussianMixture(n_components=clusters, covariance_type='full')

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

        name, operator = clustering_algorithms[ui.comboBox_4.currentIndex()]

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
        operator.fit(data)

        #Generate labels list
        if hasattr(operator, 'labels_'):
            labels = operator.labels_.astype(int)
        else:
            labels = operator.predict(data)

        #Finished close thread and emit complete signal
        self.msg.emit('Finished ' + name + ' ! \n -------------------------')
        self.progress.emit(1, 1, False, False, True)
        self.finished.emit()
        self.status.emit(True)

    def run_embedding(self): ###Re implement as above, too much redundancy.
        self.status.emit(False)
        technique = ui.comboBox_3.currentIndex()
        '''
            Perform a range of dimensionality reduction analyses.
        '''
        global embedding
        global data

        if ui.checkBox_12.checkState() == QtCore.Qt.Checked and ui.lineEdit_3.text().isnumeric():
            data = self.load_data(ui.lineEdit_4.text(), do_subset=True, fraction=int(ui.lineEdit_3.text()))
            self.msg.emit("Subsampling latent space to %s \n -------------------------" % int(ui.lineEdit_3.text()))
        else:
            data = self.load_data(ui.lineEdit_4.text(), do_subset=False)

        fraction = 0.05
        subset = int(len(data) * fraction)
        idx = np.random.randint(data.shape[0], size=subset)
        small_data = data[idx]

        if technique == 0:
            self.msg.emit("Running umap... ")
            import umap
            operator = umap.UMAP(random_state=42, verbose=1)

            ### TIMING TEST
            t0 = time.time()
            dummy = operator.fit_transform(small_data)
            ETA = 50 * (time.time() - t0)
            self.progress.emit(ETA, t0, True, True, False)
            ### END TIMING TES

            embedding = operator.fit_transform(data)
            self.msg.emit("Finished umap! \n -------------------------")

        if technique == 1:
            self.msg.emit("Running PCA... ")
            from sklearn.decomposition import PCA
            operator = PCA(n_components=2)

            ### TIMING TEST
            t0 = time.time()
            operator.fit(small_data)
            ETA = (1 / fraction) * (time.time() - t0)
            self.progress.emit(ETA, t0, True, True, False)
            ### END TIMING TEST

            operator.fit(data)
            embedding = operator.transform(data)
            self.msg.emit("Finished PCA! \n -------------------------")

        if technique == 2:
            self.msg.emit("Running tSNE...")
            from sklearn.manifold import TSNE
            operator = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1)

            ### TIMING TEST
            t0 = time.time()
            dummy = operator.fit_transform(small_data)
            ETA = 1.5 * (1/fraction)*(time.time() - t0)
            self.progress.emit(ETA, t0, True, True, False)
            ### END TIMING TEST

            embedding = operator.fit_transform(data)
            self.msg.emit("Finished tSNE! \n -------------------------")

        if technique == 3:
            self.msg.emit("Running PHATE... ")
            QtWidgets.qApp.processEvents()
            import phate
            operator = phate.PHATE(n_jobs=-2)

            ### TIMING TEST
            t0 = time.time()
            dummy = operator.fit_transform(small_data)
            ETA = 5 * (time.time() - t0)
            self.progress.emit(ETA, t0, True, True, False)
            ### END TIMING TEST

            embedding = operator.fit_transform(data)
            self.msg.emit("Finished PHATE! \n -------------------------")

        if technique == 4:
            self.msg.emit("Running cVAE... ")
            QtWidgets.qApp.processEvents()
            from cvae import cvae

            ### TIMING TEST
            t0 = time.time()
            operator = cvae.CompressionVAE(small_data)
            operator.train()
            dummy = operator.embed(small_data)
            ETA = 5 * (time.time() - t0)
            self.progress.emit(ETA, t0, True, True, False)
            ### END TIMING TEST

            operator = cvae.CompressionVAE(data)
            operator.train()
            embedding = operator.embed(data)
            self.msg.emit("Finished cVAE! \n -------------------------")

        self.progress.emit(ETA, t0, False, False, True)
        self.finished.emit()
        self.status.emit(True)

class Main(object):
    # def __init__(self, ui):
    #     global self.ui
    #     self.ui = ui

    def browse_dir(self, comment, button):
        dirName = str(QtWidgets.QFileDialog.getExistingDirectory(None, comment, ""))
        if dirName:
            self.string = dirName + '/'
            if button == self.pushButton_3:
                self.lineEdit.setText(self.string)

    def browse_file(self, comment, button, type):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, comment, "", type, options=options)

        if fileName:
            if button == self.pushButton_3:
                self.lineEdit.setText(fileName)
            elif button == self.pushButton_4:
                self.lineEdit_2.setText(fileName)
            elif button == self.pushButton_6:
                self.lineEdit_6.setText(fileName)
            elif button == self.pushButton_7:
                self.lineEdit_5.setText(fileName)
            elif button == self.pushButton_8:
                self.lineEdit_4.setText(fileName)
                self.comboBox_3.setEnabled(True)

    def status(self, kill: bool):
        try:
            self.state
        except:
            self.state = 'ying'

        if self.state == 'ying':
            self.statusBar.showMessage("Status: ")
            self.statusBar.setStyleSheet("background-color : pink")
            self.state = 'yang'
            self.statusTimer.start(700)
        elif self.state == 'yang':
            self.statusBar.showMessage("Status: BUSY")
            self.statusBar.setStyleSheet("background-color : pink")
            self.state = 'ying'
            self.statusTimer.start(700)
        elif self.state == 'idle':
            self.state == 'ying'
            self.statusTimer.start(700)

        if kill:
            self.statusBar.showMessage("Status: IDLE")
            self.statusBar.setStyleSheet("background-color: rgb(239, 239, 239);")
            self.statusTimer.stop()
            self.state == 'idle'

    def reportProgress(self, ETA, t0, init: bool, ping: bool, kill: bool):
        if init:
            self.ETA = ETA
            self.t0 = t0
            progText = 'Running... Estimated time for completion: ' + str(self.ETA/60)[0:5] + ' minutes.'
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
                print('no good')

        if kill:
            self.progressBar_7.setValue(int(100))
            self.timer.stop()

    def launchEmbedding(self):
        self.progressBar_7.reset()
        self.pushButton_19.setEnabled(False)
        # Step 2: Create a QThread object
        self.thread = QtCore.QThread()
        # Step 3: Create a worker object
        self.worker = Worker()
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run_embedding)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.reportProgress)
        self.worker.status.connect(self.status)
        self.worker.msg.connect(self.textBrowser.append)
        # Step 6: Start the thread
        self.thread.start()

        # Final resets
        self.worker.finished.connect(
            lambda: self.comboBox_2.setEnabled(True)
        )

        self.worker.finished.connect(self.update_plot)

        self.worker.finished.connect(
            lambda : self.progressBar_7.setValue(100))

        self.worker.finished.connect(
            lambda: self.pushButton_19.setEnabled(True)
        )

        self.worker.finished.connect(self.plotArea.autoRange)

    def launchKmeans(self):
        self.progressBar_7.reset()
        self.pushButton_23.setEnabled(False)
        # Step 2: Create a QThread object
        self.thread = QtCore.QThread()
        # Step 3: Create a worker object
        self.worker = Worker()
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run_clustering)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.reportProgress)
        self.worker.msg.connect(self.textBrowser.append)
        self.worker.status.connect(self.status)
        # Step 6: Start the thread
        self.thread.start()

        # Final resets
        self.worker.finished.connect(
            lambda: self.checkBox_5.setEnabled(True)
        )

        self.worker.finished.connect(self.update_plot)

        self.worker.finished.connect(
            lambda: self.progressBar_7.setValue(100)
        )

        self.worker.finished.connect(
            lambda: self.pushButton_23.setEnabled(True)
        )

    def user_anchor_points(self, index, outName, save: bool, reset: bool, query: bool):

        if query:
            print('Query is triggered, return list')
            print(self.current_indices)
            flat_list = [item for sublist in self.current_indices for item in sublist]
            print('Selected indices: ' + ' '.join([str(elem) for elem in flat_list]))
        else:
            #Check whether index is a group or individual point
            if len(index) > 1:
                index = [index[0]]

            try:
                self.current_indices
            except:
                self.current_indices = []
                self.current_indices.append(index)
                self.textBrowser_4.setText(str(len(self.current_indices)))
            else:
                self.current_indices.append(index)
                self.textBrowser_4.setText(str(len(self.current_indices)))

        if reset:
            self.current_indices = []
            self.textBrowser_4.setText(str(len(self.current_indices)))

        if save:
            outFile = str(os.getcwd()) + '/' + outName
            np.savetext(outFile, self.current_indices, delimiter=",")

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
            lx = np.argwhere(embedding[:, 0] == x)
            ly = np.argwhere(embedding[:, 1] == y)
            #lx = np.argwhere(plot.data['x'] == x)
            #ly = np.argwhere(plot.data['y'] == y)
            i = np.intersect1d(lx, ly).tolist()
            indices += i
        indices = list(set(indices))  #Indicies are returned in order, undecided if this is necessarily OK...
        lastClicked = points
        lastPlot = plot
        print('Selected indices: ' + ' '.join([str(elem) for elem in indices]))

        if self.radioButton_2.isChecked():
            self.user_anchor_points(indices, '', save=False, reset=False, query=False)


        #Save the Z index for miniDRGN
        if len(indices) > 1:
            self.currentZind = indices[0]
        else:
            self.currentZind = indices

    def update_plot(self): ###Try break this up or move to a seperate thread for smoothness and speed.

        def map_to_colour(slice):
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
            return cm[slice]

        def plot_trajectories():
            '''
            ###########################################################################################################
            Plot (pseudo)trajectories if required
            ###########################################################################################################
            '''
            # Define trajectories by index
            traj = pg.mkPen((181, 34, 34), width=3, style=QtCore.Qt.DotLine)
            shadowPen = None  # or pg.mkPen((255, 255, 255), width=3)
            index_dic = ([4640, 13515, 3635, 8262, 7310, 12757],
                         [4608, 15361, 10880, 15873, 6660, 5004, 9344, 15362])
            try:
                self.curvePlots
            except:
                self.curvePlots = []
            else:
                for curvePlot in self.curvePlots:
                    self.plotArea.removeItem(curvePlot)

            # Iterate over trajectories and plot these.
            for n, index in enumerate(index_dic):
                curvePlot = pg.PlotCurveItem(
                    pen=traj,
                    shadowPen=shadowPen,
                    name="trajectory_" + str(n)
                )

                curvePlot.setData(x=embedding[index][:, 0], y=embedding[index][:, 1])
                self.curvePlots.append(curvePlot)
                self.plotArea.addItem(curvePlot)

        def plot_rois():
            '''
            ###########################################################################################################
            Plot regions of interest (ROIs) if applicable
            ###########################################################################################################
            '''
            axX = self.plotArea.getAxis('bottom')
            x_min, x_max = axX.range  # <------- get range of x axis
            axY = self.plotArea.getAxis('left')
            y_min, y_max = axY.range  # <------- get range of y axis

            try:
                self.ROIs
            except:
                com_x, com_y = (x_max + x_min) / 2, (y_max + y_min) / 2

                roiPen = pg.mkPen((22, 208, 115), width=3, style=QtCore.Qt.SolidLine)
                HoverPen = pg.mkPen((255, 255, 0), width=3, style=QtCore.Qt.DashLine)

                self.ROIs = pg.PolyLineROI(
                    [[0.2 * x_min + com_x, com_y],
                     [com_x, 0.2 * y_max + com_y],
                     [0.2 * x_max + com_x, com_y],
                     [com_x, 0.2 * y_min + com_y]],
                    closed=True, pen=roiPen, handlePen=(153, 51, 255), hoverPen=HoverPen)

                ## handles rotating around center
                #self.ROIs.addRotateHandle([com_x, com_y + 0.1 * y_max], [com_x, com_y])
                #self.ROIs.addScaleHandle([com_x + 0.05 * x_max, com_y - 0.05 * y_max], [com_x, com_y])
            finally:
                self.plotArea.addItem(self.ROIs)

        def plot_scatter(colour: bool):
            '''
            ###########################################################################################################
            Plot latent space as scatter plot
            ###########################################################################################################
            '''
            self.plotArea.clear()
            self.plotArea.legend.clear()
            self.scatterPlots = []
            interactPlot = pg.ScatterPlotItem(
                pen=None,  # pg.mkPen((0, 0, 0), width=1),
                symbol='o',
                symbolPen=None,
                symbolSize=5,
                symbolBrush=(65, 105, 225, 100),
                downsample=True,
                downsampleMethod='mean',
                hoverable=True,
                hoverSymbol='s',
                hoverSize=15,
                hoverPen=pg.mkPen((0, 0, 0), width=3),
                hoverBrush=pg.mkBrush(250, 128, 114),
                brush=pg.mkBrush(65, 105, 225, 100),
                name='obs A'
            )
            interactPlot.setData(
                pos=embedding,
                data=range(0, len(embedding)))
            interactPlot.sigClicked.connect(self.plot_clicked)
            self.scatterPlots.append(interactPlot)
            self.plotArea.addItem(interactPlot)
            self.plotArea.disableAutoRange()

            if colour:
                try:
                    labels
                except:
                    colours = pg.mkBrush(65, 105, 225, 100)
                else:
                    # if len(labels) == len(embedding):
                    unique = set(labels)
                    unique_scaled = [item / (len(unique) - 1) for item in unique]
                    colour_set = list(map(map_to_colour, unique_scaled))

                    cluster_list = []
                    for label in set(labels):
                        cluster = [pt for subPlot in self.scatterPlots for pt in subPlot.points() if
                                   labels[pt.data()] == label]
                        cluster_list.append(cluster)

                    legend_items = ['obs; clstr_' + str(label) for label in set(labels)]

                    ###########################################################################################################
                    # Plot latent space after embedding.

                    self.plotArea.clear()
                    self.plotArea.legend.clear()
                    for subPlot in self.scatterPlots:
                        self.plotArea.removeItem(subPlot)

                    self.scatterPlots = []
                    for cluster_n, cluster in enumerate(cluster_list):
                        interactPlot = pg.ScatterPlotItem(
                            pen=pg.mkPen((0, 0, 0), width=1),
                            symbol='o',
                            symbolPen=None,  # pg.mkPen((0, 0, 0), width=3),
                            symbolSize=5,
                            symbolBrush=(65, 105, 225, 100),
                            downsample=True,
                            downsampleMethod='mean',
                            hoverable=True,
                            hoverSymbol='s',
                            hoverSize=15,
                            hoverPen=pg.mkPen((0, 0, 0), width=3),
                            hoverBrush=pg.mkBrush(250, 128, 114),
                            brush=colour_set[cluster_n],
                            name=legend_items[cluster_n]
                        )
                        interactPlot.setData(
                            pos=[pt.pos() for pt in cluster],
                            data=[pt.index() for pt in cluster])
                        interactPlot.sigClicked.connect(self.plot_clicked)

                        self.scatterPlots.append(interactPlot)
                        self.plotArea.addItem(interactPlot)

                    for plot in self.scatterPlots:
                        self.plotArea.addItem(plot)

        def heat_map():
            x_min, y_min = np.amin(embedding, axis=0) #self.plotArea.getAxis('bottom')
            #x_min, y_min = axX.range  # <------- get range of x axis
            x_max, y_max = np.amax(embedding, axis=0) #self.plotArea.getAxis('left')
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

            histo, _, _ = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=[x_range, y_range])

            densityColourMesh = pg.PColorMeshItem(edgecolors=None, antialiasing=False, cmap='grey')
            densityColourMesh.setZValue(-1)
            self.plotArea.addItem(densityColourMesh)

            densityColourMesh.setData(x, y, histo[:-1, :-1])

        '''
        ###########################################################################################################
        Process user input and update plotArea accordingly
        ###########################################################################################################
        '''

        if self.checkBox_5.checkState() != QtCore.Qt.Checked:
            plot_scatter(colour=False)
        else:
            plot_scatter(colour=True)

        if self.checkBox_8.checkState() == QtCore.Qt.Checked:
            plot_trajectories()
        else:
            try:
                self.curvePlots
            except:
                self.curvePlots = []
            else:
                for curvePlot in self.curvePlots:
                    self.plotArea.removeItem(curvePlot)

        if self.radioButton_3.isChecked():
            plot_rois()
        else:
            try:
                self.ROIs
            except:
                pass
            else:
                self.plotArea.removeItem(self.ROIs)

        #heat_map()

    def export_roi(self):
        if self.radioButton_3.isChecked():
            try:
                self.scatterPlots
            except:
                self.scatterPlots = [self.interactPlot]
            finally:
                if self.checkBox_7.isChecked():
                    select = [pt for subPlot in self.scatterPlots
                              for pt in subPlot.points()
                              if not self.ROIs.mapToItem(subPlot, self.ROIs.shape()).contains(pt.pos())]
                else:
                    select = [pt for subPlot in self.scatterPlots
                              for pt in subPlot.points()
                              if self.ROIs.mapToItem(subPlot, self.ROIs.shape()).contains(pt.pos())]

            indices_ROI = []
            for pt in select:
                indices_ROI.append(pt.index())

            #DO something with the indicies!

            self.textBrowser.append("A total of " + str(len(indices_ROI)) + " particles were exported and converted to RELION and cryoSPARC formats."
                                                                            "\n -------------------------")

    def export_clusters(self):
        if self.radioButton_4.isChecked():
            try:
                labels
            except:
                self.textBrowser.append("You must run clustering before you can export observation clusters "
                                        "\n -------------------------")
            else:
                try:
                    self.scatterPlots
                except:
                    self.scatterPlots = [self.interactPlot]
                finally:
                    cluster_list = []
                    for label in set(labels):
                        cluster = [pt for subPlot in self.scatterPlots for pt in subPlot.points() if labels[pt.data()] == label]
                        cluster_list.append(cluster)

                    self.textBrowser.append("Exporting particles as cryoSPARC and RELION formats... \n")

                    for num, cluster in enumerate(cluster_list):
                        indices_cluster = []
                        for pt in cluster:
                            indices_cluster.append(pt.index())

                    ###DO something with the indicies!

                        self.textBrowser.append("\t Done 'cluster_" + str(num) + "'. Contains " + str(len(indices_cluster)) + " particles.")
                self.textBrowser.append("\n -------------------------")

    def return_pressed(self):
        self.apix = 1

        # The use has pressed the Return key; log the current text as HTML

        self.currentZ = data[self.currentZind]

        z = [1.541084408760070801e+00, -1.007112026214599609e+00,
             -1.474042654037475586e+00, 1.732866287231445312e+00,
             -1.431217432022094727e+00, 2.217696428298950195e+00,
             1.486245989799499512e+00, -6.664801239967346191e-01]

        # ToolInstance has a 'session' attribute...
        print("Calling miniDRGN with config: %s, Z: %s, weights: %s, apix = %s"
              %
              (self.lineEdit_6.displayText(),
              self.currentZ,
              self.lineEdit_5.displayText(),
              self.apix)
              )

        #miniDRGN = miniDRGN('cryodrgn/49/config.pkl', z, 'cryodrgn/49/weights.49.pkl', apix=1)

        # vol = volume_from_grid_data(ArrayGridData(miniDRGN.vol, name='cryoDRGN vol'), wiggle)
        # vol.show()

    # def trajectory_inference(self):
    #     import pyVIA.core as via
    #     v0 = via.VIA(data, coloursIndex, jac_std_global=0.15, dist_std_local=1, knn=20, too_big_factor=v0_too_big,
    #                  root_user=[1], dataset='', random_seed=42, is_coarse=True, preserve_disconnected=True)
    #     v0.run_VIA()

    def setup(self, MainWindow):
        from ui_compiled import MainWindowInterface

        MainWindowInterface.setup_ui(self, MainWindow)
        MainWindowInterface.retranslate_ui(self, MainWindow)
        self.worker = Worker()

        # Set up the pyqtgraph
        self.plotArea = self.graphicsView.addPlot() #title="Interactive Latent Coordinate Space"
        self.plotArea.addLegend()
        self.interactPlot = pg.ScatterPlotItem(
            pen=None, #pg.mkPen((0, 0, 0), width=0.5), #Colour and style of edge/outline
            symbol='o',
            # symbolPen=None, #Colour and style of connecting line
            # symbolSize=5,
            # symbolBrush=(65, 105, 225, 100),
            Downsample=True,
            hoverable=True,
            hoverSymbol='s', #Hover symbol is shape of hovered point
            hoverSize=15,
            hoverPen=pg.mkPen((0, 0, 0), width=3), #Hover pen is outline
            hoverBrush=pg.mkBrush(250, 128, 114), #Hover brush is colour of fill
            name='Observations (obs)'
        )
        self.plotArea.addItem(self.interactPlot)
        self.label.setPixmap(QtGui.QPixmap("./Wiggle.PNG"))

        '''
        Connections, signals and actions
        '''
        self.interactPlot.sigClicked.connect(self.plot_clicked)
        self.stackedWidget.setCurrentIndex(0)
        self.stackedWidget_3.setCurrentIndex(5)

        self.comboBox_9.setCurrentIndex(2)
        self.comboBox_7.setCurrentIndex(2)
        self.comboBox_8.setCurrentIndex(3)
        self.checkBox_5.setEnabled(False)
        self.comboBox_5.currentIndexChanged['int'].connect(self.update_plot)
        self.comboBox.currentIndexChanged['int'].connect(self.stackedWidget.setCurrentIndex)
        self.comboBox.activated['int'].connect(self.stackedWidget.setCurrentIndex)
        self.comboBox_2.activated['int'].connect(self.stackedWidget_3.setCurrentIndex)

        self.radioButton_4.toggled['bool'].connect(self.checkBox_7.setDisabled)
        self.radioButton_4.clicked['bool'].connect(self.pushButton_14.setEnabled)
        self.radioButton_4.clicked['bool'].connect(self.pushButton_15.setEnabled)
        self.radioButton_3.clicked['bool'].connect(self.pushButton_15.setEnabled)
        self.radioButton_3.clicked['bool'].connect(self.pushButton_14.setEnabled)
        self.radioButton_3.toggled['bool'].connect(self.checkBox_7.setEnabled)
        self.radioButton_3.toggled['bool'].connect(self.update_plot)

        self.checkBox.toggled['bool'].connect(self.pushButton_27.setDisabled)
        self.checkBox_12.toggled['bool'].connect(self.lineEdit_3.setEnabled)
        self.checkBox_12.clicked['bool'].connect(self.lineEdit_3.setEnabled)
        self.checkBox_8.stateChanged.connect(self.update_plot)
        self.checkBox_5.stateChanged.connect(self.update_plot)

        self.pushButton_15.setEnabled(False)
        self.pushButton_14.setEnabled(False)
        self.pushButton_3.clicked.connect(
            partial(self.browse_file, "Select canonical map", self.pushButton_3, "Volumes (*.mrc)"))
        self.pushButton_4.clicked.connect(
            partial(self.browse_file, "Select latent space", self.pushButton_4, "Pickle (*.pkl)"))
        self.pushButton_6.clicked.connect(
            partial(self.browse_file, "Select consensus map", self.pushButton_6, "Pickle (*.pkl)"))
        self.pushButton_7.clicked.connect(
            partial(self.browse_file, "Select network weights", self.pushButton_7, "Pickle (*.pkl)"))
        self.pushButton_8.clicked.connect(
            partial(self.browse_file, "Select latent space (z.pkl)", self.pushButton_8, "Pickle (*.pkl)"))
        self.pushButton_19.clicked.connect(self.launchEmbedding)
        self.pushButton_23.clicked.connect(self.launchKmeans)

        self.timer.timeout.connect(lambda: self.reportProgress('', '', False, True, False))
        self.statusTimer.timeout.connect(lambda: self.status(False))
        self.pushButton_2.clicked.connect(lambda : self.user_anchor_points('', '', save=False, query=False, reset=True))
        self.pushButton.clicked.connect(lambda : self.user_anchor_points('', '', save=False, query=True, reset=False))
        self.pushButton_15.clicked.connect(lambda : self.export_roi())
        self.pushButton_15.clicked.connect(lambda : self.export_clusters())
        self.pushButton_27.clicked.connect(self.return_pressed)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

class Wiggle(object):
    import sys
    app = QtWidgets.QApplication(sys.argv)

    #Initialise the Worker class
    #worker = Worker()

    #Initialise the Ui_MainWindow class
    MainWindow = QtWidgets.QMainWindow()
    global ui
    ui = Main()

    #Run UI setup method of Ui_MainWindow class
    ui.setup(MainWindow)

    #Show UI/execute
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    UiInstance = Wiggle()
