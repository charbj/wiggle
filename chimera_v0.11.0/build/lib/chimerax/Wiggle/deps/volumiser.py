from PyQt5 import QtCore

class Volumiser(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(float, float, bool, bool, bool)
    msg = QtCore.pyqtSignal(str)
    status = QtCore.pyqtSignal(bool)
    def __init__(self, data, clusters, method):
        self.data = data
        self.clusters = clusters
        self.method = method
        self.labels = ''
        self.flip = True
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

    def volumiser(self):
        try:
            self._last_vol
        except:
            self._last_vol = None

        # The user has pressed the Return key; log the current text as HTML
        from .cryodrgn_minimal import miniDRGN
        from chimerax.map import volume_from_grid_data
        from chimerax.map_data import ArrayGridData

        # ToolInstance has a 'session' attribute...
        print(self.config)
        print(self.weights)
        print(self.apix)
        miniDRGN = miniDRGN(self.config, self.weights, self.apix, self.flip)

        grid = ArrayGridData(miniDRGN.generate(self.data[self.currentZind]), name='_vol_i')
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

        self.flip = True

        from .cryodrgn_minimal import miniDRGN
        from chimerax.map import volume_from_grid_data
        from chimerax.map_data import ArrayGridData
        from chimerax.map_filter.morph import morph_maps
        from chimerax.map_filter.morph_gui import MorphMapSlider

        Z_trajectory = self.TrajLists[ui.comboBox_10.currentIndex()]
        miniDRGN = miniDRGN(self.config, self.weights, self.apix, self.flip)

        volumes = []
        for coord in Z_trajectory:
            vol = volume_from_grid_data(
                ArrayGridData(
                    miniDRGN.generate(self.data[coord]), name='_vol_i'), self.wiggle, open_model=True, show_dialog=False)
            volumes.append(vol)

        frames = len(volumes) * 10 - 1
        step = 1 / frames
        volTraj = morph_maps(volumes, frames, 0, step, 1, (0.0, 1.0), False, False, None, True, False, 'all', 1, None)

        morph_slider = MorphMapSlider(self.wiggle, volTraj)
        self.slider_list.append(morph_slider.slider)

        for slider in self.slider_list:
            slider.valueChanged.connect(self.refresh_trajectories)