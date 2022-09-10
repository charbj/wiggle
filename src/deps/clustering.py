from PyQt5 import QtCore
import time
import numpy as np

class Clusterer(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(float, float, bool, bool, bool)
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
        print('Begin clustering ...')

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

        print('Calculating ' + name + ' ...')

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
        print('Finished ' + name + ' ! \n -------------------------')
        self.progress.emit(1, 1, False, False, True)
        self.finished.emit()
        self.status.emit(True)