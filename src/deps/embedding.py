from PyQt5 import QtCore
import time

class Embedder(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    exit = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(float, float, bool, bool, bool)
    status = QtCore.pyqtSignal(bool)
    def __init__(self, ui, technique_index, data_path, frac, subset: bool):
        self.ui = ui
        self.technique = technique_index
        self.data_path = data_path
        self.fraction = frac
        self.do_subset = subset
        self.previous_data = []
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
        if self.do_subset:
            if len(self.previous_data) != int(self.ui.lineEdit_3.text()):
                print("Using a subset of your data...")
                # self.msg.emit("1. Subsampling latent space to %s" % int(self.fraction))
                DataManager = DataManager(input=self.data_path, mode=self.ui.inputType[self.ui.comboBox.currentIndex()],
                                            subset=True, fraction=int(self.fraction))
                *self.data, = DataManager.load_data()
            else:
                # self.msg.emit("2. Subset size hasn't changed, using previous subset...")
                self.data = self.previous_data
        else:
            print("Using whole dataset. Subsampling can improve interactive experience. Try 25k data points...")
            DataManager = DataManager(input=self.data_path, mode=self.ui.inputType[self.ui.comboBox.currentIndex()],
                                      subset=False)
            *self.data, = DataManager.load_data()

        # Exit if data is not defined, no user input.
        if len(self.data) == 1:
            print("Exiting embedding; path not found, wrong type, or none")
            self.progress.emit(1, 1, False, False, True)
            self.exit.emit()
            self.status.emit(True)
            self.apix = 1
            return

        self.apix = DataManager.apix
        z_space = self.data[2]

        # global embedding
        if self.technique == 0:
            print("Running umap... ")
            import umap
            operator = umap.UMAP(random_state=42, verbose=1, densmap=True)
            ETA, t0 = self.estimate_time(operator)
            self.embedding = operator.fit_transform(z_space)
            print("Finished umap! \n -------------------------")

        if self.technique == 1:
            print("Running PCA... ")
            from sklearn.decomposition import PCA
            operator = PCA(n_components=2)
            operator.fit(z_space)
            ETA, t0 = self.estimate_time(operator)
            self.embedding = operator.transform(z_space)
            print("Finished PCA! \n -------------------------")

        if self.technique == 2:
            print("Running tSNE...")
            from sklearn.manifold import TSNE
            operator = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1)
            ETA, t0 = self.estimate_time(operator)
            self.embedding = operator.fit_transform(z_space)
            print("Finished tSNE! \n -------------------------")

        if self.technique == 3:
            print("Running PHATE... ")
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
            print("Finished PHATE! \n -------------------------")

        if self.technique == 4:
            print("Running cVAE... ")
            from cvae import cvae
            operator = cvae.CompressionVAE(z_space)
            operator.train()
            ETA, t0 = self.estimate_time(operator)
            self.embedding = operator.embed(z_space)
            print("Finished cVAE! \n -------------------------")

        self.progress.emit(ETA, t0, False, False, True)
        self.finished.emit()
        self.status.emit(True)