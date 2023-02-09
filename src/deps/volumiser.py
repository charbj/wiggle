from Qt import QtCore
import time
import numpy as np

class Volumiser(QtCore.QObject):
    finished = QtCore.Signal()
    progress = QtCore.Signal(float, float, bool, bool, bool)
    msg = QtCore.Signal(str)
    status = QtCore.Signal(bool)
    def __init__(self, ui, session, state, volume_engine, data, apix, mode, flip):
        self.ui = ui
        self.wiggle = session
        self.wiggle_state = state
        self.engine = volume_engine
        self.data = data
        self.mode = mode
        self.zflip = flip
        super().__init__()

    # def volumiser_by_cluster(self):
    #     pass

    def volume_by_cluster(self):
        pass

    def volume_by_traj(self):
        self.status.emit(False)
        ## TIMING TEST
        t0 = time.time()
        # dummy = operator.fit(small_data)
        time.sleep(1)
        ETA = 30
        self.progress.emit(ETA, t0, True, True, False)

        from chimerax.map_data import ArrayGridData

        self.msg.emit('Rendering...')
        key = [k for k, v in self.wiggle_state.items() if v['legend'] == self.ui.comboBox_10.currentText()][0]
        Z_trajectory = self.wiggle_state[key]['indices']
        self.grids = []
        for coord in Z_trajectory:
            grid = ArrayGridData(self.engine.generate(self.data[2][coord], self.zflip),
                                 name=''.join((self.mode, '_vol_i')),
                                 step=(self.engine.apix_curr, self.engine.apix_curr, self.engine.apix_curr))
            self.grids.append(grid)

        # Finished close thread and emit complete signal
        self.msg.emit('Finished.')
        self.progress.emit(1, 1, False, False, True)
        self.finished.emit()
        self.status.emit(True)

    def volume_by_MEP(self):
        self.status.emit(False)
        ## TIMING TEST
        t0 = time.time()
        # dummy = operator.fit(small_data)
        time.sleep(1)
        ETA = 30
        self.progress.emit(ETA, t0, True, True, False)

        from chimerax.map_data import ArrayGridData

        self.msg.emit('Rendering...')
        key = [k for k, v in self.wiggle_state.items() if v['legend'] == self.ui.comboBox_6.currentText()][0]
        Z_trajectory = self.wiggle_state[key]['indices']
        self.grids = []
        for coord in Z_trajectory:
            grid = ArrayGridData(self.engine.generate(self.data[2][coord], self.zflip),
                                 name=''.join((self.mode, '_vol_i')),
                                 step=(self.engine.apix_curr, self.engine.apix_curr, self.engine.apix_curr))
            self.grids.append(grid)

        # Finished close thread and emit complete signal
        self.msg.emit('Finished.')
        self.progress.emit(1, 1, False, False, True)
        self.finished.emit()
        self.status.emit(True)

    def volume_by_component(self):
        self.status.emit(False)
        ### TIMING TEST
        t0 = time.time()
        # dummy = operator.fit(small_data)
        time.sleep(1)
        ETA = 30
        self.progress.emit(ETA, t0, True, True, False)

        from chimerax.map_data import ArrayGridData

        self.msg.emit('Rendering...')
        component_ind = self.ui.comboBox_12.currentIndex()
        steps = self.ui.spinBox_3.value()
        scaleFactorsObs = self.data[2][:, component_ind]
        std = np.std(scaleFactorsObs)
        scaleFactors = np.linspace(-2*std, 2*std, steps)
        weightVector = np.zeros(len(self.data[2][0]))

        self.grids = []
        self.samples = []
        for s in scaleFactors:
            weightVector[component_ind] = s
            self.samples.append(list(weightVector[:]))
            grid = ArrayGridData(self.engine.generate(weightVector, self.zflip),
                                 name=''.join((self.mode, '_vol_i')),
                                 step=(self.engine.apix_curr, self.engine.apix_curr, self.engine.apix_curr))
            self.grids.append(grid)

        # Finished close thread and emit complete signal
        self.msg.emit('Finished.')
        self.progress.emit(1, 1, False, False, True)
        self.finished.emit()
        self.status.emit(True)