import mrcfile
import numpy as np
import os

class DataManager():
    def __init__(self, input, mode, subset=False, fraction=-1):
        self.input = input # list of file paths
        self.mode = mode # {cryodrgn, cryosparc_3dva, cryosparc_3dflx}
        self.subset = subset # bool
        self.fraction = fraction # int
        self.dtype = None  # {single, multi}


    def load_data(self):
        if self.dtype is None:
            self.check_data_type()

        if self.mode == 'cryodrgn':
            return self.cryodrgn_data_load()
        elif self.mode == 'cryosparc_3dva':
            return self.cryosparc_3dva_data_load()
        elif self.mode == 'cryosparc_3dflx':
            return self.cryosparc_3dflex_data_load()
        else:
            print("Problem parsing data file - data type (cryoDRGN or cryoSPARC) not recognised")
            return 0

    def check_data_type(self):
        "Check whether the data is cryoDRGN, cryoSPARC 3DVA or 3DFlex"
        input = self.input
        if len(input) == 3:
            print("Looks like multiple inputs - making a guess")
            # if os.path.isfile(input[0]) and os.path.isfile(input[1]) and os.path.isfile(input[2]):
            #     print("Found all files")
            self.dtype = 'multi'
        elif len(input) == 1:
            print("Looks like single input - making a guess")
            if os.path.isfile(input[0]):
                print("Found all files")
                self.dtype = 'single'
        else:
            print("Got too many or too few inputs. This shouldn't happen")
            self.dtype = None

    def unpack_single(self, _in):
        if self.mode == 'cryodrgn':
            _load = np.load(_in, allow_pickle=True)
            return _load['config'], _load['weights'], _load['z_space']
        elif self.mode == 'cryosparc_3dva':
            _load = np.load(_in, allow_pickle=True)
            return _load['particles'], _load['consensus_map'], _load['components']
        elif self.mode == 'cryosparc_3dflx':
            print("Not yet supported")
        elif self.mode is None:
            print("Got unknown datatype. This shoudn't happenen.")

    def get_subset(self, data):
        if self.fraction <= 0:
            return data
        idx = np.random.randint(data.shape[0], size=self.fraction)
        return data[idx]

    def cryodrgn_data_load(self):
        input = self.input
        if self.dtype == 'single':
            config, weights, z_space = self.unpack_single(input[0])
            d = z_space
        elif self.dtype == 'multi':
            config, weights, z_space = input[0], input[1], input[2]
            d = np.load(z_space, allow_pickle=True)
        else:
            print("No dtype defined...")
            return 0

        if self.subset:
            return config, weights, self.get_subset(d)
        else:
            return config, weights, d

    def cryosparc_3dva_data_load(self):
        import numpy.lib.recfunctions as rf
        input = self.input
        if self.dtype == 'single':
            particles, consensus_map, components = self.unpack_single(input[0])

        elif self.dtype == 'multi':
            particles, consensus_map, components = input[0], input[1], input[2]

            consensus_map = mrcfile.open(consensus_map).data
            particles = np.load(particles)
            components = components.split(',')
            components = np.array([mrcfile.open(component).data for component in components])
            # e.g. components = ['XXX_component_0.mrc', ..., 'XXX_component_2.mrc']

        else:
            print("No dtype defined...")
            return 0

        keys = [''.join(('components_mode_', str(component), '/value')) for component, _ in
                enumerate(components)]
        n = particles[keys]
        n = rf.repack_fields(n)
        d = n.copy().view('<f4').reshape((n.shape[0], len(components)))

        if self.subset:
            return consensus_map, components, self.get_subset(d)
        else:
            return consensus_map, components, d

    def cryosparc_3dflex_data_load(self):
        return 0