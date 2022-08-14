import mrcfile
import numpy as np

class miniSPARC:
    def __init__(self, consensus, components, apix, flip: bool):
        self.apix = apix
        self.flip = flip
        self.box = consensus.shape[0]
        self.consensus = consensus
        self.components = components
        self._c = len(components)
        self.component_vector = self.components.reshape(self._c, self.box**3)

    def generate(self, zdim):
        _apply = np.dot(zdim.reshape(1, self._c), self.component_vector).reshape(self.box, self.box, self.box)
        return self.consensus + _apply.astype(np.float32)

if __name__ == "__main__":
    consensus = 'cryosparc/cryosparc_P44_J320_map.mrc'
    consensus = mrcfile.open(consensus).data
    components = ['cryosparc/cryosparc_P44_J320_component_0.mrc',
                  'cryosparc/cryosparc_P44_J320_component_1.mrc',
                  'cryosparc/cryosparc_P44_J320_component_2.mrc',
                  'cryosparc/cryosparc_P44_J320_component_3.mrc',
                  'cryosparc/cryosparc_P44_J320_component_4.mrc']
    components = np.array([mrcfile.open(component).data for component in components])

    z = np.array([-100, 50, 50, 50, 50])

    miniSPARC = miniSPARC(consensus, components, apix=1, flip=False)
    vol = miniSPARC.generate(z)
    with mrcfile.new('something.mrc', overwrite=True) as mrc:
        mrc.set_data(vol)