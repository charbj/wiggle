import mrcfile
import numpy as np
import time
import cupy as cp

class miniSPARC:
    def __init__(self, consensus, components, apix, low_pass, flip: bool):
        self.apix = apix
        self.flip = flip
        self.box = consensus.shape[0]
        self.consensus = consensus
        self.components = components
        self._c = len(components)
        self.component_vector = self.components.reshape(self._c, self.box**3)
        self.low_pass = low_pass
        self.r = self.resolution_to_radius(self.apix, self.low_pass, self.box)
        self.mask_np = self.generate_sphere(self.box, 2 * self.r)
        self.mask_cp = cp.array(self.generate_sphere(self.box, 2 * self.r))

    def generate(self, zdim):
        _apply = np.dot(zdim.reshape(1, self._c), self.component_vector).reshape(self.box, self.box, self.box)
        vol = self.consensus + _apply.astype(np.float32)
        if self.low_pass == -1:
            return vol
        else:
            return self.fourier_filter_gpu(vol).astype(np.float32)

    def fourier_filter(self, map):
        fft_3d = np.fft.fftn(map)
        shift = np.fft.fftshift(fft_3d)
        return np.real(np.fft.ifftn(np.fft.ifftshift(np.multiply(shift, self.mask_np))))

    def fourier_filter_gpu(self, map):
        fft_3d = cp.fft.fftn(cp.array(map))
        shift = cp.fft.fftshift(fft_3d)
        return np.real(cp.asnumpy(cp.fft.ifftn(cp.fft.ifftshift(cp.multiply(shift, self.mask_cp)))))

    def generate_sphere(self, volumeSize, radius):
        x_ = np.linspace(0, volumeSize, volumeSize)
        y_ = np.linspace(0, volumeSize, volumeSize)
        z_ = np.linspace(0, volumeSize, volumeSize)
        center = int(volumeSize / 2)  # center can be changed here
        u, v, w = np.meshgrid(x_, y_, z_, indexing='ij')
        a = np.power(u - center, 2) + np.power(v - center, 2) + np.power(w - center, 2)
        b = np.where(a <= radius * radius, 1, 0)
        return b

    def resolution_to_radius(self, apix, resolution, box_size):
        pwr = (np.log(resolution / apix) / np.log(2)) + 1
        return (2**-pwr * box_size)

if __name__ == "__main__":
    start = time.time()
    consensus = 'gpcr/cryosparc_P28_J31_map.mrc'
    consensus = mrcfile.open(consensus).data
    components = ['gpcr/cryosparc_P28_J31_component_0.mrc',
                  'gpcr/cryosparc_P28_J31_component_1.mrc',
                  'gpcr/cryosparc_P28_J31_component_2.mrc',
                  'gpcr/cryosparc_P28_J31_component_3.mrc',
                  'gpcr/cryosparc_P28_J31_component_4.mrc',
                  'gpcr/cryosparc_P28_J31_component_5.mrc']
    components = np.array([mrcfile.open(component).data for component in components])

    z = np.array([0, 0, 0, 0, 0, 0])

    miniSPARC = miniSPARC(consensus, components, apix=0.65, low_pass=8, flip=False)
    end = time.time()
    print("load")
    print(end - start)

    start = time.time()
    for i in range(0,10):
        vol = miniSPARC.generate(z)
    end = time.time()
    print("total_gen_time")
    print(end - start)
    with mrcfile.new('cryosparc_P28_J31_map_wlp_8.mrc', overwrite=True) as mrc:
        mrc.set_data(vol)