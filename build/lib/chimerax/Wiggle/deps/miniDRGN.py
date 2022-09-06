import pickle
import torch
import torch.nn as nn
import numpy as np
from PyQt5 import QtCore
import mrcfile
import math

class params:
    def __init__(self, config, weights):
        self.D = None
        self.Df = -1
        self.low_pass = -1
        self.activation = 'relu'
        self.config = config
        self.domain = None
        self.downsample = None # hard coded, need to fix
        self.enc_mask = None
        self.encode_mode = None
        self.l_extent = None
        self.n = 10
        self.norm = None
        # self.o = '/mnt/4TB_HD/projects/Charles_Bayly-Jones/Software/Wiggle/output'
        self.pdim = None
        self.pe_dim = None
        self.pe_type = None
        self.players = None
        self.prefix = 'vol_'
        self.qdim = None
        self.qlayers = None
        self.verbose = False
        self.weights = weights
        # self.z_end = None
        # self.z_start = None
        self.zdim = None
        # self.zfile = None

class ResidLinearMLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(ResidLinearMLP, self).__init__()
        layers = [ResidLinear(in_dim, hidden_dim) if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim), activation()]
        for n in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(ResidLinear(hidden_dim, out_dim) if out_dim == hidden_dim else nn.Linear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResidLinear(nn.Module):
    def __init__(self, nin, nout):
        super(ResidLinear, self).__init__()
        self.linear = nn.Linear(nin, nout)
        #self.linear = nn.utils.weight_norm(nn.Linear(nin, nout))

    def forward(self, x):
        z = self.linear(x) + x
        return z

class HetOnlyVAE(nn.Module):
    # No pose inference
    def __init__(self, lattice,  # Lattice object
                 qlayers, qdim,
                 players, pdim,
                 in_dim, zdim=1,
                 encode_mode='resid',
                 enc_mask=None,
                 enc_type='linear_lowf',
                 enc_dim=None,
                 domain='fourier',
                 activation=nn.ReLU):
        super(HetOnlyVAE, self).__init__()
        self.lattice = lattice
        self.zdim = zdim
        self.in_dim = in_dim
        self.enc_mask = enc_mask
        if encode_mode == 'conv':
            self.encoder = ConvEncoder(qdim, zdim * 2)
        elif encode_mode == 'resid':
            self.encoder = ResidLinearMLP(in_dim,
                                          qlayers,  # nlayers
                                          qdim,  # hidden_dim
                                          zdim * 2,  # out_dim
                                          activation)
        elif encode_mode == 'mlp':
            self.encoder = MLP(in_dim,
                               qlayers,
                               qdim,  # hidden_dim
                               zdim * 2,  # out_dim
                               activation)  # in_dim -> hidden_dim
        elif encode_mode == 'tilt':
            self.encoder = TiltEncoder(in_dim,
                                       qlayers,
                                       qdim,
                                       zdim * 2,
                                       activation)
        else:
            raise RuntimeError('Encoder mode {} not recognized'.format(encode_mode))
        self.encode_mode = encode_mode
        self.decoder = self.get_decoder(3 + zdim, lattice.D, players, pdim, domain, enc_type, enc_dim, activation)

    @classmethod
    def load(self, config, weights=None, device=None):
        '''Instantiate a model from a config.pkl

        Inputs:
            config (str, dict): Path to config.pkl or loaded config.pkl
            weights (str): Path to weights.pkl
            device: torch.device object

        Returns:
            HetOnlyVAE instance, Lattice instance
        '''
        cfg = miniDRGN.load_pkl(config) if type(config) is str else config
        c = cfg['lattice_args']
        lat = Lattice(c['D'], extent=c['extent'])
        c = cfg['model_args']
        if c['enc_mask'] > 0:
            enc_mask = lat.get_circular_mask(c['enc_mask'])
            in_dim = int(enc_mask.sum())
        else:
            assert c['enc_mask'] == -1
            enc_mask = None
            in_dim = lat.D ** 2
        activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c['activation']]
        model = HetOnlyVAE(lat,
                           c['qlayers'], c['qdim'],
                           c['players'], c['pdim'],
                           in_dim, c['zdim'],
                           encode_mode=c['encode_mode'],
                           enc_mask=enc_mask,
                           enc_type=c['pe_type'],
                           enc_dim=c['pe_dim'],
                           domain=c['domain'],
                           activation=activation)
        if weights is not None:
            ckpt = torch.load(weights) if type(weights) is str else weights.flatten()[0]
            #ckpt = torch.load(weights)
            #ckpt = weights.flatten()[0]
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if device is not None:
            model.to(device)
        return model, lat

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, *img):
        img = (x.view(x.shape[0], -1) for x in img)
        if self.enc_mask is not None:
            img = (x[:, self.enc_mask] for x in img)
        z = self.encoder(*img)
        return z[:, :self.zdim], z[:, self.zdim:]

    def cat_z(self, coords, z):
        '''
        coords: Bx...x3
        z: Bxzdim
        '''
        assert coords.size(0) == z.size(0)
        z = z.view(z.size(0), *([1] * (coords.ndimension() - 2)), self.zdim)
        z = torch.cat((coords, z.expand(*coords.shape[:-1], self.zdim)), dim=-1)
        return z

    def decode(self, coords, z, mask=None):
        '''
        coords: BxNx3 image coordinates
        z: Bxzdim latent coordinate
        '''
        return self.decoder(self.cat_z(coords, z))

    # Need forward func for DataParallel -- TODO: refactor
    def forward(self, *args, **kwargs):
        return self.decode(*args, **kwargs)

    def get_decoder(self, in_dim, D, layers, dim, domain, enc_type, enc_dim=None, activation=nn.ReLU):
        if enc_type == 'none':
            if domain == 'hartley':
                model = ResidLinearMLP(in_dim, layers, dim, 1, activation)
                ResidLinearMLP.eval_volume = PositionalDecoder.eval_volume  # EW FIXME
            else:
                model = FTSliceDecoder(in_dim, D, layers, dim, activation)
            return model
        else:
            model = PositionalDecoder if domain == 'hartley' else FTPositionalDecoder
            return model(in_dim, D, layers, dim, activation, enc_type=enc_type, enc_dim=enc_dim)

class Lattice:
    def __init__(self, D, extent=0.5, ignore_DC=True):
        assert D % 2 == 1, "Lattice size must be odd"
        x0, x1 = np.meshgrid(np.linspace(-extent, extent, D, endpoint=True),
                             np.linspace(-extent, extent, D, endpoint=True))
        coords = np.stack([x0.ravel(), x1.ravel(), np.zeros(D ** 2)], 1).astype(np.float32)
        self.coords = torch.tensor(coords)
        self.extent = extent
        self.D = D
        self.D2 = int(D / 2)

        # todo: center should now just be 0,0; check Lattice.rotate...
        # c = 2/(D-1)*(D/2) -1
        # self.center = torch.tensor([c,c]) # pixel coordinate for img[D/2,D/2]
        self.center = torch.tensor([0., 0.])

        self.square_mask = {}
        self.circle_mask = {}

        self.freqs2d = self.coords[:, 0:2] / extent / 2

        self.ignore_DC = ignore_DC

    def get_downsample_coords(self, d):
        assert d % 2 == 1
        extent = self.extent * (d - 1) / (self.D - 1)
        x0, x1 = np.meshgrid(np.linspace(-extent, extent, d, endpoint=True),
                             np.linspace(-extent, extent, d, endpoint=True))
        coords = np.stack([x0.ravel(), x1.ravel(), np.zeros(d ** 2)], 1).astype(np.float32)
        return torch.tensor(coords)

    def get_square_lattice(self, L):
        b, e = self.D2 - L, self.D2 + L + 1
        center_lattice = self.coords.view(self.D, self.D, 3)[b:e, b:e, :].contiguous().view(-1, 3)
        return center_lattice

    def get_square_mask(self, L):
        '''Return a binary mask for self.coords which restricts coordinates to a centered square lattice'''
        if L in self.square_mask:
            return self.square_mask[L]
        assert 2 * L + 1 <= self.D, 'Mask with size {} too large for lattice with size {}'.format(L, self.D)
        log('Using square lattice of size {}x{}'.format(2 * L + 1, 2 * L + 1))
        b, e = self.D2 - L, self.D2 + L
        c1 = self.coords.view(self.D, self.D, 3)[b, b]
        c2 = self.coords.view(self.D, self.D, 3)[e, e]
        m1 = self.coords[:, 0] >= c1[0]
        m2 = self.coords[:, 0] <= c2[0]
        m3 = self.coords[:, 1] >= c1[1]
        m4 = self.coords[:, 1] <= c2[1]
        mask = m1 * m2 * m3 * m4
        self.square_mask[L] = mask
        if self.ignore_DC:
            raise NotImplementedError
        return mask

    def get_circular_mask(self, R):
        '''Return a binary mask for self.coords which restricts coordinates to a centered circular lattice'''
        if R in self.circle_mask:
            return self.circle_mask[R]
        assert 2 * R + 1 <= self.D, 'Mask with radius {} too large for lattice with size {}'.format(R, self.D)
        #log('Using circular lattice with radius {}'.format(R))
        r = R / (self.D // 2) * self.extent
        mask = self.coords.pow(2).sum(-1) <= r ** 2
        if self.ignore_DC:
            assert self.coords[self.D ** 2 // 2].sum() == 0.0
            mask[self.D ** 2 // 2] = 0
        self.circle_mask[R] = mask
        return mask

    def rotate(self, images, theta):
        '''
        images: BxYxX
        theta: Q, in radians
        '''
        images = images.expand(len(theta), *images.shape)  # QxBxYxX
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        rot = torch.stack([cos, sin, -sin, cos], 1).view(-1, 2, 2)
        grid = self.coords[:, 0:2] / self.extent @ rot  # grid between -1 and 1
        grid = grid.view(len(rot), self.D, self.D, 2)  # QxYxXx2
        offset = self.center - grid[:, self.D2, self.D2]  # Qx2
        grid += offset[:, None, None, :]
        rotated = F.grid_sample(images, grid)  # QxBxYxX
        return rotated.transpose(0, 1)  # BxQxYxX

    def translate_ft(self, img, t, mask=None):
        '''
        Translate an image by phase shifting its Fourier transform

        Inputs:
            img: FT of image (B x img_dims x 2)
            t: shift in pixels (B x T x 2)
            mask: Mask for lattice coords (img_dims x 1)

        Returns:
            Shifted images (B x T x img_dims x 2)

        img_dims can either be 2D or 1D (unraveled image)
        '''
        # F'(k) = exp(-2*pi*k*x0)*F(k)
        coords = self.freqs2d if mask is None else self.freqs2d[mask]
        img = img.unsqueeze(1)  # Bx1xNx2
        t = t.unsqueeze(-1)  # BxTx2x1 to be able to do bmm
        tfilt = coords @ t * -2 * np.pi  # BxTxNx1
        tfilt = tfilt.squeeze(-1)  # BxTxN
        c = torch.cos(tfilt)  # BxTxN
        s = torch.sin(tfilt)  # BxTxN
        return torch.stack([img[..., 0] * c - img[..., 1] * s, img[..., 0] * s + img[..., 1] * c], -1)

    def translate_ht(self, img, t, mask=None):
        '''
        Translate an image by phase shifting its Hartley transform

        Inputs:
            img: HT of image (B x img_dims)
            t: shift in pixels (B x T x 2)
            mask: Mask for lattice coords (img_dims x 1)

        Returns:
            Shifted images (B x T x img_dims)

        img must be 1D unraveled image, symmetric around DC component
        '''
        # H'(k) = cos(2*pi*k*t0)H(k) + sin(2*pi*k*t0)H(-k)
        coords = self.freqs2d if mask is None else self.freqs2d[mask]
        center = int(len(coords) / 2)
        img = img.unsqueeze(1)  # Bx1xN
        t = t.unsqueeze(-1)  # BxTx2x1 to be able to do bmm
        tfilt = coords @ t * 2 * np.pi  # BxTxNx1
        tfilt = tfilt.squeeze(-1)  # BxTxN
        c = torch.cos(tfilt)  # BxTxN
        s = torch.sin(tfilt)  # BxTxN
        return c * img + s * img[:, :, np.arange(len(coords) - 1, -1, -1)]

class FTPositionalDecoder(nn.Module):
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation, enc_type='linear_lowf', enc_dim=None):
        super(FTPositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_type = enc_type
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 2, activation)

    def positional_encoding_geom(self, coords):
        '''Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi'''
        freqs = torch.arange(self.enc_dim, dtype=torch.float)
        if self.enc_type == 'geom_ft':
            freqs = self.DD * np.pi * (2. / self.DD) ** (freqs / (self.enc_dim - 1))  # option 1: 2/D to 1
        elif self.enc_type == 'geom_full':
            freqs = self.DD * np.pi * (1. / self.DD / np.pi) ** (freqs / (self.enc_dim - 1))  # option 2: 2/D to 2pi
        elif self.enc_type == 'geom_lowf':
            freqs = self.D2 * (1. / self.D2) ** (freqs / (self.enc_dim - 1))  # option 3: 2/D*2pi to 2pi
        elif self.enc_type == 'geom_nohighf':
            freqs = self.D2 * (2. * np.pi / self.D2) ** (freqs / (self.enc_dim - 1))  # option 4: 2/D*2pi to 1
        elif self.enc_type == 'linear_lowf':
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError('Encoding type {} not recognized'.format(self.enc_type))
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords):
        '''Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2 + 1, dtype=torch.float)
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def forward(self, lattice):
        '''
        Call forward on central slices only
            i.e. the middle pixel should be (0,0,0)

        lattice: B x N x 3+zdim
        '''
        # if ignore_DC = False, then the size of the lattice will be odd (since it
        # includes the origin), so we need to evaluate one additional pixel
        c = lattice.shape[-2] // 2  # top half
        cc = c + 1 if lattice.shape[-2] % 2 == 1 else c  # include the origin
        assert abs(lattice[..., 0:3].mean()) < 1e-4, '{} != 0.0'.format(lattice[..., 0:3].mean())
        image = torch.empty(lattice.shape[:-1])
        top_half = self.decode(lattice[..., 0:cc, :])
        image[..., 0:cc] = top_half[..., 0] - top_half[..., 1]
        # the bottom half of the image is the complex conjugate of the top half
        image[..., cc:] = (top_half[..., 0] + top_half[..., 1])[..., np.arange(c - 1, -1, -1)]
        return image

    def decode(self, lattice):
        '''Return FT transform'''
        assert (lattice[..., 0:3].abs() - 0.5 < 1e-4).all()
        # convention: only evalute the -z points
        w = lattice[..., 2] > 0.0
        lattice[..., 0:3][w] = -lattice[..., 0:3][w]  # negate lattice coordinates where z > 0
        result = self.decoder(self.positional_encoding_geom(lattice))
        result[..., 1][w] *= -1  # replace with complex conjugate to get correct values for original lattice positions
        return result

    def eval_volume(self, coords, D, extent, norm, mask=None, zval=None):
        '''
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        '''
        assert extent <= 0.5
        if zval is not None:
            zdim = len(zval)
            z = torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D, D, D), dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent, extent, D, endpoint=True, dtype=np.float32)):
            x = coords + torch.tensor([0, 0, dz])
            keep = x.pow(2).sum(dim=1) <= extent ** 2
            x = x[keep]
            if zval is not None:
                x = torch.cat((x, z.expand(x.shape[0], zdim)), dim=-1)
            with torch.no_grad():
                if dz == 0.0:
                    y = self.forward(x)
                else:
                    y = self.decode(x)
                    y = y[..., 0] - y[..., 1]
                slice_ = torch.zeros(D ** 2, device='cpu')
                slice_[keep] = y.cpu()
                slice_ = slice_.view(D, D).numpy()
            vol_f[i] = slice_
        vol_f = vol_f * norm[1] + norm[0]

        if mask is not None:
            vol_f = np.multiply(vol_f, mask)

        vol = fft.ihtn_center(vol_f[:-1, :-1, :-1])  # remove last +k freq for inverse FFT
        return vol

class fft:
    # def fft2_center(img):
    #     return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img, axes=(-1, -2))), axes=(-1, -2))

    def fftn_center(img):
        return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(img)))

    def ifftn_center(V):
        V = np.fft.ifftshift(V)
        V = np.fft.ifftn(V)
        V = np.fft.ifftshift(V)
        return V

    # def ht2_center(img):
    #     f = fft2_center(img)
    #     return f.real - f.imag

    def htn_center(img):
        f = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(img)))
        return f.real - f.imag

    def ihtn_center(V):
        V = np.fft.fftshift(V)
        V = np.fft.fftn(V)
        V = np.fft.fftshift(V)
        V /= np.product(V.shape)
        return V.real - V.imag

    def symmetrize_ht(ht):
        if len(ht.shape) == 2:
            D = ht.shape[0]
            ht = ht.reshape(1, *ht.shape)
        else:
            assert len(ht.shape) == 3
            D = ht.shape[1]
        assert D % 2 == 0
        B = ht.shape[0]
        sym_ht = np.empty((B, D + 1, D + 1), dtype=ht.dtype)
        sym_ht[:, 0:-1, 0:-1] = ht
        sym_ht[:, -1, :] = sym_ht[:, 0]  # last row is the first row
        sym_ht[:, :, -1] = sym_ht[:, :, 0]  # last col is the first col
        sym_ht[:, -1, -1] = sym_ht[:, 0, 0]
        if len(sym_ht) == 1:
            sym_ht = sym_ht[0]
        return sym_ht

class miniDRGN(QtCore.QObject):
    running = QtCore.pyqtSignal()
    done = QtCore.pyqtSignal()
    def __init__(self, ui, config, weights, apix):
        self.ui = ui
        self.config = config
        self.weights = weights
        self.apix = apix
        self.apix_curr = apix
        self.args = params(self.config, self.weights)
        self.load_into_gpu(self.args)
        super().__init__()

    def load_into_gpu(self, args):
        ## set the device
        use_cuda = torch.cuda.is_available()

        if use_cuda:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            print("GPU is detected")
        else:
            print('WARNING: No GPUs detected')

        cfg = self.overwrite_config(args.config, args)

        self.D = cfg['lattice_args']['D']  # image size + 1
        self.norm = cfg['dataset_args']['norm']

        if args.downsample:
            assert args.downsample % 2 == 0, "Boxsize must be even"
            assert args.downsample <= self.D - 1, "Must be smaller than original box size"

        self.model, self.lattice = HetOnlyVAE.load(cfg, args.weights)
        self.model.eval()

    def overwrite_config(self, config_pkl, args):
        config = self.load_pkl(config_pkl) if type(config_pkl) is str else config_pkl.flatten()[0]
        #config = config_pkl.flatten()[0] #self.load_pkl(config_pkl)
        # {
        #     'dataset_args': {
        #         'particles': '/mnt/RAID/charles/projects/vip1/200730/cryodrgn/particles.128.mrcs',
        #         'norm': [0, 383.01376],
        #         'invert_data': True,
        #         'ind': None,
        #         'keepreal': False,
        #         'window': True,
        #         'datadir': None,
        #         'ctf': '/mnt/RAID/charles/projects/vip1/200730/cryodrgn/ctf.pkl',
        #         'poses': '/mnt/RAID/charles/projects/vip1/200730/cryodrgn/pose.pkl',
        #         'do_pose_sgd': False
        #     },
        #
        #     'lattice_args': {
        #         'D': 129,
        #         'extent': 0.5,
        #         'ignore_DC': True
        #     },
        #
        #     'model_args': {
        #         'qlayers': 3,
        #         'qdim': 256,
        #         'players': 3,
        #         'pdim': 256,
        #         'zdim': 8,
        #         'encode_mode': 'resid',
        #         'enc_mask': 64,
        #         'pe_type': 'geom_lowf',
        #         'pe_dim': None,
        #         'domain': 'fourier'
        #     },
        #     'seed': 48647
        # }

        if args.norm is not None:
            config['dataset_args']['norm'] = args.norm
        v = vars(args)
        if 'D' in v and args.D is not None:
            config['lattice_args']['D'] = args.D + 1
        if 'l_extent' in v and args.l_extent is not None:
            config['lattice_args']['extent'] = args.l_extent
        for arg in (
        'qlayers', 'qdim', 'zdim', 'encode_mode', 'players', 'pdim', 'enc_mask', 'pe_type', 'pe_dim', 'domain',
        'activation'):
            # Set default pe_dim to None to maintain backwards compatibility
            if arg == 'pe_dim' and arg not in config['model_args']:
                assert v[arg] is None
                config['model_args']['pe_dim'] = None
                continue
            # Set default activation to ReLU to maintain backwards compatibility with v0.3.1 and earlier
            if arg == 'activation' and arg not in config['model_args']:
                assert v[arg] == 'relu'
                config['model_args']['activation'] = 'relu'
                continue
            if v[arg] is not None:
                config['model_args'][arg] = v[arg]
        return config

    def load_pkl(self, pkl):
        with open(pkl, 'rb') as f:
            x = pickle.load(f)
        return x

    def generate(self, z, flip):
        args = self.args
        mask = None
        if args.low_pass > 0:
            if args.downsample:
                shape = args.downsample + 1
            else:
                shape = self.D

            if shape > 0:
                mask = self.circular_mask(shape, (self.args.low_pass))

        if args.downsample and args.downsample > 16:
            extent = self.lattice.extent * (args.downsample/(self.D-1))
            vol = self.model.decoder.eval_volume(self.lattice.get_downsample_coords(args.downsample+1),
                                            args.downsample+1, extent, self.norm, mask, z)
        else:
            vol = self.model.decoder.eval_volume(self.lattice.coords, self.lattice.D, self.lattice.extent, self.norm, mask, z)

        if flip:
            vol = vol[::-1]

        if args.Df > 16:
            vol = self.crop_array(vol, args.Df)

        return vol.astype(np.float32)

    def crop_array(self, map, crop):
        if map.shape[0] != map.shape[1] != map.shape[2]:
            print("Map is not cubic... taking the first dimension. This may fail.")
        if map.shape[0] % 2 == 0:
            m = int((map.shape[0] / 2) - 1)
            l = m + 1 - crop
            h = m + crop
            return map[l:h, l:h, l:h]
        else:
            m = int((map.shape[0] / 2) - 1)
            l = m - crop
            h = m + crop + 1
            return map[l:h, l:h, l:h]

    def crop_to_box(self):
        crop = self.ui.spinBox_9.value()
        if crop >= self.D:
            return -1
        else:
            return int(crop / 2) if (crop % 2 == 0) else int((crop + 1) / 2)

    def downsample(self):
        if self.ui.spinBox_2B.value() != -1:
            newBox = math.ceil(self.ui.spinBox_2B.value() / 2) * 2
            self.apix_curr = self.apix * (self.D / newBox)
            return newBox
        else:
            self.apix_curr = self.apix
            return -1

    def circular_mask(self, D, diameter):
        """Generate an n-dimensional spherical mask."""
        position = (0.5*D, 0.5*D, 0.5*D)
        position = np.array(position).reshape((-1,) + (1,) * 3)
        arr = np.linalg.norm(np.indices((D,D,D)) - position, axis=0)
        return (arr <= diameter).astype(int)

    def resolution_to_radius(self, apix, resolution, box_size):
        pwr = (np.log(resolution / apix) / np.log(2)) + 1
        return (2 ** -pwr * box_size)

    def _update(self):
        self.running.emit()

        self.args.low_pass = self.resolution_to_radius(self.apix, self.ui.doubleSpinBox_9.value(), self.D)
        self.args.Df = self.crop_to_box()
        self.args.downsample = self.downsample()

        self.done.emit()

