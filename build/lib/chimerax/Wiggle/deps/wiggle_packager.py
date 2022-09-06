import numpy as np
import mrcfile
import torch
import argparse
import os

class PackItUp:
    def __init__(self, args):
        self.args = args
        if self.args.mode == 'cryodrgn':
            self.read_cryodrgn()
        elif self.args.mode == '3dva':
            self.read_cryosparc()
        elif self.args.mode == '3dflex':
            print("not yet implemented, sorry...")
        else:
            print("Not recognised mode... this shouldn't happen.")

    def read_cryodrgn(self):
        config = np.load(self.args.config, allow_pickle=True)
        weights = torch.load(self.args.weights)
        z_space = np.load(self.args.z_space, allow_pickle=True)
        apix = self.args.apix
        if not os.path.isfile(self.args.output):
            np.savez_compressed(self.args.output, apix=apix, config=config, weights=weights, z_space=z_space)
        else:
            print(f'\033[93m {self.args.output} already exists. Exiting... \033[0m')

    def read_cryosparc(self):
        particles = np.load(self.args.particles)
        consensus_map = mrcfile.open(self.args.map).data
        components = [mrcfile.open(component).data for component in self.args.components]
        if not os.path.isfile(self.args.output):
            np.savez_compressed(self.args.output, particles=particles, consensus_map=consensus_map,
                                components=components)
        else:
            print(f'\033[93m {self.args.output} already exists. Exiting... \033[0m')

if __name__ == "__main__":
    def valid_mrc(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.mrc','.MRC'):
            raise argparse.ArgumentTypeError('File must have a .mrc extension')
        return param

    def valid_pkl(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.pkl',):
            raise argparse.ArgumentTypeError('File must have a .pkl extension')
        return param

    def valid_cs(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.cs',):
            raise argparse.ArgumentTypeError('File must have a .cs extension')
        return param

    def valid_output(param):
        base, ext = os.path.splitext(param)
        if ext.lower() not in ('.npz',):
            return param + '.npz'
        return param

    # create the top-level parser
    my_parser = argparse.ArgumentParser(
        description="Simple program to compile and bundle cryoEM conformational heterogeneity data types.",
        prog="python wiggle_packager.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # create sub-parser
    sub_parsers = my_parser.add_subparsers(
        title="Operating modes",
        description="Select the operating mode. Supports cryoDRGN, cryoSPARC 3DVA, or cryoSPARC 3DFlex file types.",
        dest="mode",
        required=True,
    )

    # create the parser for the "agent" sub-command
    parser_agent = sub_parsers.add_parser("cryodrgn", help="cryoDRGN mode")
    parser_agent.add_argument(
        "--config",
        type=str,
        help="config file (.pkl)",
        required=True
    )
    parser_agent.add_argument(
        "--weights",
        type=str,
        help="cryoDRGN weights file containing learned parameters (.pkl)",
        required=True
    )
    parser_agent.add_argument(
        "--z_space",
        type=str,
        help="cryoDRGN latent space (z-space) file containing learned per-particle latent variables (.pkl)",
        required=True
    )
    parser_agent.add_argument(
        "--apix",
        type=float,
        help="Pixel size in Angstrom per pixel.",
        required=True
    )
    parser_agent.add_argument(
        "--output",
        type=valid_output,
        help="Name of the output file. If no extension given, string will be appended. (.npz)",
        required=True
    )


    # create the parse for the "learner" sub-command
    parser_learner = sub_parsers.add_parser("3dva", help="cryoSPARC 3DVA mode")
    parser_learner.add_argument(
        "--map",
        type=valid_mrc,
        help="Consensus or basis map (.mrc)",
        required=True
    )
    parser_learner.add_argument(
        "--components",
        type=valid_mrc,
        help="List of component maps. Accepts wildcard e.g. components*.mrc (.mrc)",
        nargs='+',
        required=True
    )
    parser_learner.add_argument(
        "--particles",
        type=valid_cs,
        help="cryoSPARC particle.cs file containing per-particle latent variables (.cs)",
        required=True
    )
    parser_learner.add_argument(
        "--output",
        type=valid_output,
        help="Name of the output file. If no extension given, string will be appended. (.npz)",
        required=True
    )

    # create the parse for the "tester" sub-command
    parser_tester = sub_parsers.add_parser("3dflex", help="cryoSPARC 3DFlex mode")
    parser_tester.add_argument(
        "-t",
        "--max_steps",
        type=int,
        help="Number of agent's steps",
        default=int(1e6),
    )
    parser_tester.add_argument(
        "--render", action="store_true", help="Render the environment"
    )
    parser_tester.add_argument(
        "-f", "--model_path", type=str, help="Path to saved model"
    )

    args = my_parser.parse_args()
    PackItUp(args)