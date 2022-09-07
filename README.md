![WIGGLE logo](https://github.com/charbj/wiggle/blob/main/src/resources/Wiggle.PNG)

# WIGGLE 0.2.1 (alpha)
Graphical user interface to integrate cryo-EM flexibility analyses with ChimeraX.

## What is WIGGLE?
Wiggle is a software package that integrates within [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/) to perform conformational flexibility analysis. Wiggle facilitates more interactive, intuitive, and rapid analysis of cryo-EM flexibility data types. Wiggle provides a tool to structural biologists to seemlessly visualise [cryoDRGN](http://cb.csail.mit.edu/cb/cryodrgn/) or [cryoSPARC](https://cryosparc.com/) [3D variability](https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/tutorial-3d-variability-analysis-part-one) results while simultaneously visualising 3D volumetric data. The goal is to facilitate the sharing, deposition, analysis, and interpretation of next generation cryo-EM data types. Wiggle provides several key tools to achieve these goals:
* Integrated into UCSF ChimeraX (easy install).
* Read and write a single file format (compressed numpy array, .npz). 
* Facilitates the sharing and deposition of structural dynamics data. 
* Perform complex tasks with a GUI and interact with the variability landscape.
* Rapid volume rendering to enable near realtime feedback.
* Import and export cryoSPARC data file times to enable further analysis.

## Dependencies
* UCSF ChimeraX 1.3 (not currently comptatible with version 1.4, should work on versions below 1.3 but not tested).
* CryoSPARC and cryoDRGN functionality depends on a CUDA accelerated GPU. CPU can work in principal, although it's quite slow and not currently implemented.
* The following pip-installable packaged:
  * cupy_cuda100 - !! Important that this matches your CUDA version. Here CUDA 10.0
  * mrcfile
  * phate
  * umap-learn
  * pyqtgraph
  * scikit-learn
  * torch

## Installation - developmental version
This is an experimental and developmental version, currently in testing. In the future, Wiggle will be available via the UCSF ChimeraX toolshed. For now, to use Wiggle you must install it manually (see below).

  ### Verify your CUDA version. 
Cupy must match your CUDA library versions, otherwise it will complain. Check the suffix on the end of your CUDA libXXX.so files. My version is CUDA 10, so the libraries end with libXXX.so.10.0 - use this for your cupy installation (even if nvidia-smi reports something else e.g. cuda 10.1 or 10.2, etc).

      $ ls /usr/local/cuda/lib64/*
      
      libcudart.so.10.0   libcusolver.so.10.0  libcufftw.so.10.0  etc ...

The CUDA libraries must be on the terminal path for cupy to work. Either 1) set the following path before launching ChimeraX via the command line, 2) set the CUDA lib path on your system in your `~/.bashrc` file
      
      export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
      
  ### Get dependencies
  
First install [UCSF ChimeraX 1.3](https://www.cgl.ucsf.edu/chimerax/older_releases.html).

Install the above pip packages using the UCSF ChimeraX python

      /path/to/ChimeraX/bin/python3.9 -m pip install cupy-cuda100 mrcfile phate umap-learn pyqtgraph scikit-learn torch

### Clone WIGGLE and install into ChimeraX.

      mkdir /path/to/save/software/
      cd /path/to/save/software/
      git clone https://github.com/charbj/wiggle.git
      
Launch UCSF ChimeraX (either via GUI or command line) e.g.

      /usr/local/programs/chimerax-1.3/bin/ChimeraX
      
In the UCSF ChimeraX command line, run the following command (ensuring you modify the path appropriately for your system):

      devel build ~/path/to/save/software/wiggle/; devel install ~/path/to/save/software/wiggle/

The path should match your git cloned directory...

To launch WIGGLE either run the command `ui tool show wiggle` or launch from `Tools > General > Wiggle`

## Running WIGGLE via the command line
Wiggle has a packaging feature that compiles cryoSPARC or cryoDRGN output files into the single binary format for ease of sharing, deposition, and distribution. 

While Wiggle will natively read the cryoDRGN and cryoSPARC analysis output, it will also read a compressed single-file format. This single file is easier to share amongst colleagues and can be generated with a simple command. 

    /path/to/chimerax-1.3/bin/python3.9 /path/to/wiggle/src/wiggle.py --help

e.g. 1 - Compile cryoDRGN outputs into single-file format

    /path/to/chimerax-1.3/bin/python3.9 /path/to/wiggle/src/wiggle.py cryodrgn --config config.pkl --weights weights.49.pkl --z_space z.49.pkl --apix 1.6 --output example.npz

e.g. 2 - Compile cryoSPARC outputs into single-file format
    
    /usr/local/programs/chimerax-1.3/bin/python3.9 ~/projects/software/wiggle/wiggle_0.2.1/src/wiggle.py 3dva --map cryosparc_P49_J924_map.mrc --components cryosparc_P49_J924_component_*.mrc --particles cryosparc_P49_J924_particles.cs --output example.npz
    
## FAQs

### How do I use the cryoSPARC 3D Flex mode?
This is not currently available, pending further details from the cryoSPARC team. This will be implemented in a future release. It is currently a place holder... sorry!

### I get an error that cupy can't find a specific libXXX.so.X.Y.Z file?
Make sure your cupy installation exactly matches your cuda version. See installation instructions above.

### How can I explore WIGGLE if I don't have any cryoSPARC or cryoDRGN results?
[Ellen Zhong](https://github.com/zhonge), the main author behind cryoDRGN, has made some pre-computed results available via [Zenodo](https://zenodo.org/record/4355284#.YxiKXNJBy4o). Check out her [paper](https://www.nature.com/articles/s41592-020-01049-4) for details.

## Screen captures and GUI example
### Night and day mode examples of the Wiggle UI
![WIGGLE night](https://github.com/charbj/wiggle/blob/main/screengrabs/wiggle.png)
![WIGGLE day](https://github.com/charbj/wiggle/blob/main/screengrabs/wiggle2_ui.png)

### Example of Wiggle within the ChimeraX interface
![WIGGLE ChimeraX](https://github.com/charbj/wiggle/blob/main/screengrabs/wiggle_chimera.png)
