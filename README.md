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

## Video tutorials
1. [Setup & installation (6:53)](https://youtu.be/y99k88MszrY)
2. [Bundle (cryoDRGN or cryoSPARC 3DVA) results into .npz format (7:39)](https://youtu.be/k_-ghuqPsCM)
3. [User interface overview (5:13)](https://youtu.be/aIonC1oEYoo)
4. [A Wiggle/cryoDRGN walkthrough (in depth; 15:07)](https://youtu.be/IjUIO7fd5RI)

## Dependencies
* UCSF ChimeraX >=1.3 (Should work on versions below 1.3 but not tested).
* CryoSPARC and cryoDRGN functionality depends on a CUDA accelerated GPU. CPU can work in principal, although it's quite slow and not currently implemented.
* The following pip-installable packaged:
  * cupy_cuda100 - !! Important that this matches your CUDA version. Here CUDA 10.0
  * mrcfile
  * phate
  * umap-learn
  * pyqtgraph
  * scikit-image
  * torch [useful installation guide](https://pytorch.org/get-started/previous-versions/)
  * sklearn
  * pyyaml

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

      /path/to/ChimeraX/bin/python3.9 -m pip install cupy-cuda100 mrcfile phate umap-learn pyqtgraph scikit-image sklearn torch pyyaml

Some users have reported that the above method fails (maybe if ChimeraX was installed via `apt install .deb` - see https://github.com/charbj/wiggle/issues/2). If this is the case, try the following:

      /path/to/ChimeraX/bin/ChimeraX -m pip install cupy-cuda100 mrcfile phate umap-learn pyqtgraph scikit-image sklearn torch pyyaml

### Clone WIGGLE and install into ChimeraX.

      mkdir /path/to/save/software/
      cd /path/to/save/software/
      git clone https://github.com/charbj/wiggle.git
      
Launch UCSF ChimeraX (either via GUI or command line) e.g.

      /usr/local/programs/chimerax-1.3/bin/ChimeraX
      
In the UCSF ChimeraX command line, run the following command (ensuring you modify the path appropriately for your system):

      devel clean ~/path/to/where/you/saved/wiggle; devel build ~/path/to/where/you/saved/wiggle/; devel install ~/path/to/where/you/saved/wiggle/

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

### Volume rendering is slow for large boxes (cryoDRGN or cryoSPARC).
Before rendering many volumes, start in the interactive mode and tune the cropping and downsampling options. Rendering whole volumes at the original sampling size is usually not necessary and can be cumbersomely slow. To improve speeds try the following:

In cryoDRGN mode, try down sampling to 128 pixels and then find an appropriate cropping to remove unneccessary empty voxel. 

Likewise, in cryoSPARC mode, first crop the volume to remove empty solvent voxels and then downsample by a factor of ~2. 

## Screen captures and GUI example
### Night and day mode examples of the Wiggle UI
![WIGGLE night](https://github.com/charbj/wiggle/blob/main/screengrabs/wiggle.png)
![WIGGLE day](https://github.com/charbj/wiggle/blob/main/screengrabs/wiggle_ui2.png)

### Example of Wiggle within the ChimeraX interface
![WIGGLE ChimeraX](https://github.com/charbj/wiggle/blob/main/screengrabs/wiggle_chimera.png)


## Useful reading material for conformational flexibility and cryoEM
-[cryoDRGN](https://www.nature.com/articles/s41592-020-01049-4), & [2](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhong_CryoDRGN2_Ab_Initio_Neural_Reconstruction_of_3D_Protein_Structures_From_ICCV_2021_paper.pdf)

-[3DFlex](https://www.biorxiv.org/content/10.1101/2021.04.22.440893v1)

-[cryoSPARC 3DVA](https://www.sciencedirect.com/science/article/pii/S1047847721000071)

-[AlphaCryo4D](https://www.mdpi.com/1422-0067/23/16/8872/htm)

-[ManifoldEM 1](https://www.pnas.org/doi/10.1073/pnas.1419276111), [2](https://www.nature.com/articles/s41467-020-18403-x), & [3](https://www.biorxiv.org/content/10.1101/2021.06.18.449029v2.full) - also [ManifoldEM GUI](https://github.com/evanseitz/ManifoldEM_Python)




