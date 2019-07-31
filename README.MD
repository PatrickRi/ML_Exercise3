#README

All scripts and notebooks were implemented using Python.
The classical feature extraction is implemented in the Jupyter notebooks, which must be configured once, using the 
central `config.yml` file, which contains relevant properties. Additionally Jupyter exhibits sometimes problems with the 
current working directory, therefore the property `config_path` must be defined on the beginning of every notebook,
which points to the location of the `config.yml` file, unfortunately it is not possible to work with command line
options in Jupyter notebooks, therefore this manually edit is necessary.

The CNNs are implemented in simple Python scripts (`cars_cnn.py` and `fruits_cnn.py`), and are also configured using the
`config.yml` file, and 2 system properties `config-file`, pointing to the YAML file and `expname` indicating the name
of the experiment run, used for Tensorboard (for examples look into the relevant files, where default options are given 
as hint).

The required PIP packages are listed in `requirements.txt`. Additionally Tensorflow should be installed, including CUDA
and GPU support. One requirement for SIFT is `opencv-contrib`, which contains opencv + relevant extensions like SIFT,
which however is only working in an older version: , which must be installed.

Additionally an attempt was made, to implement the CNNs using GoogleNet or AlexNet as architecture, which however was 
not successful (see Report for details), therefore it resides in its own files (`fruits_alexnet.py, fruits_googlenet.py`).
