# Optimizing an EGFR Binder via OpenFold
## Tori Grosskopf
### Applied Deep Learning in BME

### Previous Work
[**OpenFold: Retraining AlphaFold yields new insights into its learning mechanisms and capacity for generalization**](https://www.nature.com/articles/s41592-024-02272-z)

### Project Introduction and Overview
Epidermal Growth Factor Receptor (EGFR) is a transmembrane protein that has a critical role in cell growth, differentiation, and survival. It is frequently overexpressed or mutated in various cancers, and is currently a crucial target for cancer therapies. 
![image](https://github.com/user-attachments/assets/a1bbb13f-b99c-46ea-9774-2bb954d42c7d)

_Figure 1: Structure of EGFR_

The purpose of this project was to design and analyze a protein that binds to the extracellular matrix of EGFR. Previously, students in this course used various protein design models (BindCraft, RFDiffusion, AlphaFold, etc.) to generate a sequence that could potentially bind to EGFR as part of a competition held by [AdaptyvBio](https://design.adaptyvbio.com/). A smaller project was then conducted using the [ESM-2 model](https://huggingface.co/blog/AmelieSchreiber/protein-binding-partners-with-esm2) to analyze which of our sequences had the most potential to bind to the selected portion of EGFR, linked [here](https://github.com/torigrosskopf/deep-learning/blob/main/miniproject2.ipynb). This project selected the top performing sequence that project and implemented OpenFold, a new protein design model, to generate an optimized structure of our binding sequence.

**Introduction to OpenFold**
OpenFold is a model that serves as an adaptable implementation of AlphaFold2 to be adapted. It follows the architecture of AlphaFold2 neural nets while allowing customization by researchers. AlphaFold2 relies on large datasets with strict pre-processing steps, and is closed-source (the training pipeline is not disclosed). OpenFold is open-source, with pre-processing steps explicity available, documented, and adaptable. While the general convolutional methods are the same, researchers are able to modify most components and parameters. Essentially, OpenFold has opened a door for advancement in protein design via experimentation. It also does not require large datasets for training unlike AlphaFold2 - in its development, researchers intentionally trained OpenFold with progressively less data, where the model performed well based on lDDT-Cα analysis. 

![image](https://github.com/user-attachments/assets/976f61b1-1352-4aa9-a123-2c5f7570eb0c)

_Fig. 2: AlphaFold Neural Net Structure_

![image](https://github.com/user-attachments/assets/c81b1410-f94b-4d00-8c61-f4ed473ce45b)

_Fig. 3: Scatterplot of lDDT-Cα values of AlphaFold and OpenFold predictions on the CAMEO validation set. Demonstrates accuracy of OpenFold model._

The notebook used in this analysis is a simplified version of OpenFold, run with the selected EGFR binding sequence. The code can be found [here](https://github.com/aqlaboratory/openfold) on the aqlaboratory Github page. In this notebook, OpenFold has already been trained and the focus is on applying their version to the EGFR binder. In future studies, I would want to train OpenFold specifically on antibody datasets, and see how that impacts the confidence in predicting the structure of the binder. For now, the assumption is that OpenFold has been trained to exactly mimic AlphaFold2. 

## Using OpenFold to Generate a Structure from EGFR Binding Sequence

The first portion of the OpenFold notebook template ([here](https://github.com/aqlaboratory/openfold/blob/main/notebooks/OpenFold.ipynb)) takes an input sequence, which in this case was our potential EGFR binder. Following the sequence input, it initializes the model weight (AlphaFold/OpenFold) and mode parameters (monomer/multimer). For this application, OpenFold was selected as the model weight, and monomer was selected as the model mode. Monomer mode was chosen due to the goal of generating a structure from a single sequence. The model was also set to relax the prediction using AMBER, which will be explained further later. 

![DE8F30A5-E20C-4616-922F-69D5DA76D34E_4_5005_c](https://github.com/user-attachments/assets/60805dd7-e5e3-43c3-bc6d-ebca1b677c17)
_Fig. 4: Initialization of parameters in OpenFold Notebook_

Following the initialization of model parameters, the next portion of the code preprocesses the input sequence if necessary. 

``` ## Code Block
# Remove all whitespaces, tabs and end lines; upper-case
input_sequence = input_sequence.translate(str.maketrans('', '', ' \n\t')).upper()
aatypes = set('ACDEFGHIKLMNPQRSTVWY')  # 20 standard aatypes
allowed_chars = aatypes.union({':'})
if not set(input_sequence).issubset(allowed_chars):
  raise Exception(f'Input sequence contains non-amino acid letters: {set(input_sequence) - allowed_chars}. OpenFold only supports 20 standard amino acids as inputs.')

if ':' in input_sequence and weight_set != 'AlphaFold':
  raise ValueError('Input sequence is a multimer, must select Alphafold weight set')

``` 
 The next cell of the notebook imports the necessary software for running OpenFold into Colab. This saves the user from needing to create an external environment via conda on their device. It installs Mambaforge for managing Python environments and disables automatic updates to prevent unexpected dependency changes. It then imports bioinformatics tools that are used later on in the code (Kalign2, HHSuite, OpenMM, pdbfixer, Biopython) and installs python libraries (torch, ml_collections, py3Dmol, modelcif). Finally, it creates a temporary RAM-based filesystem to speed up operations (e.g., database access). and downloads OpenFold. 

 ``` ## Code
# Import third-party software


import os, time
from IPython.utils import io
from sys import version_info
import subprocess

python_version = f"{version_info.major}.{version_info.minor}"


os.system("wget -qnc https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh")
os.system("bash Mambaforge-Linux-x86_64.sh -bfp /usr/local")
os.system("mamba config --set auto_update_conda false")
os.system(f"mamba install -y -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 openmm=7.7.0 python={python_version} pdbfixer biopython=1.83")
os.system("pip install -q torch ml_collections py3Dmol modelcif")

try:
  with io.capture_output() as captured:

    # Create a ramdisk to store a database chunk to make Jackhmmer run fast.
    %shell sudo apt install --quiet --yes hmmer
    %shell sudo mkdir -m 777 --parents /tmp/ramdisk
    %shell sudo mount -t tmpfs -o size=9G ramdisk /tmp/ramdisk

    %shell wget -q -P /content \
      https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

    %shell mkdir -p /content/openfold/openfold/resources

    commit = "3bec3e9b2d1e8bdb83887899102eff7d42dc2ba9"
    os.system(f"pip install -q git+https://github.com/aqlaboratory/openfold.git@{commit}")

    os.system(f"cp -f -p /content/stereo_chemical_props.txt /usr/local/lib/python{python_version}/site-packages/openfold/resources/")

except subprocess.CalledProcessError as captured:
  print(captured)
```
The next cell of this notebook was also an initialization cell. It accessed and installed model weights from aqlaboratory GitHub - code can run with either AlphaFold or OpenFold configurations. It also accessed the Amazon Web Service (AWS) registry, which holds data for OpenFold.

```
# Download model weights

# Define constants
GIT_REPO='https://github.com/aqlaboratory/openfold'
ALPHAFOLD_PARAM_SOURCE_URL = 'https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar'
OPENFOLD_PARAMS_DIR = './openfold/openfold/resources/openfold_params'
ALPHAFOLD_PARAMS_DIR = './openfold/openfold/resources/params'
ALPHAFOLD_PARAMS_PATH = os.path.join(
  ALPHAFOLD_PARAMS_DIR, os.path.basename(ALPHAFOLD_PARAM_SOURCE_URL)
)
```
The final cell in the notebook relating to model initialization installed the rest of the necessary Python packages and imported necessary OpenFold modules. 








