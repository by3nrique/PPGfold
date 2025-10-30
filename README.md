# Manifold Learning Approaches for Characterizing Photoplethysmographic Signals
[![IEEE DOI](https://img.shields.io/badge/DOI-10.1109/TBME.2025.3625858-blue)](https://doi.org/10.1109/TBME.2025.3625858)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16736997.svg)](https://doi.org/10.5281/zenodo.16736997)


📄 **Published article:** [IEEE Transactions on Biomedical Engineering](https://doi.org/10.1109/TBME.2025.3625858)  
🧬 **PubMed entry:** [https://pubmed.ncbi.nlm.nih.gov/41144410/](https://pubmed.ncbi.nlm.nih.gov/41144410/)

This repository contains all code, preprocessing scripts, model configurations, and trained checkpoints associated with the study  
**"Manifold Learning Approaches for Characterizing Photoplethysmographic Signals"** by  

- Enrique Feito-Casares [![ORCID](https://img.shields.io/badge/ORCID-0009--0005--5068--3166-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0005-5068-3166)  
- Francisco M. Melgarejo-Meseguer [![ORCID](https://img.shields.io/badge/ORCID-0000--0001--6916--6082-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0001-6916-6082) 
- Alejandro Cobo Carbonero [![ORCID](https://img.shields.io/badge/ORCID-0009--0007--7967--6837-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0007-7967-6837) 
- Luis Baumela Molina [![ORCID](https://img.shields.io/badge/ORCID-0000--0001--6910--4359-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0001-6910-4359) 
- José-Luis Rojo-Álvarez [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--0426--8912-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0003-0426-8912)

<p align="center">
<img width="512" height="512" alt="imagen" src="https://github.com/user-attachments/assets/c3a6232f-bca7-478f-bda2-63a6ffe8e779" />
</p>

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Feito-Casares2025a,
  title = {Manifold {{Learning Approaches}} for {{Characterizing Photoplethysmographic Signals}}},
  author = {{Feito-Casares}, Enrique and {Melgarejo-Meseguer}, Francisco M and Cobo, Alejandro and Baumela, Luis and {Rojo-{\'A}lvarez}, Jos{\'e}-Luis},
  year = {2025},
  journal = {IEEE Transactions on Biomedical Engineering},
  pages = {1--14},
  issn = {0018-9294, 1558-2531},
  doi = {10.1109/TBME.2025.3625858},
  urldate = {2025-10-29},
  copyright = {https://creativecommons.org/licenses/by/4.0/legalcode}
}

@online{FeitoCasares2025b,
  title = {{Manifold Learning Approaches for Characterizing Photoplethysmographic Signals (Supplementary Code)}},
  author = {Feito Casares, Enrique and Melgarejo Meseguer, Francisco Manuel and Cobo Carbonero, Alejandro and Baumela Molina, Luis and {Rojo-{\'A}lvarez}, Jos{\'e} Luis},
  year = {2025},
  month = aug,
  doi = {10.5281/zenodo.16736998},
  howpublished = {Zenodo}
}
```
## Overview

The pipeline supports the extraction and analysis of low-dimensional embeddings of photoplethysmography (PPG) signals using advanced manifold learning methods, including:
- Fully Connected Neural Networks (FCNNs)
- Autoencoders (AEs)
- Uniform Manifold Approximation and Projection (UMAP)

## Repository Structure

```
PPGFold/
├── readme.md                    # This file
├── requirements.txt             # Python dependencies
├── Datasets/                    # Dataset storage directory
│   ├── BIDMC/                   # BIDMC dataset
│   ├── MIMIC_PERFORM/           # MIMIC-PERFORM dataset  
│   ├── UBFC/                    # UBFC dataset
│   └── WristPPG/                # Wrist PPG dataset
├── Experiments/                 # Main experimental code
│   ├── BIDMC.ipynb              # BIDMC dataset experiments
│   ├── MIMIC.ipynb              # MIMIC-PERFORM experiments
│   ├── UBFC.ipynb               # UBFC dataset experiments
│   ├── WristPPG.ipynb           # Wrist PPG experiments
│   ├── functions.py             # Core utility functions
│   ├── ppg_functions.py         # PPG-specific processing functions
│   ├── autoencoder_code.py      # Autoencoder implementation
│   ├── umap_code.py             # UMAP implementation
```

## Features

### Preprocessing Pipeline
- Signal resampling and filtering
- Spectral analysis and residual computation
- Window-based segmentation with configurable overlap
- Data splitting for train/validation/test sets

### Manifold Learning Methods
- **Autoencoders**: Deep neural networks for non-linear dimensionality reduction
- **UMAP**: Uniform Manifold Approximation and Projection for topology-preserving embeddings
- **FCNNs**: Fully connected networks for supervised embedding learning

### Downstream Tasks
- **Anomaly Detection**: Synthetic anomaly generation and detection
- **Classification**: Patient condition classification using embeddings
- **Clustering**: Unsupervised patient grouping
- **Visualization**: 2D/3D embedding visualization

## Datasets

The repository supports four major PPG datasets:

1. **BIDMC**: 53 patients, single condition classification
2. **MIMIC-PERFORM**: 200 patients, binary classification (abnormal/normal)
3. **UBFC**: rPPG with real/fake classification
4. **WristPPG**: Wearable device PPG signals

## Usage

### Running Experiments

Each dataset has its dedicated Jupyter notebook:

```bash
# BIDMC experiments
jupyter notebook Experiments/BIDMC.ipynb

# MIMIC-PERFORM experiments  
jupyter notebook Experiments/MIMIC.ipynb

# UBFC experiments
jupyter notebook Experiments/UBFC.ipynb

# Wrist PPG experiments
jupyter notebook Experiments/WristPPG.ipynb
```

### Key Parameters

**Data Processing:**
- `desired_fs`: Target sampling frequency (100 Hz)
- `window_size`: Segmentation window size (85 samples)
- `overlap_size`: Window overlap (24 samples)
- `ff`: Frequency band [0.67, 8] Hz

**Model Training:**
- `epochs`: Training epochs (500)
- `batch_size`: Mini-batch size (64)
- `learning_rate`: Optimizer learning rate (0.001)

### Example Usage

```python
# Load and preprocess data
from ppg_functions import *
X, spectrums, residuals = prepare_data(X_raw, y, fs, desired_fs, ff, nfft)

# Train autoencoder
best_model, config, errors, Htr, Hval, Hts = train_AE(
    folder, 0, Xtr, Otr, Ytr, Xval, Oval, Yval, 
    Xts, Ots, Yts, labels, experiment_name,
    architecture_options, activation_functions,
    learning_rates, batch_sizes, n_splits, epochs, force_train
)

# Train UMAP
best_umap, umap_config, errors, Htr_umap, Hval_umap, Hts_umap = train_umap(
    folder, 0, Xtr, Ytr, Xval, Yval, Xts, Yts,
    experiment_name, labels, n_neighbors, min_dist, 
    n_components, n_splits, force_train
)
```

## Reproducibility

### Environment
- Python 3.9.21+
- TensorFlow 2.x with GPU support
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for large datasets

### Seeds
All experiments use fixed random seeds:
```python
tf.random.set_seed(42)
np.random.seed(42)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work was supported by the Research Grants HERMES, LATENTIA, PCardioTrials, and ETHICFACE (PID2023-152331OA-I00, PID2022-140786NB-C31, and PID2022-140553OA-C42, and PID2022-137581OB-I00), funded by MICIU/AEI/10.13039/501100011033 and ERDF, FEDER / EU. Also supported by Rey Juan Carlos University, project HERMES 2024/00004/00, by the Programa Propio of UPM, by the CyberFold project, funded by the European Union through the NextGenerationEU instrument (Recovery, Transformation, and Resilience Plan), and managed by Instituto Nacional de Ciberseguridad de España (INCIBE), under reference number ETD202300129, and a grant from Comunidad de Madrid to the Madrid ELLIS Unit.  

<p align="center">
<img width="4678" height="410" alt="imagen" src="https://github.com/user-attachments/assets/6cd4906a-a957-41d9-974c-e26244b87fb0" />
</p>

With the collaboration of
<p align="center">
<img width="197" height="82" alt="imagen" src="https://github.com/user-attachments/assets/a35dc242-ddf6-4307-a03a-88ad3ed741af" />
</p>

