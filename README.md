# DHC-STGN: Dynamic Hierarchical Confidence Spatio-Temporal Graph Networks

Code for the paper: *"DHC-STGN: Dynamic Hierarchical Confidence Spatio-Temporal Graph Networks for Forecasting Solar Irradiance"* 

---

##  Repository Structure

- `Similarity_matrix.py` – Computes the dynamic hierarchical confidence graph based on spatial proximity, temporal similarity, and station reliability  
- `DHC-STGN1.py` – Main model training and evaluation script

---

##  Environment Setup

The project runs in a Conda virtual environment with the following key dependencies:

- **Python**: 3.11.14  
- **PyTorch**: 2.9.1 (CUDA 12.8)  
- **TorchVision**: 0.24.1  
- **TorchAudio**: 2.9.1  
- **NumPy**: 1.24.3  
- **Pandas**: 2.3.3  
- **Scikit-learn**: 1.8.0  
- **Matplotlib**: 3.10.8  
- **fastdtw**: 0.3.4  
- **NetworkX**: 3.6.1  
- **Munkres**: 1.1.4  

### Create Conda Environment

```bash
# Create environment
conda create -n dhc-stgn python=3.11
conda activate dhc-stgn

# Install core scientific stack
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia
conda install numpy pandas scikit-learn matplotlib networkx munkres -c conda-forge

# Install PyPI packages
pip install fastdtw

```
### Dataset

The solar irradiance dataset is publicly available from the Tibetan Plateau Scientific Data Center (Tang et al., 2013).
Citation

Tang, W., Yang, K., Qin, J., & Min, M. (2013). Development of a 50-year daily surface solar radiation dataset over China. Science China Earth Sciences, 56, 1555–1565.
https://doi.org/10.1007/s11430-012-4542-9
Data Format

After downloading, each station's data should be formatted as a CSV file with:

    Column 1: Station ID / name

    Column 2: Longitude (degrees)

    Column 3: Latitude (degrees)

    Column 4: Elevation (meters)

    Column 5 onwards: Daily solar irradiance (W/m²), one column per day in chronological order

Example:
text
Station_123, 98.5, 35.2, 320, 125.3, 130.8, 128.1, ...

Data Placement

Place the downloaded data files in the data/ directory before running the code.
### Quick Start
bash

# 1. Clone repository
git clone https://github.com/99bubi/DHC-STGN.git
cd DHC-STGN

# 2. Set up environment (see above)

# 3. Download dataset to data/ folder

# 4. Compute similarity matrix
python Similarity_matrix.py

# 5. Train model
python DHC-STGN1.py

### Output

    Trained model weights

    Prediction results with RMSE, MAE, and R²

### Contact

Fengjie Wang – bubiwang@mail.hnust.edu.cn
Central South University

