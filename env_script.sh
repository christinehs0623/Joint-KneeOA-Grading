conda create -n knee_mil python=3.10
conda activate knee_mil
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
conda install \
  scikit-learn pandas matplotlib opencv h5py pydicom tqdm \
  ipykernel ipywidgets jupyterlab -c conda-forge
pip install absl-py==2.2.2 grad-cam==1.5.5 ttach==0.0.3
pip install wandb
