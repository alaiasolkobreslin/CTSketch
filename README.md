# CTSketch

CTSketch is a neurosymbolic framework that uses decomposed symbolic programs and their sketched summaries to perform efficient inference.


## Install

CTSketch is compatible with Python 3.XX.XX. Run the following:
1. Install the dependencies inside a new virtual environment: `bash setup.sh`
2. Activate the conda environment: `conda activate CTSketch`
3. (Optional) Install package for GPT experiments: pip install openai

Datasets
- Leaf Identification: download the [leaf dataset](https://drive.google.com/file/d/1A9399fqTk3cR8eaRWCByCuh0_85D1JQc/view?usp=share_link) and place it under data/leaf_11.

- Scene Recognition: download the [scene dataset](https://drive.google.com/file/d/1ICXMkwP4gWzcC4My_UWALpXaAoRIiSTt/view?usp=share_link) and place it under data/scene.

- Hand-written Formula: download the [hwf dataset](https://drive.google.com/file/d/1VW--BO_CSxzB9C7-ZpE3_hrZbXDqlMU-/view?usp=share_link) and place it under data/hwf.

- Sudoku Solving: download the [SatNet dataset](https://powei.tw/sudoku.zip), unzip the data, and place features.pt, features_img.pt, labels.pt, and perm.pt under data/original_data.

- Visual Sudoku: download the [Visudo dataset](todo), unzip the data

## Experiments

## Acknowledgements