# CTSketch

CTSketch is a neurosymbolic framework that uses decomposed symbolic programs and their sketched summaries to perform efficient inference.


## Install

CTSketch is compatible with Python 3.XX.XX. Run the following:
1. Install the dependencies inside a new virtual environment: `bash setup.sh`
2. Activate the conda environment: `conda activate CTSketch`
3. (Optional) Install package for GPT experiments: pip install openai

    For the Sudoku Solving experimnet, you also need to install [prolog](https://www.swi-prolog.org).

The code for running experiment for task {task_name} is at `{method_name}_{task_name}.py` file inside the {task_name} folder. 

## Datasets

- Leaf Identification: download the [leaf dataset](https://drive.google.com/file/d/1IQIZfvx-OFrR7p4nI_H7gbtzTZkEmXFo/view?usp=share_link) and place it under data/leaf_11.

- Scene Recognition: download the [scene dataset](https://drive.google.com/file/d/1OqGhHTPycpi16jqGW4yctH1hQ9-Wm-V5/view?usp=share_link) and place it under data/scene.

- Hand-written Formula: download the [hwf dataset](https://drive.google.com/file/d/1zlzWkZmn8zQHqB_PNKZORPW0YBuU5GOp/view?usp=share_link) and place it under data/hwf.

- Sudoku Solving: download the [SatNet dataset](https://powei.tw/sudoku.zip), unzip the data, and place features.pt, features_img.pt, labels.pt, and perm.pt under data/original_data.

- Visual Sudoku: download the [4x4](https://drive.google.com/file/d/1trNLPn3Yei2u4ak9eHjpN07_Obxelnll/view?usp=share_link) or [9x9](https://drive.google.com/file/d/129rL0H_3RCB_f39YU8BM5vGt0Gh5_G4F/view?usp=share_link) Sudoku boards, unzip the data and place the folder under data/ViSudo-PC.

## Experiments

## Baselines

- To run Scallop, follow the instructions [here](https://www.scallop-lang.org/download.html) and install `scallopy`

- To run A-NeSI, follow the instructions [here](anesi/readme.md)

- To run DeepSoftLog, run: follow the instructions [here](deepsoftlog/readme.md)


## Acknowledgements

[tt_sketch](https://github.com/RikVoorhaar/tt-sketch/tree/main) 

[Visudo repository](https://github.com/linqs/visual-sudoku-puzzle-classification) to generate the dataset

