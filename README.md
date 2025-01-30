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

- Leaf Identification: download the [leaf dataset](https://drive.google.com/file/d/1A9399fqTk3cR8eaRWCByCuh0_85D1JQc/view?usp=share_link) and place it under data/leaf_11.

- Scene Recognition: download the [scene dataset](https://drive.google.com/file/d/1ICXMkwP4gWzcC4My_UWALpXaAoRIiSTt/view?usp=share_link) and place it under data/scene.

- Hand-written Formula: download the [hwf dataset](https://drive.google.com/file/d/1VW--BO_CSxzB9C7-ZpE3_hrZbXDqlMU-/view?usp=share_link) and place it under data/hwf.

- Sudoku Solving: download the [SatNet dataset](https://powei.tw/sudoku.zip), unzip the data, and place features.pt, features_img.pt, labels.pt, and perm.pt under data/original_data.

- Visual Sudoku: download the [4x4](https://drive.google.com/file/d/1E-_37z4eSGdsUqlh3BZW2m6r8NLkvbNi/view?usp=share_link) or [9x9](https://drive.google.com/file/d/1l-GjaWUFZxFBKKwbSndWQNBMRRmqpS3c/view?usp=share_link) Sudoku boards, unzip the data and place the folder under data/ViSudo-PC.

## Experiments

## Baselines

- To run Scallop, follow the instructions [here](https://www.scallop-lang.org/download.html) and install `scallopy`

- To run A-NeSI, follow the instructions [here](anesi/readme.md)

- To run DeepSoftLog, run: follow the instructions [here](deepsoftlog/readme.md)


## Acknowledgements

[tt_sketch](https://github.com/RikVoorhaar/tt-sketch/tree/main) 

[Visudo repository](https://github.com/linqs/visual-sudoku-puzzle-classification) to generate the dataset

