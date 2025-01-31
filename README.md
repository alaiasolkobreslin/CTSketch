# CTSketch

CTSketch is a neurosymbolic framework that uses decomposed symbolic programs and their sketched summaries to perform efficient inference.


## Install

CTSketch is compatible with Python 3.10. Run the following:
1. Create a new virtual environment: `conda create --name CTSketch python==3.10`
2. Activate the conda environment: `conda activate CTSketch`
3. Install the dependencies:  `pip install -r requirements.txt`
4. (Optional) Install package for GPT experiments: `pip install openai`

    For Sudoku Solving, you also need to install [prolog](https://www.swi-prolog.org).

The code for running experiments for task {task_name} is at `{method_name}_{task_name}.py` file inside the {task_name} folder. 

## Datasets

- Leaf Identification: download the [leaf dataset](https://drive.google.com/file/d/1IQIZfvx-OFrR7p4nI_H7gbtzTZkEmXFo/view?usp=share_link) and place it under data/leaf_11.

- Scene Recognition: download the [scene dataset](https://drive.google.com/file/d/1OqGhHTPycpi16jqGW4yctH1hQ9-Wm-V5/view?usp=share_link) and place it under data/scene.

- Hand-written Formula: download the [hwf dataset](https://drive.google.com/file/d/1zlzWkZmn8zQHqB_PNKZORPW0YBuU5GOp/view?usp=share_link) and place it under data/hwf.

- Sudoku Solving: download the [SatNet dataset](https://powei.tw/sudoku.zip), unzip the data, and place features.pt, features_img.pt, labels.pt, and perm.pt under data/original_data.

- Visual Sudoku: download the [4x4](https://drive.google.com/file/d/1trNLPn3Yei2u4ak9eHjpN07_Obxelnll/view?usp=share_link) and [9x9](https://drive.google.com/file/d/129rL0H_3RCB_f39YU8BM5vGt0Gh5_G4F/view?usp=share_link) Sudoku boards, unzip the data, and place the folders under data/ViSudo-PC.

## Baselines

- To run [Scallop](https://www.scallop-lang.org), follow the instructions [here](https://www.scallop-lang.org/download.html) and install `scallopy`.

- To run [A-NeSI](https://github.com/HEmile/a-nesi), follow the instructions [here](anesi/readme.md).

- To run [DeepSoftLog](https://github.com/jjcmoon/DeepSoftLog), run: follow the instructions [here](deepsoftlog/readme.md).

There are no additional steps required for running [IndeCateR](https://github.com/ML-KULeuven/catlog) and [ISED](https://github.com/alaiasolkobreslin/ISED)


## Acknowledgements

We adapt the code for tensor-train decomposition from [tt_sketch](https://github.com/RikVoorhaar/tt-sketch/tree/main).

We use the datasets from:
- Leaf: [A Data Repository of Leaf Images: Practice towards Plant Conversarion with Plant Pathology](https://ieeexplore.ieee.org/document/9036158) 

- Scene: [Multi-Illumination Dataset](https://projects.csail.mit.edu/illumination/databrowser/index-by-type.html#)

- HWF: [Closed Loop Neural-Symbolic Learning via Integrating Neural Perception, Grammar Parsing, and Symbolic Reasoning](https://github.com/liqing-ustc/NGS)

- Sudoku Solving: [SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver](https://github.com/locuslab/SATNet)

- Visual Sudoku: [Visual Sudoku Puzzle Classification: A Suite of Collective Neuro-Symbolic Tasks](https://github.com/linqs/visual-sudoku-puzzle-classification)

