# Traffic-safety-analyses-modeling

===========================================================================

In this projector we modeling the traffic conflicts based on the trajectory from sharing spaces under intersection. 

## Description

introduce the 

## Contents

### Table of Examples

- **The process of conflict modeling based on the trajectory under Mixed traffic environment**

  - [x] [Step_1: Formatting the original trajectory data](https://drive.google.com/file/d/1uPAuJ7qi1uWRLyAdcDiNVn8jsiBAyzfo/view?usp=share_link)
  - [x] [Step_2: Clearing the formation date ]
  - [x] [Step_3: Calucating the surgery safety indicators]
  - [x] [Step_4: Extracting the conflict events]
  - [x] [Step_5: Defining the effect variable]
  - [x] [Step_6: Establishing the model of conflict procession]
  <br>


### 1.Formatting the original trajectory data

Vehicle trajectory record the procession of vehicle motion that contain the spatio-temporal information. Due to the trajectory from different sensors with different formats, we should standardizate the output data from the different sensors befor the beginning of processing. In this project，we define the trajectory format as the Table 1. The orginal trajectory of this project extracted by Yolov7+Deepsort, MaskRcNN and Datafromsky. You can get the trajectory from Datafromsky by the [link](https://drive.google.com/file/d/1lQuGvIBc-apCCxEdDFJZZ7eMZ9YtxTnB/view?usp=share_link) in Google drive. 

The code for formatting the original trajectory use the [code]([Traffic-safty-analyses-modeling/code/data_process.ipynb](https://github.com/YANzhangcun/Traffic-safty-analyses-modeling/blob/master/code/data_process.ipynb)). with this code we can convert the original trajectory dataset into the new dataset with the format of Table 1.

#### Table 1 The format of the trajectory 
|vehicle_id|frame_id|vehicle_type|world_x|world_y|speed_x|speed_y|acc_x|acc_y|jerk_x|jerk_y|
|----------|--------|------------|-------|-------|-------|-------|-----|-----|------|------|
|     1    |  0     |     car    |-------|-------|-------|-------|-----|-----|------|------|
|     1    |  1     |     car    |-------|-------|-------|-------|-----|-----|------|------|
|     1    |  2     |     car    |-------|-------|-------|-------|-----|-----|------|------|
|    ...   |--------|     car    |-------|-------|-------|-------|-----|-----|------|------|
|    ...   |  n     |     car    |-------|-------|-------|-------|-----|-----|------|------|
|     2    |  1     |     bus    |-------|-------|-------|-------|-----|-----|------|------|



#### 2.Clearing the formation date
In the Step_2 we need remove the noise and delete the incurable trajectories. the Wavelet filter was employed to process the noise trajectory with the [code](). The incurable trajectory were delete by the manual. 


##### [Example 1]

This example is from the following paper:

> - Qibin Zhao, Liqing Zhang, Andrzej Cichocki (2015). [Bayesian CP factorization of incomplete tensors with automatic rank determination](https://doi.org/10.1109/TPAMI.2015.2392756). IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(9): 1751-1763.

<a href="https://github.com/xinychen/awesome-latex-drawing/blob/master/BayesNet/BCPF.tex">
<img src="BayesNet/BCPF.png" alt="drawing" width="280" align="right"/>
</a>
which shows the Bayesian network of Bayesian CP factorization (BCPF) model. To draw this Bayesian network example, there are some preliminaries to follow:

<br>

- **`preamble` codes**:
  1. define the `documentclass` as `standalone`, e.g., `\documentclass[border = 0.1cm]{standalone}` with 0.1cm border,
  2. use the package `tikz`, i.e., `\usepackage{tikz}`, and use `tikz` library like `\usetikzlibrary{bayesnet}` which is an important tool for drawing Bayesian networks and directed factor graphs,
  3. set the `tikz` style by using the `\tikzstyle{}` command,
  4. use math equation environments including `\usepackage{amsfonts, amsmath, amssymb}`.
- **`body` codes**:
  1. use `\begin{tikzpicture} \end{tikzpicture}` to start drawing,
  2. use `\node` to define nodes and text boxes in the Bayesian network,
  3. use `\path` to define arrows in the Bayesian network,
  4. use `\plate` to define plates in the Bayesian network.

> Please click on the image and check out the source code.

<br>




- Open [BGCP.tex](https://github.com/xinychen/awesome-latex-drawing/blob/master/BayesNet/BGCP.tex) in your overleaf project, then you will see the following pictures about BGCP (Bayesian Gaussian CP decomposition) model as a Bayesian network and a directed factor graph:

<p align="center">
<img align="middle" src="BayesNet/BGCP.png" width="700" />
</p>

<br>
