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

<br>

#### 2.Clearing the formation date
In the Step_2 we need remove the noise and delete the incurable trajectories. the Wavelet filter was employed to process the noise trajectory with the [code](). The incurable trajectory were delete by the manual. 
<a  >
   
</a>
##### [Example 1]







<br>



