Micro-driving behavior modeling and analyzing based on the High-resolution traffic video in the Intersection

The implementation of computer vision algorithms has being bombed by the breakthroughs in computing power especially in the intelligent transportation systems. But there some ramparts between computer vision algorithm and traditionally traffic theory. Here, I want to show whole procedures for Modeling the interaction behavior between motorized and Non-Motorized  vehicles based the High-resolution traffic video from roadside view. To summarize those procedures and provide cleaning tutorials for the ordinal student, Some of the details background information was omitted and just saved the main steps. 

**First**, we should prepare the basic information of object scenarios that contains the unobstructed videos, the Geometric size of the object intersection, and the signal plans. These information will help you to convert picture information to real world. For instance, the fellow figures show **the real world coordination system** and the some details about basic information.  You can establish the real world coordinator system basic on geometric information from the high-resolution measures experiment. On the other way, you can get less five marked points of the intersections by the GPS location system app on mobile phones. 

| <img src="E:\CodeResource\000_Traffic_conflict_risk_analysis\Data_clearning\Data_set\changjidong-moyu.jpg" alt="changjidong-moyu" style="zoom:12%;" /> | <img src="E:\CodeResource\000_Traffic_conflict_risk_analysis\Data_clearning\Data_set\image-20231102193939965.png" alt="image-20231102193939965" style="zoom:25%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|        a. The real coordinator of object intersection        |                  b. The method of measures                   |

​					**Figure.1. The methodology of estimating the real world coordinator**

**Second**, Make the datasets and train the detection model, you should label the object which you chosen based on your research topic(such as: Car, Bus, truck, Fright, president, electrical bicycles and bicycles so on!) in you video. Some software can help you finish this task effectively. such as, [ImageLabel](https://create.roblox.com/docs/reference/engine/classes/ImageLabel). And then, chose suite computer vision algorithm as the detection model, Here, I introduce **Yolov8** algorithm.

<img src="E:\Academic\project\Drivingbehaviormoding\study procedure\Figure\trajectory tracker.jpg" alt="trajectory tracker " style="zoom:70%;" />

​                             **Figure.2. The framework of object detection and trajectory tracking**

**Third**, Detection the object and Tracking the Trajectories. Now, we can connect the detection model with tracking model. The better tracking model is the evolution of Deepsort algorithm which is employed in our framework. 

|                       Object Detection                       |                       object Tracking                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="E:\CodeResource\000_Traffic_conflict_risk_analysis\Data_clearning\Data_set\目标检测.jpg" alt="目标检测" style="zoom:10%;" /> | <img src="E:\CodeResource\000_Traffic_conflict_risk_analysis\Data_clearning\Data_set\轨迹追踪-1699303742907-4.png" alt="轨迹追踪" style="zoom:45%;" /> |

​                                        **Figure 3. The processor of objects and trajectories tracking** 

**Fourth**, Reconstruction of the orginal trajectories. Due to the orginal trajectory is saving in different formats, I should be reorganize before next analysis step. There are four procedures should be implemented: reformation, filter and reconstruction. The details of this procedure shown as fellow.

*The code of reformation:*

```python
# Basic packages 
import math
import pandas as pd
import numpy as np
import os

# define new format of saving data.
# The basic information: vehicle_ID ，vehicle_type,  x , y, speed, tan_acc, lat_acc, time
def Data_format_switch(data_path):
    ult_res = {}
    data1 = pd.read_csv(data_path) 
    veh_ID_list = []
    veh_type_list = []
    veh_x_list = []
    veh_y_list = []
    veh_speed_list = []
    veh_tan_acc_list = []
    veh_lat_acc_list = []
    veh_time_list = []
    vehicle_angle_list = []
    for i in range(len(data1)):
        str_res = data1.iloc[i][0] 
        ls = str_res.split(";") 
        vehicle_ID = ls[0]
        vehicle_type = ls[1]
        vehicle_trajectory = ls[10:]
        vehicle_trajectory = [vehicle_trajectory[j:j+7] for j in range(0,len(vehicle_trajectory)-1,7)] 
        vehicle_trajectory = np.array(vehicle_trajectory) 
        vehicle_trajectory_T = vehicle_trajectory.T 
        ls_x = vehicle_trajectory_T[0]
        ls_x = [np.float64(m) for m in ls_x] 
        ls_y = vehicle_trajectory_T[1]
        ls_y = [np.float64(m) for m in ls_y]
        ls_speed = vehicle_trajectory_T[2]
        ls_speed = [np.float64(m) for m in ls_speed]
        ls_tan_acc = vehicle_trajectory_T[3]
        ls_tan_acc = [np.float64(m) for m in ls_tan_acc]
        ls_lat_acc = vehicle_trajectory_T[4]
        ls_lat_acc = [np.float64(m) for m in ls_lat_acc]
        ls_time = vehicle_trajectory_T[5]
        ls_time = [np.float64(m) for m in ls_time]
        vehicle_angle = vehicle_trajectory_T[6]
        vehicle_angle = [np.float64(m) for m in vehicle_angle]
        vehicle_ID_ls = [vehicle_ID] * len(ls_time)
        vehicle_type_ls = [vehicle_type]*len(ls_time)
        veh_ID_list= veh_ID_list + vehicle_ID_ls
        veh_type_list = veh_type_list + vehicle_type_ls
        veh_x_list = veh_x_list + ls_x
        veh_y_list = veh_y_list + ls_y
        veh_speed_list = veh_speed_list + ls_speed
        veh_tan_acc_list = veh_tan_acc_list + ls_tan_acc
        veh_lat_acc_list = veh_lat_acc_list + ls_lat_acc
        veh_time_list = veh_time_list + ls_time
        vehicle_angle_list = vehicle_angle_list + vehicle_angle
    ult_res['vehicle_id'] = veh_ID_list
    ult_res['vehicle_type'] = veh_type_list
    ult_res['frame_time'] = veh_time_list
    ult_res['world_x'] = veh_x_list
    ult_res['world_y'] = veh_y_list
    ult_res['vehicle_speed'] = veh_speed_list
    ult_res['vehicle_tan_acc'] = veh_tan_acc_list
    ult_res['vehicle_lat_acc'] = veh_lat_acc_list
    ult_res['Angle'] = vehicle_angle_list
    ult_res = pd.DataFrame(ult_res)
    return ult_res

def File_procession(Input_file_path,Output_file_path):
    "This function will process the csv in the file path"
    files1 = os.listdir(Input_file_path)
    for i in range(len(files1)):
        work_file = Input_file_path +files1[i]
        print(work_file)
        save_path = Output_file_path +'/'+ files1[i]
        Trajectory_denoise = Data_format_switch(work_file)
        Trajectory_denoise.to_csv(save_path, index=False, header=True)
    return Trajectory_denoise

# The pathfile of the data 
input_path = r'E:/CodeResource/000_Traffic_conflict_risk_analysis/'
output_path = r'E:/CodeResource/000_Traffic_conflict_risk_analysis/Data_clearning'
Trajectory_denoise = File_procession(input_path,output_path)
```

*The code of denoise, in the first step, we should calculate the variable of the vehicle motion, then denoise the trajectory.*

```python
# calculate the kinetic parameter
def XY(groundtraj,caompartraj):
    g_World_x = np.array(groundtraj['world_x'].astype(float))
    g_World_y = np.array(groundtraj['world_y'].astype(float))
    com_World_x = np.array(caompartraj['world_x'].astype(float))
    com_World_y = np.array(caompartraj['world_y'].astype(float))
    return g_World_x,g_World_y,com_World_x,com_World_y

def Velocity(trajdata):
    """recalculate the velocity, the velocity contain the X_velocity and Y-velocity and the speed"""
    """定义初速度为0"""
    len_x = trajdata.shape[0]
    wordld_x = np.array(trajdata.world_x)
    wordld_y = np.array(trajdata.world_y)
    velocity_x = (wordld_x[1:len_x]-wordld_x[0:len_x-1])/0.04
    velocity_y = (wordld_y[1:len_x]-wordld_y[0:len_x-1])/0.04
    velocity_x = np.insert(velocity_x,0,0)
    velocity_y = np.insert(velocity_y,0,0)
    return velocity_x,velocity_y

def Accelection(trajdata):
    """定结束时刻的加速度为0"""
#     print(trajdata)
    len_x = trajdata.shape[0]
    velocity_x = np.array(trajdata.speed_x)
    velocity_y = np.array(trajdata.speed_y)
    accelection_x = (velocity_x[1:len_x]-velocity_x[0:len_x-1])/0.04
    accelection_y = (velocity_y[1:len_x]-velocity_y[0:len_x-1])/0.04
    accelection_x = np.insert(accelection_x,0,0)
    accelection_y = np.insert(accelection_y,0,0)
#     print(accelection_x)
    accelection_x[1] = 0
    accelection_y[1] = 0
    return accelection_x,accelection_y

def Aclculate_Jerk(trajdata):
    """计算急动度"""
    len_x = trajdata.acc_x.shape[0]
    acc_x = np.array(trajdata.acc_x)
    acc_y = np.array(trajdata.acc_y)
    JJerk_x = (acc_x[1:len_x] - acc_x[0:(len_x-1)])/0.04
    JJerk_y = (acc_y[1:len_x] - acc_y[0:(len_x-1)])/0.04
    JJerk_x = np.insert(JJerk_x,0,0)
    JJerk_y = np.insert(JJerk_y,0,0)
    JJerk_x[2] = 0
    JJerk_y[2] = 0
    return JJerk_x,JJerk_y

def Angle(trajectorydata):
    "calculate the Angle of vehilce"
    Angle_veh = np.array(trajectorydata.Angle)
    return Angle_veh
```

```python
# wavelet algorithm for denoising
import numpy as np
import pywt
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt

# Denoise the trajectory
def wavelet_reduce_noise(input_data_path,output_data_path):
    'if you use this code you should format the trajectory at first.'
    Traje_clearn = pd.read_csv(input_data_path)
    traj_data_df = pd.DataFrame(Traje_clearn)
    df = traj_data_df
    vehid = pd.unique(df.vehicle_id)
    Wavelet_traj = df[['vehicle_id','frame_time','vehicle_type']]
    Wt = pd.DataFrame(Wavelet_traj)
    Wt['world_x']=''
    Wt['world_y']=''
    Wt['speed_x']=''
    Wt['speed_y']=''
    Wt['acc_x']=''
    Wt['acc_y']=''
    Wt['Jerk_x']=''
    Wt['Jerk_y']=''
    Wt['Angle'] =''
    for id in range(0,len(vehid),1):
        veh_id = vehid[id]
        B =[]
        V =[]
        ACC = []
        Jerk=[]
        Angle_veh = []
        traj_data = df[df.vehicle_id==veh_id]
        if len(traj_data)>=20:
            Frame_id = pd.unique(traj_data.frame_time)
            TRAJ_world_x = traj_data['world_x']
            TRAJ_world_y = traj_data['world_y']
            min_row = traj_data.loc[traj_data['frame_time']== min(Frame_id),].index[0]
            max_row = traj_data.loc[traj_data['frame_time']== max(Frame_id),].index[0]
            AA = traj_data.iloc[min_row:max_row+1,[3,4]]
            A = df.iloc[min_row:max_row+1,[3,4]]
            # x_denoise = denoise_wavelet(TRAJ_world_x, method='BayesShrink', mode='soft', wavelet_levels=3, wavelet='sym8',
            #                             rescale_sigma='True')
            W_X = pd.array(denoise_wavelet(TRAJ_world_x, method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            B.append(W_X)
            W_Y = pd.array(denoise_wavelet(TRAJ_world_y, method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            B.append(W_Y)
            BB = pd.DataFrame(B)
            BBB = np.transpose(BB)
            if A.shape[0]<BBB.shape[0]:
                CB = BBB.iloc[0:A.shape[0],[0,1]]
            else:
                CB = BBB.iloc[0:len(BBB),[0,1]]
            Wt.iloc[min_row:max_row+1,[3,4]] = CB
            'Calculate the speed with wordx and word y'
            WTraj = Wt.iloc[min_row:max_row+1,0:5]
            speed_x,speed_y = Velocity(WTraj)
            "Denoise the speed of the vehicle"
            speed_x = pd.Series(speed_x)
            speed_x = speed_x.astype(np.float64)
            W_speed_x = pd.array(denoise_wavelet(speed_x, method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            V.append(W_speed_x)
            speed_y = pd.Series(speed_y)
            speed_y = speed_y.astype(np.float64)
            W_speed_Y = pd.array(denoise_wavelet(speed_y, method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            V.append(W_speed_Y)
            VV = pd.DataFrame(V)
            VVV = np.transpose(VV)
            if A.shape[0]<VVV.shape[0]:
                CBB = VVV.iloc[0:A.shape[0],[0,1]]
            else:
                CBB = VVV.iloc[0:len(VVV),[0,1]]
            Wt.iloc[min_row:max_row+1,[5,6]] = CBB
            # 计算加速度
            WWTraj = Wt.iloc[min_row:max_row+1,0:7]
            acc_x,acc_y = Accelection(WWTraj)
            acc_x = pd.Series(acc_x)
            acc_x = acc_x.astype(np.float64)
            W_acc_x = pd.array(denoise_wavelet(acc_x,method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            ACC.append(W_acc_x)
            acc_y = pd.Series(acc_y)
            acc_y = acc_y.astype(np.float64)
            W_acc_y = pd.array(denoise_wavelet(acc_y,method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            ACC.append(W_acc_y)
            ACCC = pd.DataFrame(ACC)
            AACCC = np.transpose(ACCC)
            if A.shape[0]<AACCC.shape[0]:
                CBBB = AACCC.iloc[0:A.shape[0],[0,1]]
            else:
                CBBB = AACCC.iloc[0:len(AACCC),[0,1]]
            Wt.iloc[min_row:max_row+1,[7,8]] = CBBB
            'calculate the jerk value'
            WWWTTraj = Wt.iloc[min_row:max_row+1,0:9]
            WWWTTTraj = pd.DataFrame(WWWTTraj)
            jerxx,jeryy = Aclculate_Jerk(WWWTTTraj)
            jerxx = pd.Series(jerxx)
            jerxx = jerxx.astype(np.float64)
            W_jerk_x = pd.array(denoise_wavelet(jerxx,method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            Jerk.append(W_jerk_x)
            jeryy = pd.Series(jeryy)
            jeryy = jeryy.astype(np.float64)
            W_jerk_y = pd.array(denoise_wavelet(jeryy,method='BayesShrink', mode='soft', wavelet_levels=5, wavelet='sym8',rescale_sigma='True'))
            Jerk.append(W_jerk_y)
            Jerkk = pd.DataFrame(Jerk)
            JJerkk = np.transpose(Jerkk)
            if A.shape[0]<VVV.shape[0]:
                CBBBB = JJerkk.iloc[0:A.shape[0],[0,1]]
            else:
                CBBBB = JJerkk.iloc[0:len(JJerkk),[0,1]]
            Wt.iloc[min_row:max_row+1,[9,10]] = CBBBB
            "Add the Angle of the vehicle"
            Angle_vehicle_traj = Angle(traj_data)
            Angle_veh.append(Angle_vehicle_traj)
            Angle_veh = pd.DataFrame(Angle_veh)
            Anngle_veh = np.transpose(Angle_veh)
            Wt.iloc[min_row:max_row+1,[11]] = Anngle_veh
        #     print(Wt.iloc[min_row:max_row+1,[9,10]])
    Wt.to_csv(output_data_path,index=False, header=True)
    return Wt
```

```python
# processing the file one by one
def File_procession(Input_file_path,Output_file_path):
    "This function will process the csv in the file path"
    files1 = os.listdir(Input_file_path)
    for i in range(len(files1)):
        work_file = Input_file_path +files1[i]
        print(work_file)
        save_path = Output_file_path +'/'+ files1[i]
        Trajectory_denoise = wavelet_reduce_noise(work_file,save_path)
    return Trajectory_denoise

# the dataset files about input path and the output data path
Input_file_path = r'D:/dataset/Changji-moyu/output_Trajectory/Reorginazation/step1/'
Output_file_path = r'D:/dataset/Changji-moyu/output_Trajectory/Reorginazation/step2'
Trajectory_denoise = File_procession(Input_file_path,Output_file_path)
```

**Show the result of trajectories, compare the variable related to the speed and the acceleration **

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def SpeedHistmap(data1,data2):
  plt.hist(data1, bins=20, density=True, alpha=0.4, label='Speed_x')
  plt.hist(data2, bins=20, density=True, alpha=0.4, label='Speed_y')
  plt.xlabel('speed(km/h)',fontsize=12)
  plt.xticks(fontsize=12,rotation=0)
  plt.yticks(fontsize=12)
  plt.ylabel('Frequency',fontsize=12)
  plt.legend(loc="upper right",fontsize=12)   #设置图例字体大小
  plt.tight_layout()
  plt.grid()
  plt.show()
    
def AccHistmap(data1,data2):
  plt.hist(data1, bins=20,color='red',density=True, alpha=0.4, label='acc_x')
  plt.hist(data2, bins=20,color='blue', density=True, alpha=0.4, label='acc_y')
  plt.xlabel('acc(m/s^2)',fontsize=12)
  plt.xticks(fontsize=12,rotation=0)
  plt.yticks(fontsize=12)
  plt.ylabel('Frequency',fontsize=12)
  plt.legend(loc="upper right",fontsize=12)   #设置图例字体大小
  plt.tight_layout()
  plt.grid()
  plt.show()
```

```python
def Compare_montion(Trajectory_No_filter,Trajectory_filter,veh_id):
    'compare the denoise effect'
    Traj_no_veh = pd.read_csv(Trajectory_No_filter)
    Traj_no_veh = pd.DataFrame(Traj_no_veh)
    Traj_filter_veh = pd.read_csv(Trajectory_filter)
    Traj_filter_veh = pd.DataFrame(Traj_filter_veh)
    Traj_nf_vehid = Traj_no_veh[Traj_no_veh['vehicle_id']==veh_id]
    Traj_f_vehid = Traj_filter_veh[Traj_filter_veh['vehicle_id']==veh_id]
    # plt.scatter(Traj_nf_vehid['frame_time'],Traj_nf_vehid['speed_x'])
    plt.scatter(Traj_nf_vehid['frame_time'], Traj_nf_vehid['speed_y'])
    # plt.scatter(Traj_f_vehid['frame_time'], Traj_f_vehid['speed_x'])
    plt.scatter(Traj_f_vehid['frame_time'], Traj_f_vehid['speed_y'])
    plt.show()
```

**Firth, extracting the conflict event chain! **The next step is calculate the indictors of traffic conflicts, such as the TTC, PET, Delta-V and the risk field in the real time data. Here, summary the key step of processing. 

The idea of calculating the inductor traffic conflict between different vehicles. 



<img src="E:\CodeResource\000_Traffic_conflict_risk_analysis\Data_clearning\Data_set\conflict_calculate processing.png" alt="conflict_calculate processing" style="zoom:25%;" />

​							**Figure. 4. Calculate the TTC of two vehicles**

```python
 # calculate the based element of the risk inductor
 def Speed_Caluca(Traj_OD,time_gap):
    # 计算速度值
    Delta = (Traj_OD.values[1] - Traj_OD.values[0])
    squre = [num * num for num in Delta]
    Sum = sum(squre)
    Distance = math.sqrt(Sum)
    Speed = Distance / time_gap
    return Speed

def Montion_Equations(Traj_OD):
    # 根据已知的两点计算直线方程
    # 创建一个空的 DataFrame
    Line_parmeter = pd.DataFrame(columns=['A', 'B', 'C'],index=[1]) # if you want create a new datafram you should add index=[1], it mean row
    Line_parmeter['A'] = Traj_OD['world_y'].values[1]-Traj_OD['world_y'][0]
    Line_parmeter['B'] = Traj_OD['world_x'].values[0] - Traj_OD['world_x'][1]
    Line_parmeter['C']= Traj_OD['world_x'].values[1]*Traj_OD['world_y'].values[0]-Traj_OD['world_x'].values[0]*Traj_OD['world_y'].values[1]
    return Line_parmeter

    
def Point_of_cross(Line_parmeter_1,Line_parmeter_2):
    # 计算两条直线的相交点
    Point_cross = pd.DataFrame()
    D = (Line_parmeter_1['A'].values)*(Line_parmeter_2['B'].values)-(Line_parmeter_2['A'].values)*(Line_parmeter_1['B'].values)
    # what is the D?
    if D !=0:
        point_x = ((Line_parmeter_1['B'].values)*(Line_parmeter_2['C'].values)-(Line_parmeter_2['B'].values)*(Line_parmeter_1['C'].values))/D
        Point_cross['world_x'] = point_x
        point_y = ((Line_parmeter_2['A'].values)*(Line_parmeter_1['C'].values)-(Line_parmeter_1['A'].values)*(Line_parmeter_2['C'].values))/D
        Point_cross['world_y'] = point_y
    # else:
    #     print("两直线平行")
    return Point_cross  
```

```python
# The inductor of the traffic conflict risk 
def PET(Point_cross,speed_A,speed_B,Traj_OD_A,Traj_OD_B):
  Delta_dis = Traj_OD_A.values[1]-Traj_OD_B.values[1]
  squre_dis = [num*num for num in Delta_dis]
  sum_dis = sum(squre_dis)
  Dis = math.sqrt(sum_dis)
  if Dis <= 10:
    if len(Point_cross)>=1:
      Delta_a = Point_cross.values[0]-Traj_OD_A.values[1]
      squre_a = [num*num for num in Delta_a]
      sum_dis_a = sum(squre_a)
      Dis_cross_a = math.sqrt(sum_dis_a)
      time_point_a = Dis_cross_a/speed_A
      # Calculate the time for car B to reach the conflict point
      Delta_b = Point_cross.values[0]-Traj_OD_B.values[1]
      squre_b = [num*num for num in Delta_b]
      sum_dis_b = sum(squre_b)
      Dis_cross_b = math.sqrt(sum_dis_b)
      time_point_b = Dis_cross_b/speed_B
      # 计算经过冲突点的时间
      PET_value = abs(time_point_a-time_point_b)
    else:
      # print('There is no cross point')
      PET_value = 10000
  else:
    PET_value = 9999
  return PET_value

def TTC(Point_cross,speed_A,speed_B,Traj_OD_A,Traj_OD_B,Delta):
  Delta_dis = Traj_OD_A.values[1]-Traj_OD_B.values[1]
  squre_dis = [num*num for num in Delta_dis]
  sum_dis = sum(squre_dis)
  Dis = math.sqrt(sum_dis)
  if Dis <= 10:
    if len(Point_cross)>=1:
      Delta_a = Point_cross.values[0]-Traj_OD_A.values[1]
      squre_a = [num*num for num in Delta_a]
      sum_dis_a = sum(squre_a)
      Dis_cross_a = math.sqrt(sum_dis_a)
      time_point_a = Dis_cross_a/speed_A
      # Calculate the time for car B to reach the conflict point
      Delta_b = Point_cross.values[0]-Traj_OD_B.values[1]
      squre_b = [num*num for num in Delta_b]
      sum_dis_b = sum(squre_b)
      Dis_cross_b = math.sqrt(sum_dis_b)
      time_point_b = Dis_cross_b/speed_B
      # 计算经过冲突点的时间
      ttc_time_aver = abs(time_point_a-time_point_b)
      if ttc_time_aver < Delta:
        TTC = min(time_point_a,time_point_b)
      else:
        TTC = 10000
    else:
      TTC = 999999
  else:
    TTC = 999999
  return TTC
```

```python
def Veh_motion_state(traj_all_data,veh_id,Time_frame):
    'extracte the motion state of vehicle_id'
    "提取主车在time_frame时刻的运动状态"
    Traj_veh_id = traj_all_data[traj_all_data['vehicle_id'] == veh_id]
    Time_frame = np.around(Time_frame, decimals=2)
    Traj_veh_id_time_O = Traj_veh_id[Traj_veh_id['frame_time'] == Time_frame]
    Next_frame = Time_frame+0.04
    Next_frame = np.around(Next_frame, decimals=2)
    Traj_veh_id_time_D = Traj_veh_id[Traj_veh_id['frame_time'] == Next_frame]
    Traj_veh_id_OD = pd.concat([Traj_veh_id_time_O, Traj_veh_id_time_D], ignore_index=True)
    Veh_a_point = Traj_veh_id_OD[['world_x', 'world_y']]
    return Veh_a_point
```

```python
#  calculate the TTC at each frame with the trajectory
def Calculate_conflict_indextor(data_path,save_path,time_gap,Delta):
    traj_data = pd.read_csv(data_path)
    ALL_Time_frames = pd.unique(traj_data['frame_time'])
    ALL_Time_frames.sort()
    Time_frames = ALL_Time_frames
    Time_frames = np.array(Time_frames)
    print(Time_frames)
    Columns_name = ['m_vehicle_id','m_frame_time','m_vehicle_type', 
                    'm_world_x', 'm_world_y', 'm_speed_x', 'm_speed_y',
                    'm_acc_x', 'm_acc_y', 'm_Jerk_x', 'm_Jerk_y','Angle_1', 
                    'sub_vehicle_id', 'sub_vehicle_type', 'sub_world_x', 
                    'sub_world_y', 'sub_speed_x', 'sub_speed_y','sub_acc_x',
                    'sub_acc_y', 'sub_Jerk_x', 'sub_Jerk_y','Angle_2','cross_point_x',
                    'cross_point_y', 'PET','TTC']
    # 'Creat the list to save all time result'
    df_all_time = pd.DataFrame(columns=Columns_name)
    for num_id in range(0,len(Time_frames),1):
        # uqdate with time_frame............................jhkjkh
        # 'Creat the list to save every frame time result'
        if num_id < len(Time_frames):
            frame_time = Time_frames[num_id]
            print(frame_time)
            Traj_same_time = traj_data[traj_data['frame_time'] == frame_time]
            Veh_same_times = pd.unique(Traj_same_time['vehicle_id'])
            #Need to determine if there is still a vehicle in the system in the next second
            Next_fram_time = frame_time+0.04
            Veh_same_times_next_frame_time = traj_data[traj_data['frame_time'] == Next_fram_time]
            if len(Veh_same_times) >= 2 and len(Veh_same_times_next_frame_time) >= 2:
                'create a list save each main vehicle'
                A = pd.DataFrame(
                        columns=['m_vehicle_id', 'm_frame_time','m_vehicle_type', 
                                 'm_world_x', 'm_world_y', 'm_speed_x','m_speed_y', 
                                 'm_acc_x','m_acc_y','m_Jerk_x','m_Jerk_y','Angle_1',
                                 'sub_vehicle_id', 'sub_vehicle_type','sub_world_x',
                               	 'sub_world_y','sub_speed_x','sub_speed_y',
                                 'sub_acc_x','sub_acc_y', 'sub_Jerk_x', 'sub_Jerk_y',
                                 'Angle_2','cross_point_x', 'cross_point_y', 
                                 'PET','TTC']
                                 )
                for veh_id in Veh_same_times:
                    # the main vehicel Uqdate with vehicle_id
                    Traj_M_veh = Traj_same_time[Traj_same_time['vehicle_id'] == veh_id]
                    Traj_M_veh = pd.DataFrame(Traj_M_veh)
                    df_main_veh = pd.DataFrame(
                        columns=['m_vehicle_id', 'm_frame_time','m_vehicle_type', 
                                 'm_world_x', 'm_world_y', 'm_speed_x','m_speed_y', 
                                 'm_acc_x', 'm_acc_y', 'm_Jerk_x','m_Jerk_y',
                                 'Angle_1','sub_vehicle_id','sub_vehicle_type', 
                                 'sub_world_x','sub_world_y', 'sub_speed_x',
                                 'sub_speed_y','sub_acc_x', 'sub_acc_y','sub_Jerk_x',
                                 'sub_Jerk_y','Angle_2','cross_point_x',
                                 'cross_point_y', 'PET','TTC'])
               df_main_veh[['m_vehicle_id','m_frame_time','m_vehicle_type','m_world_x', 'm_world_y', 'm_speed_x', 'm_speed_y','m_acc_x', 'm_acc_y', 'm_Jerk_x','m_Jerk_y','Angle']] = Traj_M_veh
                    Veh_a_point = Veh_motion_state(traj_data, veh_id, frame_time)
                    if len(Veh_a_point) >= 2:
                        Speed_a = Speed_Caluca(Veh_a_point,time_gap)
                        Line_parmeter_1 = Montion_Equations(Veh_a_point)
                        Sub_Veh_same_time = Veh_same_times
                        'creat a list to save the result of every sub_veh_variable'
                        Sub_vehicle_temp = pd.DataFrame(columns=['sub_veh_id', 'cross_point_x', 'cross_point_y', 'PET','TTC'])
                        Sub_veh_id_A = []
                        cross_point_B = []
                        PET_C = []
                        TTC_C =[]
                        for Sub_veh_id in Sub_Veh_same_time:
                            #for Sub_veh_id in Sub_Veh_same_time:
                            'extracte the motion state of sub vehicle'
                            Traj_sub_veh = Traj_same_time[Traj_same_time['vehicle_id'] == Sub_veh_id]
                            Veh_b_point = Veh_motion_state(traj_data, Sub_veh_id, frame_time)
                            if len(Veh_b_point) >= 2:
                                Speed_b = Speed_Caluca(Veh_b_point,time_gap)
                                Line_parmeter_2 = Montion_Equations(Veh_b_point)
                                'Calculate the cross point'
                                cross_point = Point_of_cross(Line_parmeter_1, Line_parmeter_2)
                                # if there are no cross points, we should stop the next step
                                if len(cross_point) != 0:
                                    'calculate the PET'
                                    conflict_pet = PET(cross_point, Speed_a, Speed_b, Veh_a_point, Veh_b_point)
                                    'Calculate the TTC'
                                    conflict_TTC = TTC(cross_point, Speed_a, Speed_b, Veh_a_point, Veh_b_point,Delta)
                                    # creat a list which contain the vehicle information and the PET at every frame
                                    Sub_veh_id_A.append(Sub_veh_id)
                                    cross_point_B.append(cross_point.values[0])
                                    PET_C.append(conflict_pet)
                                    TTC_C.append(conflict_TTC)
                        Sub_veh_id = pd.DataFrame(Sub_veh_id_A)
                        Sub_vehicle_temp['sub_veh_id'] = Sub_veh_id_A
                        Sub_vehicle_temp[['cross_point_x','cross_point_y']] = cross_point_B
                        Sub_vehicle_temp['PET'] = PET_C
                        Sub_vehicle_temp['TTC'] = TTC_C
                        Min_pet_value = min(Sub_vehicle_temp['TTC'])
                        Min_pet_sub_veh = Sub_vehicle_temp[Sub_vehicle_temp['TTC'] == Min_pet_value]
                        MIN_PET_SUB = Min_pet_sub_veh[['cross_point_x', 'cross_point_y', 'PET','TTC']]
                        sub_veh_min_pet_id = Min_pet_sub_veh['sub_veh_id'].values[0]

                    sub_veh_inf = Traj_same_time[Traj_same_time['vehicle_id'] == sub_veh_min_pet_id]
                    traj_data_sub_min_pet_id = sub_veh_inf[                   ['vehicle_id', 'vehicle_type', 'world_x', 'world_y', 'speed_x', 'speed_y', 'acc_x', 'acc_y', 'Jerk_x','Jerk_y','Angle']]
                    traj_data_sub_min_pet_id = pd.DataFrame(traj_data_sub_min_pet_id)
                    df_main_veh[                  ['sub_vehicle_id','sub_vehicle_type','sub_world_x','sub_world_y', 'sub_speed_x', 'sub_speed_y','sub_acc_x', 'sub_acc_y', 'sub_Jerk_x', 'sub_Jerk_y','Angle_2']] = traj_data_sub_min_pet_id.values[0]
                    df_main_veh[['cross_point_x', 'cross_point_y', 'PET','TTC']] = MIN_PET_SUB.values[0]
                    # Integration of all vehicle data at the same moment
                    integer_A_df_main_veh = [A,df_main_veh]
                    A = pd.concat(integer_A_df_main_veh)

        Integer_A_all_time = [df_all_time,A]
        df_all_time = pd.concat(Integer_A_all_time)
        # print(df_all_time)
    df_all_time.to_csv(save_path,index=False, header=True)
```

```python
Delta = 0.5
time_gap = 0.04
data_path = r'E:\CodeResource\000_Traffic_conflict_risk_analysis\Data_clearning\Data_set\Denoising\LC\wfLC_AM_C0004_Trajectory.csv'
save_path = r'E:\CodeResource\000_Traffic_conflict_risk_analysis\Data_clearning\Data_set\Conflict_index\c0004.csv'
Calculate_conflict_indextor(data_path,save_path,time_gap,Delta)
```

The distribution of the conflict risk shows as fellow, the data from the Longchang-Ningwu intersections.

|                ***Distribution of the PET***                 |      ***The distribution of conflict point (TTC<2s)***       |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="E:\CodeResource\000_Traffic_conflict_risk_analysis\Data_clearning\Data_set\distribution of pet by WN.png" alt="distribution of pet by WN" style="zoom:65%;" /> | <img src="E:\CodeResource\000_Traffic_conflict_risk_analysis\Data_clearning\Data_set\PET_LC.png" alt="PET_LC" style="zoom: 60%;" /> |

​                    **Figure. 5. The result of the risk inductors calculated from the trajectories**

**sixth, Modeling the evolving procession of interaction behavior between the motorized and Non-motorized vehicles.** 

a) The first step is extract the event chain about the course of interacting.

b) The second step define the variables related to the severity of conflict risk.

c) The  third step model the interaction behavior based on the ordinal logit model and the causality inference model.

d) The forth step model the interaction behavior based on the dynamic theory, dynamic Bayesian model and dynamic causality discovery theory.

e) The finally step build the model deflection the anomaly behavior based the dynamic theory.
