# calculate the TTC for each pair of trajectory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from itertools import combinations as comb
import os


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


def TTC(Point_cross,speed_A,speed_B,Traj_OD_A,Traj_OD_B,Delta):
  Delta_dis = Traj_OD_A.values[1]-Traj_OD_B.values[1]
  squre_dis = [num*num for num in Delta_dis]
  sum_dis = sum(squre_dis)
  Dis = math.sqrt(sum_dis)
  if len(Point_cross)>=1:
    Delta_a = Point_cross.values[0]-Traj_OD_A.values[1]
    squre_a = [num*num for num in Delta_a]
    sum_dis_a = sum(squre_a)
    Dis_cross_a = math.sqrt(sum_dis_a)
    if speed_A!=0:
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
        TTC = max(time_point_a,time_point_b)
      else:
        TTC = 10000
    else:
      TTC = 22222
  else:
    TTC = 9999999
  return TTC



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

def Creat_trajectory_pair(trajectory_data):
    "This function create the trajectory pair form the trajectory data"
    Objectory_ids = pd.unique(trajectory_data['vehicle_id'])
    # the pair of the vehicle id
    combinations1 = list(comb(Objectory_ids, 2))
    combinations1 = pd.DataFrame(combinations1)
    return combinations1

def Extract_same_time(first_vehicle_inf,second_vehicle_inf):
    "extract the same time frame time"
    first_vehicle_all_frame = first_vehicle_inf['frame_time']
    second_vehicle_all_frame = second_vehicle_inf['frame_time']
    # same_time_frame = pd.Series(list(set(first_vehicle_all_frame['frame_time']) & set(second_vehicle_all_frame['frame_time'])))
    same_time_frame = pd.Series(list(set(first_vehicle_all_frame) & set(second_vehicle_all_frame)))
    return same_time_frame

def Extract_traj_by_frame_time(trajectory,same_time_frame):
    "extract the trajectory by the frame time"
    "the input is the first_vehicle_inf or the second_vehicle_inf"
    same_time_inf = trajectory[trajectory['frame_time'].isin(same_time_frame)]
    return same_time_inf

def Ture_veh_pairs(combinations1,trajectory_data):
    "this function make sure the same time of the two vehicles"
    delete_id = []
    for id in range(0,len(combinations1),1):
        print(str(id) + '/' + str(len(combinations1)))
        F_veh_id = combinations1.iloc[id][0]
        S_veh_id = combinations1.iloc[id][1]
        first_vehicle_inf = trajectory_data[trajectory_data['vehicle_id']==F_veh_id]
        second_vehicle_inf = trajectory_data[trajectory_data['vehicle_id']==S_veh_id]
        "Search the same frame time of the two vehicles"
        second_vehicle_inf = second_vehicle_inf.reset_index(drop=True)
        df_one_veh_pair = pd.DataFrame(columns=Columns_name)
        # because of the frame time is related with the row, so we can use frame_time do the cycle
        first_veh_frme_times = pd.unique(first_vehicle_inf['frame_time'])
        # we dont't need calculate every frame in the trajectory, if the two vehicles's frame time gap is bigger than 20s wiil don't need continue
        F_vehicle_frame_time = first_vehicle_inf['frame_time']
        S_vehicle_frame_time = second_vehicle_inf['frame_time']
        min_f_veh_frame_time = min(F_vehicle_frame_time)
        max_f_veh_frame_time = max(F_vehicle_frame_time)
        min_s_veh_frame_time = min(S_vehicle_frame_time)
        max_s_veh_frame_time = max(S_vehicle_frame_time)
        same_time_frame = np.intersect1d(F_vehicle_frame_time, S_vehicle_frame_time)
        same_time_frame = np.sort(same_time_frame)
        if np.size(same_time_frame)<=0:
            # print(same_time_frame)
            "如果两辆车出现的时间没有交集，判断他们时间差，是否小于20秒"
            delete_id.append(id)
    combinations1 = combinations1.drop(delete_id)
    return combinations1




def Calculate_TTC_value(trajectory_path,Columns_name,time_gap,Delta):
    "this function main to check in the same time of the trajectory"
    trajectory_data = pd.read_csv(trajectory_path)
    combinations2 = Creat_trajectory_pair(trajectory_data)
    combinations1 = Ture_veh_pairs(combinations2, trajectory_data)

    # saving the conflict information
    df_all_veh_pairs = pd.DataFrame(columns=Columns_name)
    for id in range(0,len(combinations1),1):
        print(str(id)+'/'+str(len(combinations1)))
        F_veh_id = combinations1.iloc[id][0]
        S_veh_id = combinations1.iloc[id][1]
        first_vehicle_inf = trajectory_data[trajectory_data['vehicle_id']==F_veh_id]
        second_vehicle_inf = trajectory_data[trajectory_data['vehicle_id']==S_veh_id]
        first_vehicle_frame_time = first_vehicle_inf['frame_time']
        second_vehicle_frame_time = second_vehicle_inf['frame_time']
        same_time_frame = np.intersect1d(first_vehicle_frame_time, second_vehicle_frame_time)
        same_time_frame = np.sort(same_time_frame)
        "extract the same time series"
        df_one_veh_pair = pd.DataFrame(columns=Columns_name)
        if len(same_time_frame)>=0:
            first_trajectory_same_time = Extract_traj_by_frame_time(first_vehicle_inf,same_time_frame)
            second_trajectory_same_time = Extract_traj_by_frame_time(second_vehicle_inf,same_time_frame)
            for frame_time_id in  same_time_frame:
                df_one_frame = pd.DataFrame(columns=Columns_name)
                first_trajectory_rame_time_id = first_trajectory_same_time[first_trajectory_same_time['frame_time']==frame_time_id]
                second_trajectory_rame_time_id = second_trajectory_same_time[second_trajectory_same_time['frame_time'] == frame_time_id]
                # Next_fram_time = frame_time_id + 0.04
                # # first_traj_next_frame_time = first_trajectory_same_time[first_trajectory_same_time['frame_time'] == Next_fram_time]
                # # Second_traj_next_frame_time = second_trajectory_same_time[second_trajectory_same_time['frame_time'] == Next_fram_time]
                f_Veh_a_point = Veh_motion_state(trajectory_data, F_veh_id, frame_time_id)
                s_Veh_a_point = Veh_motion_state(trajectory_data, S_veh_id, frame_time_id)
                if (len(f_Veh_a_point) >= 2) & (len(s_Veh_a_point) >= 2):
                    F_veh_speed = Speed_Caluca(f_Veh_a_point, time_gap)
                    S_veh_speed = Speed_Caluca(s_Veh_a_point, time_gap)

                    F_Line_parmeter = Montion_Equations(f_Veh_a_point)
                    S_Line_parmeter = Montion_Equations(s_Veh_a_point)

                    'Calculate the cross point'
                    cross_point = Point_of_cross(F_Line_parmeter, S_Line_parmeter)
                    # if there are no cross points, we should stop the next step
                    if len(cross_point) != 0:
                        'Calculate the TTC'
                        conflict_TTC = TTC(cross_point, F_veh_speed, S_veh_speed, f_Veh_a_point, s_Veh_a_point, Delta)
                        # creat a list which contain the vehicle information and the PET at every fram
                        df_one_frame[['F_vehicle_id', 'F_frame_time','F_vehicle_type','F_world_x', 'F_world_y', 'F_speed_x','F_speed_y','F_acc_x','F_acc_y','F_Jerk_x','F_Jerk_y','F_Angle']] = first_trajectory_rame_time_id
                        df_one_frame['S_vehicle_id'] = second_trajectory_rame_time_id['vehicle_id'].values
                        df_one_frame['S_vehicle_type'] = second_trajectory_rame_time_id['vehicle_type'].values
                        df_one_frame['S_world_x'] = second_trajectory_rame_time_id['world_x'].values
                        df_one_frame['S_world_y'] = second_trajectory_rame_time_id['world_y'].values
                        df_one_frame['S_speed_x'] = second_trajectory_rame_time_id['speed_x'].values
                        df_one_frame['S_speed_y'] = second_trajectory_rame_time_id['speed_y'].values
                        df_one_frame['S_acc_x'] = second_trajectory_rame_time_id['acc_x'].values
                        df_one_frame['S_acc_y'] = second_trajectory_rame_time_id['acc_y'].values
                        df_one_frame['S_Jerk_x'] = second_trajectory_rame_time_id['Jerk_x'].values
                        df_one_frame['S_Jerk_y'] = second_trajectory_rame_time_id['Jerk_y'].values
                        df_one_frame['S_Angle'] = second_trajectory_rame_time_id['Angle'].values
                        df_one_frame['cross_point_x'] = cross_point.values[0][0]
                        df_one_frame['cross_point_y'] = cross_point.values[0][1]
                        df_one_frame['TTC'] = conflict_TTC
                        df_one_veh_pair_add = [df_one_veh_pair,df_one_frame]
                        df_one_veh_pair = pd.concat(df_one_veh_pair_add)
            df_all_veh_pairs_add = [df_all_veh_pairs,df_one_veh_pair]
            df_all_veh_pairs = pd.concat(df_all_veh_pairs_add)
    return df_all_veh_pairs

def File_procession(Input_file_path,Output_file_path,time_gap,Delta,Columns_name):
    "This function will process the csv in the file path"
    files1 = os.listdir(Input_file_path)
    for i in range(len(files1)):
        work_file = Input_file_path +files1[i]
        save_path = Output_file_path +'/conflict_'+ files1[i]
        Trajectory_conflict = Calculate_TTC_value(work_file,Columns_name,time_gap,Delta)
        Trajectory_conflict.to_csv(save_path, index=False, header=True)
    return Trajectory_conflict

Columns_name = ['F_vehicle_id','F_frame_time','F_vehicle_type','F_world_x','F_world_y','F_speed_x','F_speed_y','F_acc_x', 'F_acc_y', 'F_Jerk_x', 'F_Jerk_y','F_Angle',
                'S_vehicle_id','S_vehicle_type','S_world_x','S_world_y','S_speed_x','S_speed_y','S_acc_x','S_acc_y','S_Jerk_x','S_Jerk_y','S_Angle','cross_point_x','cross_point_y','TTC']
Delta = 10
time_gap = 0.04
data_path =  r'E:/CodeResource/000_Traffic_conflict_risk_analysis/Data_clearning/Data_set/Denoising/LC/test/'
output_path = r'E:/CodeResource/000_Traffic_conflict_risk_analysis/Data_clearning/Data_set/Denoising/test'
Trajectory_conflict = File_procession(data_path,output_path,time_gap,Delta,Columns_name)
