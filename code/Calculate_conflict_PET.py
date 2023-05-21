# calculate the PET for each pair of trajectory
"PEF = Time_veh_1_pass_cross_point - Time_veh_2_pass_cross_point"
"here we need to calculate the distance between each vehicle pair's every coordinates "
# if the distance less the threshold value then is means exticing the cross point


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from itertools import combinations as comb
import os
import time


def Creat_trajectory_pair(trajectory_data):
    "This function create the trajectory pair form the trajectory data"
    Objectory_ids = pd.unique(trajectory_data['vehicle_id'])
    # the pair of the vehicle id
    combinations1 = list(comb(Objectory_ids, 2))
    combinations1 = pd.DataFrame(combinations1)
    return combinations1


def Ture_veh_pairs(combinations1,trajectory_data):
    "this function make sure the same time of the two vehicles"
    delet_id = []
    for id in range(0,len(combinations1)-1,1):
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
        if np.size(same_time_frame)<=1:
            # print(same_time_frame)
            "如果两辆车出现的时间没有交集，判断他们时间差，是否小于20秒"
            if ((min_f_veh_frame_time - max_s_veh_frame_time)>20) or ((min_s_veh_frame_time-max_f_veh_frame_time)>20):
                delet_id.append(id)
    combinations1 = combinations1.drop(delet_id)
    return combinations1



def Calculate_PET_value(trajectory_path,Columns_name,Dis_threshold):
    "this function main to check in the same time of the trajectory"
    trajectory_data = pd.read_csv(trajectory_path)
    combinations2 = Creat_trajectory_pair(trajectory_data)
    combinations1 = Ture_veh_pairs(combinations2, trajectory_data)
    # saving the conflict information
    df_all_veh_pairs = pd.DataFrame(columns=Columns_name)
    for id in range(0,len(combinations1),1):
        start_time = time.time()
        print(str(id)+'/'+str(len(combinations1)))
        F_veh_id = combinations1.iloc[id][0]
        S_veh_id = combinations1.iloc[id][1]
        first_vehicle_inf = trajectory_data[trajectory_data['vehicle_id']==F_veh_id]
        second_vehicle_inf = trajectory_data[trajectory_data['vehicle_id']==S_veh_id]
        "Search the same frame time of the two vehicles"
        second_vehicle_inf = second_vehicle_inf.reset_index(drop=True)
        df_one_veh_pair = pd.DataFrame(columns=Columns_name)
        # because of the frame time is related with the row, so we can use frame_time do the cycle
        first_veh_frme_times = pd.unique(first_vehicle_inf['frame_time'])
        for frame_time_id in first_veh_frme_times:
            # 这边只需要计算时间差值小于20秒的轨迹点对之间的距离

            frame_time_trajectory = first_vehicle_inf[first_vehicle_inf['frame_time']==frame_time_id]
            F_frame_time = frame_time_trajectory['frame_time'].values[0]

            Pre_frame_time = F_frame_time-20
            Pro_frame_time = F_frame_time+20

            second_vehicle_inf = second_vehicle_inf[second_vehicle_inf['frame_time'] <= Pro_frame_time]
            if Pre_frame_time>=0:
                second_vehicle_inf = second_vehicle_inf[second_vehicle_inf['frame_time']>=Pre_frame_time]

            "Create an empty table for save the pet information"
            df_one_frame_veh_pair = pd.DataFrame(columns=Columns_name)
            df_main_veh = pd.DataFrame(columns=['f_vehicle_id', 'f_frame_time', 'f_vehicle_type', 'f_world_x', 'f_world_y','f_speed_x', 'f_speed_y', 'f_acc_x', 'f_acc_y', 'f_Jerk_x', 'f_Jerk_y', 'f_Angle'])
            df_main_veh[
                ['f_vehicle_id', 'f_frame_time', 'f_vehicle_type', 'f_world_x', 'f_world_y','f_speed_x', 'f_speed_y', 'f_acc_x', 'f_acc_y', 'f_Jerk_x', 'f_Jerk_y', 'f_Angle']] = frame_time_trajectory
            first_row = df_main_veh.iloc[0]  # 获取第一行
            new_rows = pd.DataFrame([first_row] * (len(second_vehicle_inf) - 1),columns=df_main_veh.columns)  # copy len(second_vehicle_inf) row

            df_main_veh_1 = pd.concat([df_main_veh,new_rows], ignore_index=True)  # 将复制的行与原始DataFrame合并
            df_main_veh_add = pd.concat([df_main_veh_1, second_vehicle_inf], axis=1,ignore_index=True)
            df_main_veh_add = pd.DataFrame(df_main_veh_add)
            df_main_veh_add.columns = ['F_vehicle_id','F_frame_time','F_vehicle_type','F_world_x','F_world_y','F_speed_x','F_speed_y','F_acc_x', 'F_acc_y', 'F_Jerk_x', 'F_Jerk_y','F_Angle',
                'S_vehicle_id','s_frame_time','S_vehicle_type','S_world_x','S_world_y','S_speed_x','S_speed_y','S_acc_x','S_acc_y','S_Jerk_x','S_Jerk_y','S_Angle']
            "here i'd like calculate the distance between two point by the normal"
            DIS_AT_FSAME_FRAME = np.sqrt((df_main_veh_add['F_world_x'] - df_main_veh_add['S_world_x']) ** 2 + (df_main_veh_add['F_world_y'] - df_main_veh_add['S_world_y']) ** 2)
            negative_indices = np.where(DIS_AT_FSAME_FRAME < Dis_threshold)[0]
            if len(negative_indices) >0:
                for id in negative_indices:
                    # loction_id = negative_indices[id]
                    "search the trajectory information and save in the empty table"
                    event_inf = df_main_veh_add.iloc[id]
                    "Calculate the pet value"
                    Event_inf = pd.DataFrame(event_inf)
                    # transpose the row and the column
                    transposed_df = Event_inf.transpose()
                    transposed_df['PET'] = transposed_df['F_frame_time'] - transposed_df['s_frame_time']
                    transposed_df = transposed_df.reset_index(drop=True)
                    df_one_frame_veh_pair = pd.concat([df_one_frame_veh_pair,transposed_df])
                df_one_veh_pair = pd.concat([df_one_veh_pair,df_one_frame_veh_pair])
        df_all_veh_pairs = pd.concat([df_all_veh_pairs,df_one_veh_pair])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("函数运行时长：", elapsed_time, "秒")
    return df_all_veh_pairs


def File_procession(Input_file_path,Output_file_path,Columns_name,Dis_threshold):
    "This function will process the csv in the file path"
    files1 = os.listdir(Input_file_path)
    for i in range(len(files1)):
        work_file = Input_file_path +files1[i]
        save_path = Output_file_path +'/conflict_'+ files1[i]
        Trajectory_conflict = Calculate_PET_value(work_file,Columns_name,Dis_threshold)
        Trajectory_conflict.to_csv(save_path, index=False, header=True)
    return Trajectory_conflict



Columns_name = ['F_vehicle_id','F_frame_time','F_vehicle_type','F_world_x','F_world_y','F_speed_x','F_speed_y','F_acc_x', 'F_acc_y', 'F_Jerk_x', 'F_Jerk_y','F_Angle',
                'S_vehicle_id','s_frame_time','S_vehicle_type','S_world_x','S_world_y','S_speed_x','S_speed_y','S_acc_x','S_acc_y','S_Jerk_x','S_Jerk_y','S_Angle','PET']

Dis_threshold = 0.5
data_path =  r'E:/CodeResource/000_Traffic_conflict_risk_analysis/Data_clearning/Data_set/Denoising/WN/test/'
output_path = r'E:/CodeResource/000_Traffic_conflict_risk_analysis/Data_clearning/Data_set/Conflict_index/WN'
Trajectory_conflict = File_procession(data_path,output_path,Columns_name,Dis_threshold)
