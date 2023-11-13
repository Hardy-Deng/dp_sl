# coding=gbk
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import datetime
import math

# 添加归一化模型
min_max_scaler = joblib.load(r'C:\data\代码\model\feature_extraction1.pkl')

# 导入dp神经网络计算动态系数
# ann_dp_model = load_model(r"C:\data\代码\model\dp_poi_model_ure.h5")
rfc_dp_model = joblib.load(r"C:\data\代码\model/dp_predict_rfc_feature4.pkl")

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

# 环境常量
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 设定环境信息
np.random.seed(1)  # 设置随机数种子，使得实验能重复
evaluate_data = pd.read_csv(r"C:\data\代码\深度强化环境\最终环境\评估\evaluate_2.csv",
                            index_col=["s_id", "e_id"])
probability = pd.read_csv(r"C:\data\代码\深度强化环境\最终环境\拾取概率\pick_up_probability_new.csv",
                          index_col=["grid_id", "hour", "min_half", "daytype"])
transition_matrix = pd.read_csv(r"C:\data\代码\深度强化环境\最终环境\转移概率和收益\transition_matrix_new.csv",
                                index_col=["s_id", "s_hour", "s_min_half", "daytype"])
transition_matrix.sort_index(inplace=True)
poi_env = pd.read_csv(r"C:\data\代码\深度强化环境\最终环境\poi_count_in_grid.csv", index_col=["grid_id"])

poi_gat_data = pd.read_csv(r"C:\data\代码\深度强化环境\最终环境\mdp环境源数据\poi_count_grid_id_1000_gat.csv",
                           index_col=0)


# 创建一个网格环境类,reward_dy为是否采用动态定价[1为采用动态定价，0为不采用]
class Env:
    def __init__(self, s_id, s_hour, s_min_half, daytype, q_type, reward_dy=1):
        # 定义开始时候的状态
        self.start_state = (s_id, s_hour, s_min_half, daytype)
        # 定义刚开始是的状态向量
        self.start_state_vector = self.tran_vector(self.start_state)
        # 定义动作空间为九个动作
        self.action_name = ['left-down', 'down', 'right-down', 'right', 'stand', 'left', 'left-up', 'up', 'right-up']
        # 获取动作数量
        self.n_actions = len(self.action_name)
        # 动作空间映射为列表
        self.actions = [ac for ac in range(1, 10)]
        # 定义当前时刻
        self.time = datetime.time(s_hour, 30 * s_min_half, 0)
        # 定义是否使用动态定价
        self.reward_dy = reward_dy
        # 定义当前状态(暂时定为开始状态）
        self.currentstate = self.start_state
        # 定义当前状态向量，用于获得当前状态的dp系数
        self.currentstate_vector = self.start_state_vector
        # 定义Q网络的类型
        self.q_type = q_type
        # 定义Q网络输入的维度
        if self.q_type == 1:
            self.q_input_features = len(self.tran_q_vector1(self.currentstate))
        elif self.q_type == 2:
            self.q_input_features = len(self.tran_q_vector2(self.currentstate))
        elif self.q_type == 3:
            self.q_input_features = len(self.tran_q_vector3(self.currentstate))
        elif self.q_type == 4:
            self.q_input_features = len(self.tran_q_vector4(self.currentstate))
        elif self.q_type == 5:
            self.q_input_features = len(self.tran_q_vector5(self.currentstate))
        elif self.q_type == 6:
            self.q_input_features = len(self.tran_q_vector6(self.currentstate))
        elif self.q_type == 7:
            self.q_input_features = len(self.tran_q_vector7())

    # 定义简单的环境状态改变，方便频繁变换环境跑单次收益
    def environment_change(self, change_start_id, change_hour, change_s_min_half, change_daytype):
        # 改变开始时候的状态
        self.start_state = (change_start_id, change_hour, change_s_min_half, change_daytype)
        # 改变开始的状态向量
        self.start_state_vector = self.tran_vector(self.start_state)
        # 改变环境时刻
        self.time = datetime.time(change_hour, 30 * change_s_min_half, 0)
        # 改变当前状态(暂时定为开始状态）
        self.currentstate = self.start_state
        # 改变当前状态向量，用于获得当前状态的dp系数
        self.currentstate_vector = self.start_state_vector

    # 定义重置函数
    def reset(self):
        # 重置当前状态为开始状态
        self.currentstate = self.start_state
        self.currentstate_vector = self.start_state_vector
        # 环境时间初始化
        self.time = datetime.time(self.start_state[1], 30 * self.start_state[2], 0)

    # 为了和定义好的学习函数一致，定义一个空的刷新函数
    def render(self):
        pass

    # 为了和定义好的学习函数一致，定义一个不做改变的变换函数
    def tran_drawl_id(self, s):
        return s

    # 将将网格id转化为网格中的坐标，左边原点为左下角
    def tran_id_grid_positon(self, gridid):
        l = gridid - 1
        x = int(l / 30) + 1
        y = l - (x - 1) * 30 + 1
        return [x, y]

        # 车辆在now_grid采取动作action， 进入下一个网格。返回下一个网格id，和出界标志

    def get_next_grid(self, action):
        now_grid = self.currentstate[0]
        x, y = self.tran_id_grid_positon(now_grid)
        if action == 1:
            x -= 1
            y -= 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x -= 1
            y += 1
        elif action == 4:
            y += 1
        elif action == 5:
            return now_grid
        elif action == 6:
            y -= 1
        elif action == 7:
            x += 1
            y -= 1
        elif action == 8:
            x += 1
        elif action == 9:
            x += 1
            y += 1
        if 0 < x < 31 and 0 < y < 31:
            return 30 * (x - 1) + y
        else:
            return 0

    # 状态向量转预测网络特征向量：
    def tran_vector(self, state):
        s_id, s_hour, s_min_half, daytype = state
        poi = np.array(poi_env.loc[int(s_id)])
        hour_feature_matrix = np.eye(24)
        hour = hour_feature_matrix[int(s_hour)]
        two_dim = np.eye(2)
        min_half = two_dim[int(s_min_half)]
        day_type = two_dim[int(daytype)]
        state_vector = np.concatenate([poi, hour, min_half, day_type], axis=0)
        return state_vector

    # 获得周围空间信息
    def get_poi_gat_information(self):
        around_grid = [self.get_next_grid(i) for i in range(1, 10)]
        around_poi_iif = poi_gat_data.loc[
            around_grid, ['guo_bus', 'bus_line_count', 'guo_metro_station',
                          'metro_line_count', '交通设施', '住宿服务', '体育休闲', '公共设施', '公司企业', '医疗保健服务',
                          '商务住宅', '政府机构以及团体', '汽车销售', '生活服务', '科教文化服务', '购物服务', '风景名胜', '餐饮服务']].values
        return around_poi_iif.reshape(-1)
        # 状态转化为深度Q网络的出入特征向量

    # 第一种转化方法：单纯引入归一化
    def tran_q_vector1(self, state):
        s_id, s_hour, s_min_half, daytype = state
        poi = np.array(poi_env.loc[int(s_id)])
        hour_feature_matrix = np.eye(24)
        hour = hour_feature_matrix[int(s_hour)]
        two_dim = np.eye(2)
        min_half = two_dim[int(s_min_half)]
        day_type = two_dim[int(daytype)]
        state_vector = np.concatenate([poi, hour, min_half, day_type], axis=0)
        return min_max_scaler.transform(state_vector.reshape(1, len(state_vector))).reshape(len(state_vector))

    # 第二种方法，将网格id转化为行和列然后one_shot编码
    def tran_q_vector2(self, state):
        s_id, s_hour, s_min_half, daytype = state
        x, y = self.tran_id_grid_positon(s_id)
        positon = np.eye(30)
        grid_col = positon[int(x - 1)]
        grid_row = positon[int(y - 1)]
        hour_feature_matrix = np.eye(24)
        hour = hour_feature_matrix[int(s_hour)]
        two_dim = np.eye(2)
        min_half = two_dim[int(s_min_half)]
        day_type = two_dim[int(daytype)]
        state_vector = np.concatenate([grid_col, grid_row, hour, min_half, day_type], axis=0)
        return state_vector

    # 第三种方法，将网格id转化为行和列并且不进行one_shot编码
    def tran_q_vector3(self, state):
        s_id, s_hour, s_min_half, daytype = state
        position = np.array(self.tran_id_grid_positon(s_id))
        hour_feature_matrix = np.eye(24)
        hour = hour_feature_matrix[int(s_hour)]
        two_dim = np.eye(2)
        min_half = two_dim[int(s_min_half)]
        day_type = two_dim[int(daytype)]
        state_vector = np.concatenate([position, hour, min_half, day_type], axis=0)
        return state_vector

    # 第四种方法，将网络id转化为行和列，并且不进行one_shot编码，小时数也不进行one_shot编码
    def tran_q_vector4(self, state):
        s_id, s_hour, s_min_half, daytype = state
        position = np.array(self.tran_id_grid_positon(s_id))
        hour = np.array([s_hour])
        two_dim = np.eye(2)
        min_half = two_dim[int(s_min_half)]
        day_type = two_dim[int(daytype)]
        state_vector = np.concatenate([position, hour, min_half, day_type], axis=0)
        return state_vector

    # 第四种方法，将网络id转化为行和列，并且时空特征都不进行one_shot编码
    def tran_q_vector5(self, state):
        s_id, s_hour, s_min_half, daytype = state
        position = np.array(self.tran_id_grid_positon(s_id))
        hour = np.array([s_hour])
        min_half = np.array([s_min_half])
        day_type = np.array([daytype])
        state_vector = np.concatenate([position, hour, min_half, day_type], axis=0)
        return state_vector

    # 将连续时间特征嵌入离散分区的单位圆上
    def time_degree_feature(self):
        minutes = self.time.hour * 60 + self.time.minute
        time_intervel = 2 * math.pi * (minutes / 1440)
        return [math.cos(time_intervel), math.sin(time_intervel)]

    # 第六种方法，将网络id转化为行和列，时间特征转化为单位圆
    def tran_q_vector6(self, state):
        s_id, s_hour, s_min_half, daytype = state
        position = np.array(self.tran_id_grid_positon(s_id))
        time = np.array(self.time_degree_feature())
        two_dim = np.eye(2)
        day_type = two_dim[int(daytype)]
        state_vector = np.concatenate([position, time, day_type], axis=0)
        return state_vector

    # 第七种方法， 获取周围9个网格区域的空间信息， 时间特征转化为
    def tran_q_vector7(self):
        around_poi = self.get_poi_gat_information()
        # time = np.array(self.time_degree_feature())
        # two_dim = np.eye(2)
        # day_type = two_dim[int(self.currentstate[3])]
        # state_vector = np.concatenate([around_poi, time, day_type], axis=0)
        # return state_vector

    # 获得当前状态的动态系数
    def get_dp_coefficient(self):
        # rfc特征
        rfc_feature = [self.currentstate[0], 27, 4, self.currentstate[3], self.time.hour, self.time.minute]
        # rfc预测
        rfc_dp_distribution = rfc_dp_model.predict_proba([rfc_feature]).reshape(7)
        dp_array = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        dp = np.random.choice(dp_array, p=rfc_dp_distribution)
        return dp

    def get_probability(self):
        prob = probability.at[self.currentstate, "pick_up_possibility"]
        if prob != prob:
            return 0
        else:
            return prob

    # 当前状态接到乘客， 得到乘客的目的状态
    def get_transition(self):
        tran_group = transition_matrix.loc[self.currentstate, :]
        tran_group.reset_index(drop=True, inplace=True)
        distribute = tran_group["transition_probability"].values
        ran = range(len(distribute))
        index = np.random.choice(ran, p=distribute.ravel())
        e_id = tran_group.at[index, "e_id"]
        # e_hour = tran_group.at[index, "e_hour"]
        # e_min_half = tran_group.at[index, "e_min_half"]
        # daytype = self.currentstate[3]
        # next_state = (e_id, e_hour, e_min_half, daytype)
        final_id = e_id
        return final_id

    def cost(self, s_id, e_id):
        if s_id > e_id:
            s_id, e_id = e_id, s_id
        duration = evaluate_data.at[(s_id, e_id), "statistics_duration"]
        length = evaluate_data.at[(s_id, e_id), "linear_distance"]
        return duration, length

    # 环境时间的变化
    def time_change(self, minute):
        hour = self.time.hour
        pre_minute = self.time.minute
        now_minute = int(pre_minute + minute)
        if now_minute >= 60:
            hour = int(hour + (now_minute / 60))
            now_minute = now_minute % 60
            if hour >= 24:
                hour = hour % 24
        self.time = datetime.time(hour, now_minute, 0)

    # 车辆在状态S0采用动作a：
    def step(self, action, spend_time, seek_time, trans_time):
        '''
        :param action: 动作选择
        :param spend_time: 累计耗时
        :return: 改变网格环境的当前状态和当前状态向量，返回才去动作action之后的状态，转态向量，耗时，此次动作的收益
        '''
        next_grid = self.get_next_grid(action)
        daytype = self.currentstate[3]
        # 执行此寻客过程需要的需要的花销
        if action == 1 or action == 3 or action == 7 or action == 9:
            reward = -0.5 * 1.4
            spend_time += 2
            seek_time += 2
            self.time_change(2)
        else:
            reward = -0.5 * 1
            spend_time += 1
            seek_time += 1
            self.time_change(1)
        # 进入下一个地方寻客，更新状态
        next_state = (next_grid, self.time.hour, int(self.time.minute / 30), daytype)
        self.currentstate = next_state
        self.currentstate_vector = self.tran_vector(self.currentstate)
        prob = self.get_probability()
        flag_find = np.random.rand()
        if prob > flag_find:
            # 在下一个位置接到乘客，送往目的地，更新状态
            final_id = self.get_transition()
            duration, length = self.cost(next_grid, final_id)
            # 第一种reward计算方法，动态定价
            if self.reward_dy == 1:
                dp = self.get_dp_coefficient()
                reward += dp * (15 + 2.8 * length)
            # 第二种reward计算方法，未使用动态价格乘数
            elif self.reward_dy == 0:
                reward += 1.0 * (15 + 2.8 * length)
            # 第三种reward计算方法，使用半小时内的平均动态价格乘数
            elif self.reward_dy == 2:
                dp = probability.at[self.currentstate, "average_dp"]
                reward += dp * (15 + 2.8 * length)
            spend_time += duration
            trans_time += duration
            self.time_change(duration)
            self.currentstate = (final_id, self.time.hour, int(self.time.minute / 30), daytype)
            self.currentstate_vector = self.tran_vector(self.currentstate)
        return self.currentstate, self.currentstate_vector, spend_time, reward, seek_time, trans_time

