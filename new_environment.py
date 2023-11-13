# coding=gbk
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import datetime
import math

# ��ӹ�һ��ģ��
min_max_scaler = joblib.load(r'C:\data\����\model\feature_extraction1.pkl')

# ����dp��������㶯̬ϵ��
# ann_dp_model = load_model(r"C:\data\����\model\dp_poi_model_ure.h5")
rfc_dp_model = joblib.load(r"C:\data\����\model/dp_predict_rfc_feature4.pkl")

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

# ��������
# ��ʾ������
pd.set_option('display.max_columns', None)
# ��ʾ������
# pd.set_option('display.max_rows', None)
# ����value����ʾ����Ϊ100��Ĭ��Ϊ50
pd.set_option('max_colwidth', 100)
# �趨������Ϣ
np.random.seed(1)  # ������������ӣ�ʹ��ʵ�����ظ�
evaluate_data = pd.read_csv(r"C:\data\����\���ǿ������\���ջ���\����\evaluate_2.csv",
                            index_col=["s_id", "e_id"])
probability = pd.read_csv(r"C:\data\����\���ǿ������\���ջ���\ʰȡ����\pick_up_probability_new.csv",
                          index_col=["grid_id", "hour", "min_half", "daytype"])
transition_matrix = pd.read_csv(r"C:\data\����\���ǿ������\���ջ���\ת�Ƹ��ʺ�����\transition_matrix_new.csv",
                                index_col=["s_id", "s_hour", "s_min_half", "daytype"])
transition_matrix.sort_index(inplace=True)
poi_env = pd.read_csv(r"C:\data\����\���ǿ������\���ջ���\poi_count_in_grid.csv", index_col=["grid_id"])

poi_gat_data = pd.read_csv(r"C:\data\����\���ǿ������\���ջ���\mdp����Դ����\poi_count_grid_id_1000_gat.csv",
                           index_col=0)


# ����һ�����񻷾���,reward_dyΪ�Ƿ���ö�̬����[1Ϊ���ö�̬���ۣ�0Ϊ������]
class Env:
    def __init__(self, s_id, s_hour, s_min_half, daytype, q_type, reward_dy=1):
        # ���忪ʼʱ���״̬
        self.start_state = (s_id, s_hour, s_min_half, daytype)
        # ����տ�ʼ�ǵ�״̬����
        self.start_state_vector = self.tran_vector(self.start_state)
        # ���嶯���ռ�Ϊ�Ÿ�����
        self.action_name = ['left-down', 'down', 'right-down', 'right', 'stand', 'left', 'left-up', 'up', 'right-up']
        # ��ȡ��������
        self.n_actions = len(self.action_name)
        # �����ռ�ӳ��Ϊ�б�
        self.actions = [ac for ac in range(1, 10)]
        # ���嵱ǰʱ��
        self.time = datetime.time(s_hour, 30 * s_min_half, 0)
        # �����Ƿ�ʹ�ö�̬����
        self.reward_dy = reward_dy
        # ���嵱ǰ״̬(��ʱ��Ϊ��ʼ״̬��
        self.currentstate = self.start_state
        # ���嵱ǰ״̬���������ڻ�õ�ǰ״̬��dpϵ��
        self.currentstate_vector = self.start_state_vector
        # ����Q���������
        self.q_type = q_type
        # ����Q���������ά��
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

    # ����򵥵Ļ���״̬�ı䣬����Ƶ���任�����ܵ�������
    def environment_change(self, change_start_id, change_hour, change_s_min_half, change_daytype):
        # �ı俪ʼʱ���״̬
        self.start_state = (change_start_id, change_hour, change_s_min_half, change_daytype)
        # �ı俪ʼ��״̬����
        self.start_state_vector = self.tran_vector(self.start_state)
        # �ı价��ʱ��
        self.time = datetime.time(change_hour, 30 * change_s_min_half, 0)
        # �ı䵱ǰ״̬(��ʱ��Ϊ��ʼ״̬��
        self.currentstate = self.start_state
        # �ı䵱ǰ״̬���������ڻ�õ�ǰ״̬��dpϵ��
        self.currentstate_vector = self.start_state_vector

    # �������ú���
    def reset(self):
        # ���õ�ǰ״̬Ϊ��ʼ״̬
        self.currentstate = self.start_state
        self.currentstate_vector = self.start_state_vector
        # ����ʱ���ʼ��
        self.time = datetime.time(self.start_state[1], 30 * self.start_state[2], 0)

    # Ϊ�˺Ͷ���õ�ѧϰ����һ�£�����һ���յ�ˢ�º���
    def render(self):
        pass

    # Ϊ�˺Ͷ���õ�ѧϰ����һ�£�����һ�������ı�ı任����
    def tran_drawl_id(self, s):
        return s

    # ��������idת��Ϊ�����е����꣬���ԭ��Ϊ���½�
    def tran_id_grid_positon(self, gridid):
        l = gridid - 1
        x = int(l / 30) + 1
        y = l - (x - 1) * 30 + 1
        return [x, y]

        # ������now_grid��ȡ����action�� ������һ�����񡣷�����һ������id���ͳ����־

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

    # ״̬����תԤ����������������
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

    # �����Χ�ռ���Ϣ
    def get_poi_gat_information(self):
        around_grid = [self.get_next_grid(i) for i in range(1, 10)]
        around_poi_iif = poi_gat_data.loc[
            around_grid, ['guo_bus', 'bus_line_count', 'guo_metro_station',
                          'metro_line_count', '��ͨ��ʩ', 'ס�޷���', '��������', '������ʩ', '��˾��ҵ', 'ҽ�Ʊ�������',
                          '����סլ', '���������Լ�����', '��������', '�������', '�ƽ��Ļ�����', '�������', '�羰��ʤ', '��������']].values
        return around_poi_iif.reshape(-1)
        # ״̬ת��Ϊ���Q����ĳ�����������

    # ��һ��ת�����������������һ��
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

    # �ڶ��ַ�����������idת��Ϊ�к���Ȼ��one_shot����
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

    # �����ַ�����������idת��Ϊ�к��в��Ҳ�����one_shot����
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

    # �����ַ�����������idת��Ϊ�к��У����Ҳ�����one_shot���룬Сʱ��Ҳ������one_shot����
    def tran_q_vector4(self, state):
        s_id, s_hour, s_min_half, daytype = state
        position = np.array(self.tran_id_grid_positon(s_id))
        hour = np.array([s_hour])
        two_dim = np.eye(2)
        min_half = two_dim[int(s_min_half)]
        day_type = two_dim[int(daytype)]
        state_vector = np.concatenate([position, hour, min_half, day_type], axis=0)
        return state_vector

    # �����ַ�����������idת��Ϊ�к��У�����ʱ��������������one_shot����
    def tran_q_vector5(self, state):
        s_id, s_hour, s_min_half, daytype = state
        position = np.array(self.tran_id_grid_positon(s_id))
        hour = np.array([s_hour])
        min_half = np.array([s_min_half])
        day_type = np.array([daytype])
        state_vector = np.concatenate([position, hour, min_half, day_type], axis=0)
        return state_vector

    # ������ʱ������Ƕ����ɢ�����ĵ�λԲ��
    def time_degree_feature(self):
        minutes = self.time.hour * 60 + self.time.minute
        time_intervel = 2 * math.pi * (minutes / 1440)
        return [math.cos(time_intervel), math.sin(time_intervel)]

    # �����ַ�����������idת��Ϊ�к��У�ʱ������ת��Ϊ��λԲ
    def tran_q_vector6(self, state):
        s_id, s_hour, s_min_half, daytype = state
        position = np.array(self.tran_id_grid_positon(s_id))
        time = np.array(self.time_degree_feature())
        two_dim = np.eye(2)
        day_type = two_dim[int(daytype)]
        state_vector = np.concatenate([position, time, day_type], axis=0)
        return state_vector

    # �����ַ����� ��ȡ��Χ9����������Ŀռ���Ϣ�� ʱ������ת��Ϊ
    def tran_q_vector7(self):
        around_poi = self.get_poi_gat_information()
        # time = np.array(self.time_degree_feature())
        # two_dim = np.eye(2)
        # day_type = two_dim[int(self.currentstate[3])]
        # state_vector = np.concatenate([around_poi, time, day_type], axis=0)
        # return state_vector

    # ��õ�ǰ״̬�Ķ�̬ϵ��
    def get_dp_coefficient(self):
        # rfc����
        rfc_feature = [self.currentstate[0], 27, 4, self.currentstate[3], self.time.hour, self.time.minute]
        # rfcԤ��
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

    # ��ǰ״̬�ӵ��˿ͣ� �õ��˿͵�Ŀ��״̬
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

    # ����ʱ��ı仯
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

    # ������״̬S0���ö���a��
    def step(self, action, spend_time, seek_time, trans_time):
        '''
        :param action: ����ѡ��
        :param spend_time: �ۼƺ�ʱ
        :return: �ı����񻷾��ĵ�ǰ״̬�͵�ǰ״̬���������ز�ȥ����action֮���״̬��ת̬��������ʱ���˴ζ���������
        '''
        next_grid = self.get_next_grid(action)
        daytype = self.currentstate[3]
        # ִ�д�Ѱ�͹�����Ҫ����Ҫ�Ļ���
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
        # ������һ���ط�Ѱ�ͣ�����״̬
        next_state = (next_grid, self.time.hour, int(self.time.minute / 30), daytype)
        self.currentstate = next_state
        self.currentstate_vector = self.tran_vector(self.currentstate)
        prob = self.get_probability()
        flag_find = np.random.rand()
        if prob > flag_find:
            # ����һ��λ�ýӵ��˿ͣ�����Ŀ�ĵأ�����״̬
            final_id = self.get_transition()
            duration, length = self.cost(next_grid, final_id)
            # ��һ��reward���㷽������̬����
            if self.reward_dy == 1:
                dp = self.get_dp_coefficient()
                reward += dp * (15 + 2.8 * length)
            # �ڶ���reward���㷽����δʹ�ö�̬�۸����
            elif self.reward_dy == 0:
                reward += 1.0 * (15 + 2.8 * length)
            # ������reward���㷽����ʹ�ð�Сʱ�ڵ�ƽ����̬�۸����
            elif self.reward_dy == 2:
                dp = probability.at[self.currentstate, "average_dp"]
                reward += dp * (15 + 2.8 * length)
            spend_time += duration
            trans_time += duration
            self.time_change(duration)
            self.currentstate = (final_id, self.time.hour, int(self.time.minute / 30), daytype)
            self.currentstate_vector = self.tran_vector(self.currentstate)
        return self.currentstate, self.currentstate_vector, spend_time, reward, seek_time, trans_time

