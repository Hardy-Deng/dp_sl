import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import copy
from new_environment import Env
import matplotlib.pyplot as plt
import time as time_clock
import os

# # 设置GPU及其占用
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 环境常量设置
# tf.random.set_seed(1)  # 设置TensorFlow架构随机数种子，方便重复实验
# np.random.seed(1)  # 设置numpy随机数种子，方便重复实验
Max_episode = 10000
MEMORY_MAX_SIZE = 1000  # 经验池大小
REPLACE_TIME = 30  # 参数替换次数
BARCH_SIZE = 64  # Q网络批量大小
LEARNING_RATE = 0.01  # 强化学习的学习率
GAMA = 0.7
SPEND_TIME = 60  # 司机运行分钟数
Exploration_rate = 1  # 探索概率
Explore_decay_rate = 0.995  # 探索衰减
EPSILON = 0.1
Lambda = 0.5

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


# backward eligibility traces
class SarsaLambdaTable:
    def __init__(self, learning_rate=LEARNING_RATE, gama=GAMA, epsilon=EPSILON, lamb=Lambda, q_table=None):
        # backward view, eligibility trace.
        self.lambda_ = lamb
        self.epsilon = epsilon
        self.gamma = gama
        self.lr = learning_rate
        self.eligibility_trace = self.create_q_table()
        self.q_table = q_table
        if self.q_table is None:
            self.q_table = self.create_q_table()

    # 初始化Q表， Q表都存储在一个数组内。Q表为所有时间的q表集合
    def create_q_table(self):
        index = pd.MultiIndex.from_product([[i for i in range(1, 901)],
                                            [i for i in range(0, 60)]])
        q_table = pd.DataFrame(columns=range(1, 10), data=np.zeros([54000, 9]),
                               index=index)
        q_table.index.names = ["grid_id", "minute"]
        return q_table

    # 随机选择qt中最大值中的一个返回
    def random_select_max(self, qt):
        max_value = max(qt)
        max_index = []
        for i in range(9):
            if qt[i] == max_value:
                max_index.append(i + 1)
        return np.random.choice(max_index)

    # 在时间t状态l是动作选择，返回动作
    def choose_action(self, state):
        epsilon = self.epsilon
        actions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if np.random.rand() < epsilon:
            # 贪婪策略随机探索动作
            action = np.random.choice(actions)
        else:

            # qt[index, acition]    index代表网格的索引， action 代表 动作
            qt = np.array(self.q_table.loc[state].values)
            # action = np.argmax(qt) + 1
            action = self.random_select_max(qt)
        return action

    # 每个轮次将e_table重新初始化
    def reset_e_table(self):
        self.eligibility_trace = 0 * self.eligibility_trace

    def learn_on_policy(self, s, a, r, s_, a_, done):
        q_predict = self.q_table.loc[s, a]
        if done:
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict

        # increase trace amount for visited state-action pair

        # Method 1:
        self.eligibility_trace.loc[s, a] += 1
        # Method 2:
        # self.eligibility_trace.loc[s, :] *= 0
        # self.eligibility_trace.loc[s, a] = 1
        # Q update
        self.q_table += self.lr * error * self.eligibility_trace
        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma * self.lambda_


# Sarsa(lambda)
def update_sarsa_lambda_leaning(q, env, episode=Max_episode):
    # 本轮次的总收益
    score = []
    # 本轮次的接单数
    n_orders = []
    # 本轮次的每单平均收益
    peer_order_revenue = []
    # 本轮次单位是时间的收益（每60分钟）
    peer_hour_revenue = []
    # 每一轮次的有效载客时间和寻客时间
    sum_trans_time = []
    sum_seek_time = []
    for i_episode in range(episode):
        # 每轮次随机开始地点
        start_id, start_hour, start_min_half, daytype = env.start_state
        star_place = np.random.randint(1, 901)
        env.environment_change(star_place, start_hour, start_min_half, daytype)
        # 当前状态
        observation = (star_place, 0)
        # 初始化e_table
        q.reset_e_table()
        a_sample = []
        r_sample = []
        # 重置网格环境
        env.reset()
        # 先依靠q表探索产生一个动作
        action = q.choose_action(observation)
        # 当执行动作action后跳出网格区域，重新选择动作
        next_id = env.get_next_grid(action)

        while next_id > 900 or next_id < 1:
            action = q.choose_action(observation)
            next_id = env.get_next_grid(action)
        a_sample.append(action)
        spend_time = 0
        trans_time = 0
        seek_time = 0
        done = 0
        while True:
            # 位置的变换,新的转状态为下一个寻客状态，跳过中间的送客过程
            observation_, observation_vector_, spend_time, reward, seek_time, trans_time = env.step(action, spend_time,
                                                                                                    seek_time,
                                                                                                    trans_time)
            r_sample.append(reward)
            observation_ = (observation_[0], env.time.minute)
            next_action = q.choose_action(observation_)
            # 当执行动作next_action后跳出网格区域，重新选择动作
            next_next_id = env.get_next_grid(next_action)
            while next_next_id > 900 or next_next_id < 1:
                next_action = q.choose_action(observation_)
                next_next_id = env.get_next_grid(next_action)
            if spend_time > SPEND_TIME:
                done = 1
            # 更新Q表
            q.learn_on_policy(observation, action, reward, observation_, next_action, done)
            # 更新状态
            observation = observation_
            action = next_action
            # 计算本轮次的收益之和
            if done:
                print("sarsa_strategy Episode {} finished after {} mins".format(i_episode, spend_time))
                score.append(sum(r_sample))
                print("sum revenue", sum(r_sample))
                n_orders.append(sum(i > 0 for i in r_sample))
                if n_orders[i_episode]:
                    peer_order_revenue.append(score[i_episode] / n_orders[i_episode])
                else:
                    peer_order_revenue.append(0)
                peer_hour_revenue.append(score[i_episode] / (spend_time / 60))
                sum_seek_time.append(seek_time)
                sum_trans_time.append(trans_time)
                break
    return [score, n_orders, peer_order_revenue, peer_hour_revenue, sum_seek_time, sum_trans_time]


class SarsaLambdaTable_new_mdp:
    def __init__(self, learning_rate=LEARNING_RATE, gama=GAMA, epsilon=EPSILON, lamb=Lambda, q_table=None):
        # backward view, eligibility trace.
        self.lambda_ = lamb
        self.epsilon = epsilon
        self.gamma = gama
        self.lr = learning_rate
        self.eligibility_trace = self.create_q_table()
        self.q_table = q_table
        if self.q_table is None:
            self.q_table = self.create_q_table()

    # 初始化Q表， Q表都存储在一个数组内。Q表为所有时间的q表集合
    def create_q_table(self):
        index = pd.MultiIndex.from_product([[i for i in range(1, 901)],
                                            [i for i in range(1, 49)],
                                            [i for i in range(0, 30)]])
        q_table = pd.DataFrame(columns=range(1, 10), data=np.zeros([1296000, 9]),
                               index=index)
        q_table.index.names = ["grid_id", "period", "minute"]
        return q_table

    # 随机选择qt中最大值中的一个返回
    def random_select_max(self, qt):
        max_value = max(qt)
        max_index = []
        for i in range(9):
            if qt[i] == max_value:
                max_index.append(i + 1)
        return np.random.choice(max_index)

    # 在时间t状态l是动作选择，返回动作
    def choose_action(self, state):
        epsilon = self.epsilon
        actions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if np.random.rand() < epsilon:
            # 贪婪策略随机探索动作
            action = np.random.choice(actions)
        else:

            # qt[index, acition]    index代表网格的索引， action 代表 动作
            qt = np.array(self.q_table.loc[state].values)
            # action = np.argmax(qt) + 1
            action = self.random_select_max(qt)
        return action

    # 每个轮次将e_table重新初始化
    def reset_e_table(self):
        self.eligibility_trace = 0 * self.eligibility_trace

    def learn_on_policy(self, s, a, r, s_, a_, done):
        q_predict = self.q_table.loc[s, a]
        if done:
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict

        # increase trace amount for visited state-action pair

        # Method 1:
        self.eligibility_trace.loc[s, a] += 1
        # Method 2:
        # self.eligibility_trace.loc[s, :] *= 0
        # self.eligibility_trace.loc[s, a] = 1
        # Q update
        self.q_table += self.lr * error * self.eligibility_trace
        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma * self.lambda_


# 没有学习，随机探索策略
def update_random(env, episode=Max_episode):
    """
        :param env: 环境类
        :return: 返回最后一个轮次的【动作，状态，时间，收益】，以及所有没轮次的【总收益， 订单数，单位订单收益，时间收益】

    """
    # 本轮次的总收益
    score = []
    # 本轮次的接单数
    n_orders = []
    # 本轮次的每单平均收益
    peer_order_revenue = []
    # 本轮次单位是时间的收益（每60分钟）
    peer_hour_revenue = []
    # 每一轮次的有效载客时间,寻客时间
    sum_trans_time = []
    sum_seek_time = []
    for i_episode in range(episode):
        # 每轮次随机开始地点
        start_id, start_hour, start_min_half, daytype = env.start_state
        star_place = np.random.randint(1, 901)
        env.environment_change(star_place, start_hour, start_min_half, daytype)
        env.reset()
        r_sample = []
        t = 0
        trans_time = 0
        seek_time = 0
        done = 0
        while True:
            env.render()
            a = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            # 当执行动作action后跳出网格区域，重新选择动作
            next_id = env.get_next_grid(a)
            while next_id > 900 or next_id < 1:
                a = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
                next_id = env.get_next_grid(a)
            s_, s_vetor, t, r, seek_time, trans_time = env.step(a, t, seek_time, trans_time)
            r_sample.append(r)
            if t > SPEND_TIME:
                done = 1
            if done:
                print("random_strategy Episode {} finished after {} mins".format(i_episode, t))
                score.append(sum(r_sample))
                print("sum revenue", sum(r_sample))
                n_orders.append(sum(i > 0 for i in r_sample))
                if n_orders[i_episode]:
                    peer_order_revenue.append(score[i_episode] / n_orders[i_episode])
                else:
                    peer_order_revenue.append(0)
                peer_hour_revenue.append(score[i_episode] / (t / 60))
                sum_seek_time.append(seek_time)
                sum_trans_time.append(trans_time)
                break
    return [score, n_orders, peer_order_revenue, peer_hour_revenue, sum_seek_time, sum_trans_time]


# 前往附近热点区域
def update_hot_spot(hot_spot, env, episode=Max_episode):
    """
        :param env: 环境类
        :return: 返回最后一个轮次的【动作，状态，时间，收益】，以及所有没轮次的【总收益， 订单数，单位订单收益，时间收益】

    """
    # 本轮次的总收益
    score = []
    # 本轮次的接单数
    n_orders = []
    # 本轮次的每单平均收益
    peer_order_revenue = []
    # 本轮次单位是时间的收益（每60分钟）
    peer_hour_revenue = []
    # 每一轮次的有效载客时间,寻客时间
    sum_trans_time = []
    sum_seek_time = []
    for i_episode in range(episode):
        # 每轮次随机开始地点
        start_id, start_hour, start_min_half, daytype = env.start_state
        star_place = np.random.randint(1, 901)
        env.environment_change(star_place, start_hour, start_min_half, daytype)
        env.reset()
        r_sample = []
        spend_time = 0
        trans_time = 0
        seek_time = 0
        done = 0
        # 初始化观测值
        observation = env.start_state
        while True:
            # 先依靠Q_learning探索和利用产生一个动作
            action = hot_spot.choose_action(observation[0])
            # 当执行动作action后跳出网格区域，重新选择动作
            next_id = env.get_next_grid(action)
            while next_id > 900 or next_id < 1:
                action = hot_spot.choose_action(observation[0])
                next_id = env.get_next_grid(action)
            # 位置的变换,新的转状态为下一个寻客状态，跳过中间的送客过程
            observation_, observation_vector_, spend_time, reward, seek_time, trans_time = env.step(action, spend_time,
                                                                                                    seek_time,
                                                                                                    trans_time)
            r_sample.append(reward)
            if spend_time > SPEND_TIME:
                done = 1
            # 更新状态
            observation = observation_
            # 计算本轮次的收益之和
            if done:
                print("hot_spot_strategy Episode {} finished after {} mins".format(i_episode, spend_time))
                score.append(sum(r_sample))
                print("sum revenue", sum(r_sample))
                n_orders.append(sum(i > 0 for i in r_sample))
                if n_orders[i_episode]:
                    peer_order_revenue.append(score[i_episode] / n_orders[i_episode])
                else:
                    peer_order_revenue.append(0)
                peer_hour_revenue.append(score[i_episode] / (spend_time / 60))
                sum_seek_time.append(seek_time)
                sum_trans_time.append(trans_time)
                break
    return [score, n_orders, peer_order_revenue, peer_hour_revenue, sum_seek_time, sum_trans_time]


# q_leaning
def update_q_leaning(q, env, episode=Max_episode):
    """
    :param q: q表格
    :param env: 环境类
    :return: 返回最后一个轮次的【动作，状态，时间，收益】，以及所有没轮次的【总收益， 订单数，单位订单收益，时间收益】
    """
    # 本轮次的总收益
    score = []
    # 本轮次的接单数
    n_orders = []
    # 本轮次的每单平均收益
    peer_order_revenue = []
    # 本轮次单位是时间的收益（每60分钟）
    peer_hour_revenue = []
    # 每一轮次的有效载客时间,寻客时间
    sum_trans_time = []
    sum_seek_time = []
    for i_episode in range(episode):
        # 每轮次随机开始地点
        start_id, start_hour, start_min_half, daytype = env.start_state
        star_place = np.random.randint(1, 901)
        env.environment_change(star_place, start_hour, start_min_half, daytype)
        # 每轮次初始化收益列表
        r_sample = []
        # 重置网格环境
        env.reset()
        # 初始化观测值
        observation = env.start_state
        # 每轮次记时
        spend_time = 0
        trans_time = 0
        seek_time = 0
        done = 0
        while True:
            # 先依靠Q_learning探索和利用产生一个动作
            action = q.choose_action(observation)
            # 当执行动作action后跳出网格区域，重新选择动作
            next_id = env.get_next_grid(action)
            while next_id > 900 or next_id < 1:
                action = q.choose_action(observation)
                next_id = env.get_next_grid(action)
            # 位置的变换,新的转状态为下一个寻客状态，跳过中间的送客过程
            # 记录这一步的寻客时间和送客时间
            current_seek_time = seek_time
            current_trans_time = trans_time
            observation_, observation_vector_, spend_time, reward, seek_time, trans_time = env.step(action, spend_time,
                                                                                                    seek_time,
                                                                                                    trans_time)
            step_seek_time = seek_time - current_seek_time
            step_trans_time = trans_time - current_trans_time
            r_sample.append(reward)
            if spend_time > SPEND_TIME:
                done = 1
            # 更新Q表
            # 第一种q值（订单收益）
            q.learn(observation, action, reward, observation_, done)
            # 第二种q值（单位时间收益）
            # q.learn(observation, action, reward/(step_seek_time + step_trans_time), observation_, done)
            # 第三种q值（利用率）
            # q.learn(observation, action, step_trans_time/(step_seek_time + step_trans_time), observation_, done)
            # 更新状态
            observation = observation_
            # 计算本轮次的收益之和
            if done:
                print("q_learning_strategy Episode {} finished after {} mins".format(i_episode, spend_time))
                score.append(sum(r_sample))
                print("sum revenue", sum(r_sample))
                n_orders.append(sum(i > 0 for i in r_sample))
                if n_orders[i_episode]:
                    peer_order_revenue.append(score[i_episode] / n_orders[i_episode])
                else:
                    peer_order_revenue.append(0)
                peer_hour_revenue.append(score[i_episode] / (spend_time / 60))
                sum_seek_time.append(seek_time)
                sum_trans_time.append(trans_time)
                break
    return [score, n_orders, peer_order_revenue, peer_hour_revenue, sum_seek_time, sum_trans_time]


# Sarsa
def update_sarsa_leaning(q, env, episode=Max_episode):
    # 本轮次的总收益
    score = []
    # 本轮次的接单数
    n_orders = []
    # 本轮次的每单平均收益
    peer_order_revenue = []
    # 本轮次单位是时间的收益（每60分钟）
    peer_hour_revenue = []
    # 每一轮次的有效载客时间和寻客时间
    sum_trans_time = []
    sum_seek_time = []
    for i_episode in range(episode):
        # 每轮次随机开始地点
        start_id, start_hour, start_min_half, daytype = env.start_state
        star_place = np.random.randint(1, 901)
        env.environment_change(star_place, start_hour, start_min_half, daytype)
        a_sample = []
        r_sample = []
        # 重置网格环境
        env.reset()
        # 获得开始状态
        observation = env.start_state
        # 先依靠q表探索产生一个动作
        action = q.choose_action(observation)
        # 当执行动作action后跳出网格区域，重新选择动作
        next_id = env.get_next_grid(action)
        while next_id > 900 or next_id < 1:
            action = q.choose_action(observation)
            next_id = env.get_next_grid(action)
        a_sample.append(action)
        spend_time = 0
        trans_time = 0
        seek_time = 0
        done = 0
        while True:
            # 位置的变换,新的转状态为下一个寻客状态，跳过中间的送客过程
            observation_, observation_vector_, spend_time, reward, seek_time, trans_time = env.step(action, spend_time,
                                                                                                    seek_time,
                                                                                                    trans_time)
            r_sample.append(reward)
            next_action = q.choose_action(observation_)
            # 当执行动作next_action后跳出网格区域，重新选择动作
            next_next_id = env.get_next_grid(next_action)
            while next_next_id > 900 or next_next_id < 1:
                next_action = q.choose_action(observation_)
                next_next_id = env.get_next_grid(next_action)
            if spend_time > SPEND_TIME:
                done = 1
            # 更新Q表
            q.learn_on_policy(observation, action, reward, observation_, next_action, done)
            # 更新状态
            observation = observation_
            action = next_action
            # 计算本轮次的收益之和
            if done:
                print("sarsa_strategy Episode {} finished after {} mins".format(i_episode, spend_time))
                score.append(sum(r_sample))
                print("sum revenue", sum(r_sample))
                n_orders.append(sum(i > 0 for i in r_sample))
                if n_orders[i_episode]:
                    peer_order_revenue.append(score[i_episode] / n_orders[i_episode])
                else:
                    peer_order_revenue.append(0)
                peer_hour_revenue.append(score[i_episode] / (spend_time / 60))
                sum_seek_time.append(seek_time)
                sum_trans_time.append(trans_time)
                break
    return [score, n_orders, peer_order_revenue, peer_hour_revenue, sum_seek_time, sum_trans_time]


class Q:
    def __init__(self,
                 epsilon=EPSILON,
                 learning_rate=LEARNING_RATE,
                 gamma=GAMA,
                 q_table=None):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = q_table
        if self.q_table is None:
            self.create_q_table()

    # 初始化Q表， Q表都存储在一个数组内。Q表为所有时间的q表集合
    def create_q_table(self):
        index = pd.MultiIndex.from_product([[i for i in range(1, 901)],
                                            [i for i in range(0, 24)],
                                            [0, 1],
                                            [0, 1]])
        q_table = pd.DataFrame(columns=range(1, 10), data=np.zeros([86400, 9]),
                               index=index)
        q_table.index.names = ["grid_id", "hour", "min_half", "daytype"]
        self.q_table = q_table

    # 随机选择qt中最大值中的一个返回
    def random_select_max(self, qt):
        max_value = max(qt)
        max_index = []
        for i in range(9):
            if qt[i] == max_value:
                max_index.append(i + 1)
        return np.random.choice(max_index)

    # 在时间t状态l是动作选择，返回动作
    def choose_action(self, state):
        epsilon = self.epsilon
        actions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if np.random.rand() < epsilon:
            # 贪婪策略随机探索动作
            action = np.random.choice(actions)
        else:

            # qt[index, acition]    index代表网格的索引， action 代表 动作
            qt = np.array(self.q_table.loc[state].values)
            # action = np.argmax(qt) + 1
            action = self.random_select_max(qt)
        return action

    # off_policy策略
    def learn(self, state, action, reward, next_state, done):
        learning_rate = self.learning_rate  # 学习率
        gama = self.gamma  # 折扣因子
        #  二维数据的贝尔曼方程的运算。
        # 估计q值
        q_predict = self.q_table.at[state, action]
        # 实际q值
        next_state_value = self.q_table.loc[next_state].values
        q_target = reward + (1 - done) * gama * max(next_state_value)
        # 记录当智能体执行一个action时候所获得的的收益。
        #  具体步骤，找回那个时间下的状态，所选择的动作
        self.q_table.at[state, action] += learning_rate * (q_target - q_predict)

    # on_policy策略
    def learn_on_policy(self, state, action, reward, next_state, next_action, done):
        learning_rate = self.learning_rate  # 学习率
        gama = self.gamma  # 折扣因子
        #  二维数据的贝尔曼方程的运算。
        # 估计q值
        q_predict = self.q_table.at[state, action]
        # 实际q值
        q_target = reward + (1 - done) * gama * self.q_table.at[next_state, next_action]
        self.q_table.at[state, action] += learning_rate * (q_target - q_predict)


# 热点区域
class HotSpot:
    def __init__(self):
        self.table = pd.read_csv(r"C:\data\代码\深度强化环境\最终环境\mdp环境源数据\hot_spot.csv", index_col=0)

    # 随机选择qt中最大值中的一个返回
    def random_select_max(self, qt):
        max_value = max(qt)
        max_index = []
        for i in range(9):
            if qt[i] == max_value:
                max_index.append(i + 1)
        return np.random.choice(max_index)

    # 在时间t状态l是动作选择，返回动作
    def choose_action(self, state):
        # qt[index, acition]    index代表网格的索引， action 代表 动作
        qt = np.array(self.table.loc[state].values)
        action = self.random_select_max(qt)
        return action


# 随机探索策略
def run_random(env):
    """
        :param env: 环境类
        :return: 返回【总收益， 订单数，寻客时间，载客时间】

    """
    r_sample = []
    t = 0
    trans_time = 0
    seek_time = 0
    done = 0
    while True:
        a = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        # 当执行动作action后跳出网格区域，重新选择动作
        next_id = env.get_next_grid(a)
        while next_id > 900 or next_id < 1:
            a = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            next_id = env.get_next_grid(a)
        s_, s_vetor, t, r, seek_time, trans_time = env.step(a, t, seek_time, trans_time)
        r_sample.append(r)
        if t > SPEND_TIME:
            done = 1
            # 计算本轮次的收益之和
        if done:
            score = sum(r_sample)
            n_orders = (sum(i > 0 for i in r_sample))
            break
    return [score, n_orders, seek_time, trans_time]


# 热点区域
def run_hot_spot(hot_spot, env):
    """
        :param env: 环境类
        :return: 返回最后一个轮次的【动作，状态，时间，收益】，以及所有没轮次的【总收益， 订单数，单位订单收益，时间收益】

    """
    env.reset()
    r_sample = []
    spend_time = 0
    trans_time = 0
    seek_time = 0
    done = 0
    # 初始化观测值
    observation = env.start_state
    while True:
        # 先依靠Q_learning探索和利用产生一个动作
        action = hot_spot.choose_action(observation[0])
        # 当执行动作action后跳出网格区域，重新选择动作
        next_id = env.get_next_grid(action)
        while next_id > 900 or next_id < 1:
            action = hot_spot.choose_action(observation[0])
            next_id = env.get_next_grid(action)
        # 位置的变换,新的转状态为下一个寻客状态，跳过中间的送客过程
        observation_, observation_vector_, spend_time, reward, seek_time, trans_time = env.step(action, spend_time,
                                                                                                seek_time,
                                                                                                trans_time)
        r_sample.append(reward)
        if spend_time > SPEND_TIME:
            done = 1
        # 更新状态
        observation = observation_
        # 计算本轮次的收益之和
        if done:
            score = sum(r_sample)
            n_orders = sum(i > 0 for i in r_sample)
            break
    return [score, n_orders, seek_time, trans_time]


# 司机运用训练好的q表，得到一个轮次的收益，寻客时间，载客时间
def run_q_leaning(q, env):
    """
    :param q: q表格
    :param env: 环境类
    :return: 【总收益， 订单数，单位订单收益，时间收益】
    """
    # 记录每一步的收益
    r_sample = []
    # 本轮次的总收益
    score = 0
    # 本轮次的接单数
    n_orders = 0
    # 初始化观测值
    observation = env.start_state
    # 每轮次记时
    spend_time = 0
    trans_time = 0
    seek_time = 0
    done = 0
    while True:
        # 先依靠Q_learning探索和利用产生一个动作
        action = q.choose_action(observation)
        # 当执行动作action后跳出网格区域，重新选择动作
        next_id = env.get_next_grid(action)
        while next_id > 900 or next_id < 1:
            action = q.choose_action(observation)
            next_id = env.get_next_grid(action)
        # 位置的变换,新的转状态为下一个寻客状态，跳过中间的送客过程
        observation_, observation_vector_, spend_time, reward, seek_time, trans_time = env.step(action, spend_time,
                                                                                                seek_time, trans_time)
        r_sample.append(reward)
        if spend_time > SPEND_TIME:
            done = 1
        # 更新Q表
        q.learn(observation, action, reward, observation_, done)
        # 更新状态
        observation = observation_
        # 计算本轮次的收益之和
        if done:
            score = sum(r_sample)
            n_orders = (sum(i > 0 for i in r_sample))
            break
    return [score, n_orders, seek_time, trans_time]


# 运用已有的q表单次sarsa
def run_sarsa_leaning(q, env):
    """
    :param q: q表格
    :param env: 环境类
    :return: 【总收益， 订单数，单位订单收益，时间收益】
    """
    # 记录每一步的收益
    r_sample = []
    # 本轮次的总收益
    score = 0
    # 本轮次的接单数
    n_orders = 0
    # 初始化观测值
    observation = env.start_state
    # 每轮次记时
    spend_time = 0
    trans_time = 0
    seek_time = 0
    done = 0
    # 先依靠q表探索产生一个动作
    action = q.choose_action(observation)
    # 当执行动作action后跳出网格区域，重新选择动作
    next_id = env.get_next_grid(action)
    while next_id > 900 or next_id < 1:
        action = q.choose_action(observation)
        next_id = env.get_next_grid(action)
    while True:
        # 位置的变换,新的转状态为下一个寻客状态，跳过中间的送客过程
        observation_, observation_vector_, spend_time, reward, seek_time, trans_time = env.step(action, spend_time,
                                                                                                seek_time,
                                                                                                trans_time)
        r_sample.append(reward)
        next_action = q.choose_action(observation_)
        # 当执行动作next_action后跳出网格区域，重新选择动作
        next_next_id = env.get_next_grid(next_action)
        while next_next_id > 900 or next_next_id < 1:
            next_action = q.choose_action(observation_)
            next_next_id = env.get_next_grid(next_action)
        if spend_time > SPEND_TIME:
            done = 1
        # 更新Q表
        q.learn_on_policy(observation, action, reward, observation_, next_action, done)
        # 更新状态
        observation = observation_
        action = next_action
        # 计算本轮次的收益之和
        if done:
            score = sum(r_sample)
            n_orders = sum(i > 0 for i in r_sample)
            break
    return [score, n_orders, seek_time, trans_time]


# 定义一个可视化函数，将每轮次的学习的相关指标（总收益，接单数，订单平均收益，单位时间收益）可视化
def visualize(sum_revenue, n_orders, peer_order_revenue, peer_hour_revenue, all_name, episode=Max_episode):
    fig = plt.figure(figsize=(8, 6), dpi=100)  # 画布 长宽8:6,分辨率=80
    # 第一个总收益子图
    ax1 = fig.add_subplot(2, 2, 1)  # 创建2行1列的子图，开始绘制第一个子图ax1
    plt.title("sum_revenue")  # 设置子标题
    plt.ylabel('revenue')  # 设置y轴名称
    plt.xlim((0, episode))  # 设置x轴范围
    plt.ylim((200, 450))  # 设置y轴范围
    xsticks = [i for i in range(0, episode + 1, int(episode / 5))]
    plt.xticks(xsticks)  # 设置x轴刻度
    plt.yticks([200, 250, 300, 350, 400, 450])  # 设置y轴刻度
    plt.plot(sum_revenue)  # 绘制总收益曲线
    average = int(sum(sum_revenue) / episode)
    plt.hlines(average, 0, episode * 0.4, "red")
    plt.text(episode * 0.4, average, "{}".format(average))
    plt.hlines(average, episode * 0.5, episode, "red")
    # plt.legend()  # 添加注解
    # 第二个子图ax2
    ax2 = fig.add_subplot(2, 2, 2)
    plt.title("# of orders")
    plt.ylabel('#')
    plt.plot(n_orders)
    average = int(sum(n_orders) / episode)
    plt.hlines(average, 0, episode * 0.4, "red")
    plt.text(episode * 0.4, average, "{}".format(average))
    plt.hlines(average, episode * 0.5, episode, "red")
    # plt.vlines()             # 设置铅垂线
    # plt.text()               # 添加文字
    # plt.legend()
    # 第三个子图ax3
    ax3 = fig.add_subplot(2, 2, 3)
    plt.title("peer_order_revenue")
    plt.xlabel('episode')
    plt.ylabel('revenue')
    plt.plot(peer_order_revenue)
    average = int(sum(peer_order_revenue) / episode)
    plt.hlines(average, 0, episode * 0.4, "red")
    plt.text(episode * 0.4, average, "{}".format(average))
    plt.hlines(average, episode * 0.5, episode, "red")
    # plt.legend()
    # 第四个子图ax3
    ax4 = fig.add_subplot(2, 2, 4)
    plt.title("peer_hour_revenue")
    plt.xlabel('episode')
    plt.ylabel('revenue')
    plt.xlim((0, episode))  # 设置x轴范围
    plt.ylim((40, 140))  # 设置y轴范围
    xsticks = [i for i in range(0, episode + 1, int(episode / 5))]
    plt.xticks(xsticks)  # 设置x轴刻度
    plt.yticks([40, 60, 80, 100, 120, 140])  # 设置y轴刻度
    plt.plot(peer_hour_revenue)
    average = int(sum(peer_hour_revenue) / episode)
    plt.hlines(average, 0, episode * 0.4, "red")
    plt.text(episode * 0.4, average, "{}".format(average))
    plt.hlines(average, episode * 0.5, episode, "red")
    # plt.legend()
    plt.suptitle(all_name)
    plt.show()


# 根据强化学习每一轮次的结果计算出收益率，收入效率，利用率
def caculate_evaluate_index(dataFram):
    dataFram["average_profit"] = dataFram["score"] / dataFram["sum_trans_time"]
    dataFram.loc[dataFram.index[dataFram["sum_trans_time"] == 0], "average_profit"] = 0
    dataFram["revenue_efficiency"] = dataFram["score"] / (dataFram["sum_seek_time"] + dataFram["sum_trans_time"])
    dataFram["utilization_rate"] = dataFram["sum_trans_time"] / (dataFram["sum_seek_time"] + dataFram["sum_trans_time"])
    return dataFram


# 将强化学习算法所返回的所有轮次记录进行处理
def recording(episode_data):
    dataFram = pd.DataFrame(np.array(episode_data).T,
                            columns=["score", 'n_orders', 'peer_order_revenue', 'peer_hour_revenue',
                                     'sum_seek_time', 'sum_trans_time'])
    dataFram["average_profit"] = dataFram["score"] / dataFram["sum_trans_time"]
    dataFram.loc[dataFram.index[dataFram["sum_trans_time"] == 0], "average_profit"] = 0
    dataFram["revenue_efficiency"] = dataFram["score"] / (dataFram["sum_seek_time"] + dataFram["sum_trans_time"])
    dataFram["utilization_rate"] = dataFram["sum_trans_time"] / (dataFram["sum_seek_time"] + dataFram["sum_trans_time"])
    return dataFram


# 训练Q表，两种q表，一种不是动态定价(q_dp为0)，一种是动态定价（q_dp == 1）
def train_q_table(output_q_table_path, q_dp):
    q = Q()
    hour = [8, 12, 17]
    time_list = [5000]
    for ho in hour:
        for time in time_list:
            #
            env = Env(250, ho, 0, 1, 1, q_dp)
            last, episode = update_q_leaning(q, env, time)
            # q_dp = pd.DataFrame(np.array(episode).T,
            #                     columns=["score", 'n_orders', 'peer_order_revenue', 'peer_hour_revenue',
            #                              'sum_seek_time', 'sum_trans_time'])
    for ho in [17]:
        for time in time_list:
            env = Env(250, ho, 0, 0, 1, q_dp)
            last, episode = update_q_leaning(q, env, time)
            # q_dp = pd.DataFrame(np.array(episode).T,
            #                     columns=["score", 'n_orders', 'peer_order_revenue', 'peer_hour_revenue',
            #                              'sum_seek_time', 'sum_trans_time'])
            # q_dp = caculate_evaluate_index(q_dp)
    q.q_table.to_csv(output_q_table_path)


# 计算不同参数不同算法对应不同收敛轮次，不同收敛收益
# 输入训练时的轮次文件地址，返回累计间隔的轮次结果
def judge_converaged(episode_resule_path, result_output_path):
    
    episode_data = pd.read_csv(episode_resule_path)
    interval_episode = 10
    episode_data = caculate_evaluate_index(episode_data)
    summary_length = round(len(episode_data) / interval_episode)
    summary_episode = pd.DataFrame(np.zeros([summary_length, 9]),
                                   columns=["score", 'n_orders', 'peer_order_revenue', 'peer_hour_revenue',
                                            'sum_seek_time', 'sum_trans_time', "average_profit",
                                            "revenue_efficiency", "utilization_rate"])
    n_episode = [i for i in range(interval_episode, len(episode_data) + 1, interval_episode)]
    for i in range(1, summary_length + 1):
        summary_episode.loc[i - 1] = episode_data[0:interval_episode * i].mean()
    summary_episode.insert(0, "n_episodes", n_episode)
    summary_episode.to_csv(result_output_path)


# 使用训练了五千次的q表，让司机在不同的时段，随机不同的其实区域跑2000次
def driver_random_place_run_q_table(q_table_path, result_path, dp_reward):
    q_ = pd.read_csv(q_table_path, index_col=["grid_id", "hour", "min_half", "daytype"])
    q_.rename(columns={"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}, inplace=True)
    q = Q(q_table=q_)
    # 先定义环境，只定义了一个环境，后续环境的改变只需再次环境的基础上做出简单修改其实状态
    env = Env(250, 8, 0, 0, 1, dp_reward)
    # 工作日
    hour = [8, 12, 17]
    for ho in hour:
        q_dp = pd.DataFrame(np.zeros([2000, 4]),
                            columns=["score", 'n_orders',
                                     'sum_seek_time', 'sum_trans_time'])
        for time in range(2000):
            start_id = np.random.randint(1, 901)
            env.environment_chage(start_id, ho, 0, 1)
            result = run_q_leaning(q, env)
            q_dp.at[time, "score"] = result[0]
            q_dp.at[time, "n_orders"] = result[1]
            q_dp.at[time, "sum_seek_time"] = result[2]
            q_dp.at[time, "sum_trans_time"] = result[3]
            time += 1
        q_dp = caculate_evaluate_index(q_dp)
        q_dp.to_csv(result_path + "/fri_" + str(ho) + ".csv")
    # 周末
    hour = [17]
    for ho in hour:
        q_dp = pd.DataFrame(np.zeros([2000, 4]),
                            columns=["score", 'n_orders',
                                     'sum_seek_time', 'sum_trans_time'])
        for time in range(2000):
            start_id = np.random.randint(1, 901)
            env.environment_chage(start_id, ho, 0, 0)
            result = run_q_leaning(q, env)
            q_dp.at[time, "score"] = result[0]
            q_dp.at[time, "n_orders"] = result[1]
            q_dp.at[time, "sum_seek_time"] = result[2]
            q_dp.at[time, "sum_trans_time"] = result[3]
            time += 1
        q_dp = caculate_evaluate_index(q_dp)
        q_dp.to_csv(result_path + "/sat_" + str(ho) + ".csv")


if __name__ == '__main__':
    # 对程序运行时间计时
    start_time = time_clock.perf_counter()

     env = Env(250, 8, 0, 1, 2, 1)
    sarsa_q = SarsaLambdaTable(lamb=0.5)
    recording(update_sarsa_lambda_leaning(sarsa_q, env, 5000)).to_csv(
        r"../sl_new.csv")
    judge_converaged(r"../sl_new.csv",
                     r"../sl_new_summary.csv")
    env = Env(250, 8, 0, 1, 2, 2)
    q_table = Q(epsilon=0.1, learning_rate=0.01, gamma=0.5)
    recording(update_q_leaning(q_table, env, 5000)).to_csv(
        r"../q_ad.csv")
    judge_converaged(r"../q_ad.csv",
                     r"../q_ad_summary.csv")

    end_time = time_clock.perf_counter()
    print("运行时间为：", round(end_time - start_time), "seconds")

pass
