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


# Sarsa(lambda)
def update_new_mdp_sarsa_lambda_leaning(q, env, episode=Max_episode):
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

        observation = (star_place, 2 * start_hour + start_min_half + 1, 0)
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
            observation_ = (
            observation_[0], int(env.time.hour) * 2 + int(env.time.minute / 30) + 1, int(env.time.minute) % 30)
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


# 深度强化算法
def update_dqn(DQN, env, episode=Max_episode):
    """
    :param DQN: 深度强化网络环境
    :param env: 网格环境（step）
    :param episode：轮次数
    :return 返回最后一个轮次的【动作，状态，时间，收益】，以及所有没轮次的【总收益， 订单数，单位订单收益，时间收益】
    """
    # 重置网格环境
    # 本轮次的总收益
    score = []
    # 本轮次的接单数
    n_orders = []
    # 本轮次的每单平均收益
    peer_order_revenue = []
    # 本轮次单位是时间的收益（每60分钟）
    peer_hour_revenue = []
    DQN_agent = DQN
    # 每一轮次的有效载客时间和寻客时间
    sum_trans_time = []
    sum_seek_time = []
    for i_episode in range(episode):
        # 每轮次初始化收益列表
        r_sample = []
        # 重置网格环境
        env.reset()
        # Q网络的输入向量
        if env.q_type == 1:
            q_observation = env.tran_q_vector1(env.start_state)
        elif env.q_type == 2:
            q_observation = env.tran_q_vector2(env.start_state)
        elif env.q_type == 3:
            q_observation = env.tran_q_vector3(env.start_state)
        elif env.q_type == 4:
            q_observation = env.tran_q_vector4(env.start_state)
        elif env.q_type == 5:
            q_observation = env.tran_q_vector5(env.start_state)
        elif env.q_type == 6:
            q_observation = env.tran_q_vector6(env.start_state)
        elif env.q_type == 7:
            q_observation = env.tran_q_vector7()
        # 每轮次记时
        spend_time = 0
        trans_time = 0
        seek_time = 0
        while True:
            # 完成信号
            done = 0
            # 从动作空间采样，得到动作
            action = DQN_agent.choose_action(q_observation)
            # 当执行动作action后跳出网格区域，重新选择动作
            next_id = env.get_next_grid(action)
            while next_id > 900 or next_id < 1:
                action = DQN_agent.choose_action(q_observation)
                next_id = env.get_next_grid(action)

            prior_seek_time = seek_time
            prior_tran_time = trans_time
            # 在当前状态执行动作action，返回下一个状态， 奖励和累计耗时
            observation_, observation_vector_, spend_time, reward, seek_time, trans_time = env.step(action, spend_time,
                                                                                                    seek_time,
                                                                                                    trans_time)

            cost_time = seek_time + trans_time - prior_seek_time - prior_tran_time
            revenue_efficiency = reward / cost_time
            # DQN的性能受到奖励函数的影响，合理的设计奖励函数，能够DQN的性能变的更好
            if spend_time > SPEND_TIME:
                done = 1
            if env.q_type == 1:
                q_observation_ = env.tran_q_vector1(observation_)
            elif env.q_type == 2:
                q_observation_ = env.tran_q_vector2(observation_)
            elif env.q_type == 3:
                q_observation_ = env.tran_q_vector3(observation_)
            elif env.q_type == 4:
                q_observation_ = env.tran_q_vector4(observation_)
            elif env.q_type == 5:
                q_observation_ = env.tran_q_vector5(observation_)
            elif env.q_type == 6:
                q_observation_ = env.tran_q_vector6(observation_)
            elif env.q_type == 7:
                q_observation_ = env.tran_q_vector7()
            DQN_agent.experience_store(q_observation, action, revenue_efficiency, q_observation_, done)
            # 每一步都进行学习，只是并没有立即更新目标网络参数
            DQN_agent.learn()
            # 更新当前状态
            q_observation = q_observation_
            r_sample.append(reward)
            if done:
                print("Episode {} finished after {} mins".format(i_episode, spend_time))
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


# 普通DQN
class DQN:
    def __init__(self,
                 n_actions,
                 n_features,
                 q_net=None,
                 epsilon=EPSILON,
                 batch_size=BARCH_SIZE,
                 learning_rate=LEARNING_RATE,
                 gamma=GAMA,
                 replace_time=REPLACE_TIME,
                 n_experience_pool=MEMORY_MAX_SIZE):
        self.n_actions = n_actions
        self.n_features = n_features
        self.batch_size = batch_size
        # 学习率
        self.learning_rate = learning_rate
        self.gamma = gamma
        # 0.05-贪心算法
        self.epsilon = epsilon
        # 总共的经验数目
        self.n_experience = 0
        # 经验池大小
        self.n_experience_pool = n_experience_pool
        # 建立经验池 建立一个n_experience_pool行，n_features * 2 + 1 + 1 列的矩阵 s a r s_
        self.experience_pool = pd.DataFrame(np.zeros([self.n_experience_pool, self.n_features * 2 + 1 + 1 + 1]))
        self.experience_pool_index = 0
        self.experience_pool_is_full = False
        # 优化器定义
        self.opt = tf.keras.optimizers.Adam(self.learning_rate)
        # 两个神经网络定义或者加载
        if q_net is None:
            self.q_pred_init()
        else:
            self.q_pred = q_net
        self.q_target_init()
        # 间隔C次，进行参数更新
        self.replace_time = replace_time
        self.now_learn_time = 0

    def loss_f(self, y_true, y_pred):
        return keras.losses.mse(y_true, y_pred)

    # 随机选择qt中最大值中的一个返回
    def random_select_max(self, qt):
        max_value = max(qt)
        max_index = []
        for i in range(9):
            if qt[i] == max_value:
                max_index.append(i + 1)
        return np.random.choice(max_index)

    # 动作为1-9，与以往的环境中step函数对应
    def choose_action(self, state):
        state = np.array(state).reshape(1, self.n_features)
        rand = np.random.rand(1)
        if rand < self.epsilon:
            return np.random.randint(1, self.n_actions + 1)
        else:
            # 采用numpy中argmax来得到最大值所对应的元素下标
            # 这里尤其要注意，不能直接输入s，这里必须输入二维的数组，而不是单独的s
            action_value = self.q_pred.predict(np.array(state))
            return np.argmax(action_value) + 1

    def q_pred_init(self):
        # 神经网络的输入维数就是特征的维数，在强化学习中，就是状态
        # shape=(self.n_features,) shape=（2，）`表示预期的输入将是一批32维向量 32列 n行
        input_features = tf.keras.Input(shape=(self.n_features,), name='input_features')
        # Dense（全连接层）的第一个参数64是他的输出维度,将输入输入层加入到全连接层上
        dense_0 = tf.keras.layers.Dense(128, activation='relu', use_bias=True)(
            input_features)
        dense_1 = tf.keras.layers.Dense(256, activation='relu', use_bias=True)(dense_0)
        dense_2 = tf.keras.layers.Dense(128, activation='relu', use_bias=True)(dense_1)
        out_put = tf.keras.layers.Dense(self.n_actions, name='prediction_q_pred', use_bias=True)(dense_2)
        self.q_pred = tf.keras.Model(inputs=input_features, outputs=out_put)
        # q_table_net

    def q_target_init(self):
        input_features_target = tf.keras.Input(shape=(self.n_features,), name='input_features')
        dense_0_target = tf.keras.layers.Dense(128, activation='relu', use_bias=True)(
            input_features_target)
        dense_1_target = tf.keras.layers.Dense(256, activation='relu', use_bias=True)(
            dense_0_target)
        dense_2_target = tf.keras.layers.Dense(128, activation='relu', use_bias=True)(
            dense_1_target)
        out_put_target = tf.keras.layers.Dense(self.n_actions, name='prediction_q_target', use_bias=True)(
            dense_2_target)
        self.q_target = tf.keras.Model(inputs=input_features_target, outputs=out_put_target)
        # 这里使用copy.deepcopy()进行神经网络的复制是不可取的，导致无法进行预测输出，原因目前还不知道
        # self.q_table_net = copy.deepcopy(self.q_table_net)
        self.q_target.set_weights(self.q_pred.get_weights())
        # print(self.q_table_net.summary())

    def experience_store(self, s, a, r, s_, done):
        # 经验的顺序依次为当前状态， 执行动作，收益， 下一个状态，结束标志
        experience = []
        for i in range(self.n_features * 2 + 2 + 1):
            if i < self.n_features:
                experience.append(s[i])
            elif self.n_features <= i < self.n_features + 1:
                experience.append(a)
            elif self.n_features + 1 <= i < self.n_features + 2:
                experience.append(r)
            elif self.n_features + 2 <= i < self.n_features * 2 + 2:
                experience.append(s_[i - self.n_features - 2])
            else:
                experience.append(done)
        self.experience_pool.loc[self.experience_pool_index] = copy.deepcopy(experience)
        self.experience_pool_index += 1
        self.n_experience += 1
        # print(self.experience_pool_index)
        if self.experience_pool_index == self.n_experience_pool:
            self.experience_pool_is_full = True
            self.experience_pool_index = 0

    def learn(self):
        # 当经验池满的的时候才开始学习
        if self.n_experience < self.n_experience_pool:
            return
        # 注意这里，如果要自己建立数据集的话，最好使用pd中的dataframe，其自带了sample函数，可以进行取样
        # 其他情况可以使用tensorflow的tf.data.Dataset.from_tensor_slices() 来进行数据集的建立，在使用shuffle进行打乱训练。
        data_pool = self.experience_pool.sample(self.batch_size)
        # 这里应该注意把DataFrame格式的转换为ndarray
        s = np.array(data_pool.loc[:, 0:self.n_features - 1])
        a = np.array(data_pool.loc[:, self.n_features], dtype=np.int32)
        r = np.array(data_pool.loc[:, self.n_features + 1])
        s_ = np.array(data_pool.loc[:, self.n_features + 2:self.n_features * 2 + 1])
        done = np.array(data_pool.loc[:, self.n_features * 2 + 2])
        with tf.GradientTape() as Tape:
            # 预测q网络的状态动作价值
            y_pred = self.q_pred(s)
            # 实际动作状态价值.(只改变对应的动作价值)
            y_target = y_pred.numpy()
            # q目标网络中下一状态的动作价值
            q_target = self.q_target(s_).numpy()
            index = np.arange(self.batch_size, dtype=np.int32)
            # 注意这个地方的语法使用, 仅仅更改了动作a所对应的动作价值
            y_target[index, a - 1] = r + (1 - done) * self.gamma * np.max(q_target, axis=1)
            loss_val = tf.keras.losses.mse(y_target, y_pred)
        gradients = Tape.gradient(loss_val, self.q_pred.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.q_pred.trainable_variables))
        # 首先选出Q-target网络的最大值
        self.now_learn_time += 1
        if self.now_learn_time == self.replace_time:
            self.replace_param()
            self.now_learn_time = 0

    def replace_param(self):
        print("replace the param")
        self.q_target.set_weights(self.q_pred.get_weights())


# DDQN
class DDQN:
    def __init__(self,
                 n_actions,
                 n_features,
                 q_net=None,
                 epsilon=Exploration_rate,
                 batch_size=BARCH_SIZE,
                 learning_rate=LEARNING_RATE,
                 gamma=GAMA,
                 replace_time=REPLACE_TIME,
                 n_experience_pool=MEMORY_MAX_SIZE):
        self.n_actions = n_actions
        self.n_features = n_features
        self.batch_size = batch_size
        # 学习率
        self.learning_rate = learning_rate
        self.gamma = gamma
        # 0.05-贪心算法
        self.epsilon = epsilon
        # 经验池大小
        self.n_experience_pool = n_experience_pool
        # 经验的总数目
        self.n_experience = 0
        # 建立经验池 建立一个n_experience_pool行，n_features * 2 + 1 + 1 列的矩阵 s a r s_
        self.experience_pool = pd.DataFrame(np.zeros([self.n_experience_pool, self.n_features * 2 + 1 + 1 + 1]))
        self.experience_pool_index = 0
        self.experience_pool_is_full = False
        self.net_init_random_seed = tf.keras.initializers.glorot_uniform(seed=1)
        # 优化器定义
        self.opt = tf.keras.optimizers.Adam(self.learning_rate)
        # 两个神经网络定义或者加载
        if q_net is None:
            self.q_pred_init()
        else:
            self.q_pred = q_net
        self.q_target_init()
        # 间隔C次，进行参数更新
        self.replace_time = replace_time
        self.now_learn_time = 0

    def loss_f(self, y_true, y_pred):
        return keras.losses.mse(y_true, y_pred)

    # 随机选择qt中最大值中的一个返回
    def random_select_max(self, qt):
        max_value = max(qt)
        max_index = []
        for i in range(9):
            if qt[i] == max_value:
                max_index.append(i + 1)
        return np.random.choice(max_index)

    # 动作为1-9，与以往的环境中step函数对应
    def choose_action(self, state):
        state = np.array(state).reshape(1, self.n_features)
        rand = np.random.rand(1)
        if rand < self.epsilon:
            self.epsilon = self.epsilon * Explore_decay_rate
            return np.random.randint(1, self.n_actions + 1)
        else:
            # 采用numpy中argmax来得到最大值所对应的元素下标
            # 这里尤其要注意，不能直接输入s，这里必须输入二维的数组，而不是单独的s
            action_value = self.q_pred.predict(np.array(state))
            return np.argmax(action_value) + 1

    # q_pred
    def q_pred_init(self):
        # 神经网络的输入维数就是特征的维数，在强化学习中，就是状态
        # shape=(self.n_features,) shape=（2，）`表示预期的输入将是一批
        input_features = tf.keras.Input(shape=(self.n_features,), name='input_features')
        # Dense（全连接层）的第一个参数32是他的输出维度,将输入输入层加入到全连接层上
        dense_0 = tf.keras.layers.Dense(128, activation='relu', use_bias=True)(
            input_features)
        dense_1 = tf.keras.layers.Dense(256, activation='relu', use_bias=True)(dense_0)
        dense_2 = tf.keras.layers.Dense(64, activation='relu', use_bias=True)(dense_1)
        out_put = tf.keras.layers.Dense(self.n_actions, name='prediction_q_pred')(dense_2)
        self.q_pred = tf.keras.Model(inputs=input_features, outputs=out_put)

    # q_table_net
    def q_target_init(self):
        input_features_target = tf.keras.Input(shape=(self.n_features,), name='input_features')
        dense_0_target = tf.keras.layers.Dense(128, activation='relu', use_bias=True)(
            input_features_target)
        dense_1_target = tf.keras.layers.Dense(256, activation='relu', use_bias=True)(
            dense_0_target)
        dense_2_target = tf.keras.layers.Dense(64, activation='relu', use_bias=True)(
            dense_1_target)
        out_put_target = tf.keras.layers.Dense(self.n_actions, name='prediction_q_target')(dense_2_target)
        self.q_target = tf.keras.Model(inputs=input_features_target, outputs=out_put_target)
        # 这里使用copy.deepcopy()进行神经网络的复制是不可取的，导致无法进行预测输出，原因目前还不知道
        # self.q_table_net = copy.deepcopy(self.q_table_net)
        self.q_target.set_weights(self.q_pred.get_weights())
        # print(self.q_table_net.summary())

    def experience_store(self, s, a, r, s_, done):
        # 经验的顺序依次为当前状态， 执行动作，收益， 下一个状态，结束标志
        experience = []
        for i in range(self.n_features * 2 + 2 + 1):
            if i < self.n_features:
                experience.append(s[i])
            elif self.n_features <= i < self.n_features + 1:
                experience.append(a)
            elif self.n_features + 1 <= i < self.n_features + 2:
                experience.append(r)
            elif self.n_features + 2 <= i < self.n_features * 2 + 2:
                experience.append(s_[i - self.n_features - 2])
            else:
                experience.append(done)
        self.experience_pool.loc[self.experience_pool_index] = copy.deepcopy(experience)
        self.experience_pool_index += 1
        self.n_experience += 1
        if self.experience_pool_index == self.n_experience_pool:
            self.experience_pool_is_full = True
            self.experience_pool_index = 0

    def learn(self):
        # 当经验池满的的时候才开始学习
        if not self.experience_pool_is_full:
            return
        # 注意这里，如果要自己建立数据集的话，最好使用pd中的dataframe，其自带了sample函数，可以进行取样
        # 其他情况可以使用tensorflow的tf.data.Dataset.from_tensor_slices() 来进行数据集的建立，在使用shuffle进行打乱训练。
        data_pool = self.experience_pool.sample(self.batch_size)
        # 这里应该注意把DataFrame格式的转换为ndarray
        s = np.array(data_pool.loc[:, 0:self.n_features - 1])
        a = np.array(data_pool.loc[:, self.n_features], dtype=np.int32)
        r = np.array(data_pool.loc[:, self.n_features + 1])
        s_ = np.array(data_pool.loc[:, self.n_features + 2:self.n_features * 2 + 1])
        done = np.array(data_pool.loc[:, self.n_features * 2 + 2])
        # 这里是DQN与DDQN的唯一区别之处
        with tf.GradientTape() as Tape:
            # 预测网络的动作状态价值
            y_pred = self.q_pred(s)
            # 实际的状态动作价值
            y_target = y_pred.numpy()
            # q预测网络中的下一状态动作价值，预测下一状态价值最大的动作
            q = self.q_pred(s_).numpy()
            arg_max_a = np.argmax(q, axis=1)
            # q目标网络中下一状态的动作价值
            q_target = self.q_target(s_).numpy()
            index = np.arange(self.batch_size, dtype=np.int32)
            y_target[index, a - 1] = r + (1 - done) * self.gamma * q_target[index, arg_max_a]
            loss_val = tf.keras.losses.mse(y_target, y_pred)
        gradients = Tape.gradient(loss_val, self.q_pred.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.q_pred.trainable_variables))

        self.now_learn_time += 1
        if self.now_learn_time == self.replace_time:
            self.replace_param()
            self.now_learn_time = 0

    def replace_param(self):
        print("replace the param")
        self.q_target.set_weights(self.q_pred.get_weights())


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


# 司机运用训练好的q网，得到一个轮次的收益，寻客时间，载客时间
def run_dqn(dqn, env):
    """
        :param DQN: 深度强化网络环境
        :param env: 网格环境（step）
        :return 返回【总收益， 订单数，寻客时间，载客时间】
        """
    # 每轮次记时
    spend_time = 0
    trans_time = 0
    seek_time = 0
    done = 0
    DQN_agent = dqn
    # 每轮次初始化收益列表
    r_sample = []
    # 初始化观测值
    observation = env.start_state
    # 初始化观测向量
    observation_vector = env.start_state_vector
    # Q网络的输入向量
    if env.q_type == 1:
        q_observation = env.tran_q_vector1(env.start_state)
    elif env.q_type == 2:
        q_observation = env.tran_q_vector2(env.start_state)
    elif env.q_type == 3:
        q_observation = env.tran_q_vector3(env.start_state)
    elif env.q_type == 4:
        q_observation = env.tran_q_vector4(env.start_state)
    elif env.q_type == 5:
        q_observation = env.tran_q_vector5(env.start_state)
    while True:
        # 从动作空间采样，得到动作
        action = DQN_agent.choose_action(q_observation)
        # 当执行动作action后跳出网格区域，重新选择动作
        next_id = env.get_next_grid(action)
        while next_id > 900 or next_id < 1:
            action = DQN_agent.choose_action(q_observation)
            next_id = env.get_next_grid(action)
        # 在当前状态执行动作action，返回下一个状态， 奖励和累计耗时
        observation_, observation_vector_, spend_time, reward, seek_time, trans_time = env.step(action, spend_time,
                                                                                                seek_time,
                                                                                                trans_time)
        # DQN的性能受到奖励函数的影响，合理的设计奖励函数，能够DQN的性能变的更好
        if spend_time > SPEND_TIME:
            done = 1
        if env.q_type == 1:
            q_observation_ = env.tran_q_vector1(observation_)
        elif env.q_type == 2:
            q_observation_ = env.tran_q_vector2(observation_)
        elif env.q_type == 3:
            q_observation_ = env.tran_q_vector3(observation_)
        elif env.q_type == 4:
            q_observation_ = env.tran_q_vector4(observation_)
        elif env.q_type == 5:
            q_observation_ = env.tran_q_vector5(observation_)
        DQN_agent.experience_store(q_observation, action, reward, q_observation_, done)
        # 每一步都进行学习，只是并没有立即更新目标网络参数
        DQN_agent.learn()
        # 更新当前q网络状态
        q_observation = q_observation_
        r_sample.append(reward)
        if done:
            score = sum(r_sample)
            n_orders = (sum(i > 0 for i in r_sample))
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

    # 三小时
    # env = Env(250, 7, 0, 1, 2, 1)
    # for hour in [7, 11, 17, 21]:
    #     env.environment_change(250, hour, 0, 1)
    #     recording(update_random(env, 10000)).to_csv(r"C:\data\代码\实验结果\different_hour/random_" + str(hour) + ".csv")
    #     judge_converaged(r"C:\data\代码\实验结果\different_hour/random_" + str(hour) + ".csv",
    #                      r"C:\data\代码\实验结果\different_hour/random_" + str(hour) + "_summary.csv")
    #     hot_spot = HotSpot()
    #     recording(update_hot_spot(hot_spot, env, 10000)).to_csv(
    #         r"C:\data\代码\实验结果\different_hour/hot_spot_" + str(hour) + ".csv")
    #     judge_converaged(r"C:\data\代码\实验结果\different_hour/hot_spot_" + str(hour) + ".csv",
    #                      r"C:\data\代码\实验结果\different_hour/hot_spot_" + str(hour) + "_summary.csv")

    # for lamb in [0.3, 0.5, 0.7, 0.9, 1]:
    # env = Env(250, 8, 0, 1, 2, 1)
    # sarsa_q = SarsaLambdaTable(lamb=lamb)
    # recording(update_sarsa_lambda_leaning(sarsa_q, env, 50000)).to_csv(
    #     r"C:\data\代码\实验结果\sl_diffirent_lambda/new_lamb_" + str(lamb) + ".csv")
    # judge_converaged(r"C:\data\代码\实验结果\sl_diffirent_lambda/new_lamb_" + str(lamb) + ".csv",
    #                  r"C:\data\代码\实验结果\sl_diffirent_lambda/new_lamb_" + str(lamb) + "_summary.csv")

    # dqn调试
    # env = Env(250, 8, 0, 1, 7, 1)
    # dqn = DQN(9, env.q_input_features,learning_rate=0.1, n_experience_pool=1000)
    # recording(update_dqn(dqn, env, 1000)).to_csv(
    #     r"C:\data\代码\实验结果\dqn/final7_dqn2.csv")
    # judge_converaged(r"C:\data\代码\实验结果\dqn/final7_dqn2.csv",
    #                  r"C:\data\代码\实验结果\dqn/final7_dqn_sum2.csv")

    for lam in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        env = Env(250, 8, 0, 1, 1, 1)
        sl_table = SarsaLambdaTable()
        recording(update_new_mdp_sarsa_lambda_leaning(sl_table, env, 5000)).to_csv(
            r"C:\data\代码\实验结果\simulate_each_hour\weekday/new_sl_" + str(hour) + ".csv")

    # for batch_size in [128]:
    #     env = Env(250, 8, 0, 1, 2, 1)
    #     dqn = DQN(9, env.q_input_features, batch_size=batch_size, n_experience_pool=1000)
    #     recording(update_dqn(dqn, env, 5000)).to_csv(
    #         r"C:\data\代码\实验结果\dqn参数\batchsize/batchsize_" + str(batch_size) + ".csv")
    #     judge_converaged(r"C:\data\代码\实验结果\dqn参数\batchsize/batchsize_" + str(batch_size) + ".csv",
    #                      r"C:\data\代码\实验结果\dqn参数\batchsize/batchsize_" + str(batch_size) + "_summary.csv")
    #
    # for replace_time in [10, 30, 60]:
    #     env = Env(250, 8, 0, 1, 2, 1)
    #     dqn = DQN(9, env.q_input_features, replace_time=replace_time, n_experience_pool=1000)
    #     recording(update_dqn(dqn, env, 5000)).to_csv(
    #         r"C:\data\代码\实验结果\dqn参数\replacetime/replacetime_" + str(replace_time) + ".csv")
    #     judge_converaged(r"C:\data\代码\实验结果\dqn参数\replacetime/replacetime_" + str(replace_time) + ".csv",
    #                      r"C:\data\代码\实验结果\dqn参数\replacetime/replacetime_" + str(replace_time) + "_summary.csv")
    # #
    # for gama in [0.3, 0.5, 0.7, 0.9, 1.0]:
    #     env = Env(250, 8, 0, 1, 2, 1)
    #     dqn = DQN(9, env.q_input_features, gamma=gama, n_experience_pool=1000)
    #     recording(update_dqn(dqn, env, 5000)).to_csv(
    #         r"C:\data\代码\实验结果\dqn参数\discount/discount_" + str(gama) + ".csv")
    #     judge_converaged(r"C:\data\代码\实验结果\dqn参数\discount/discount_" + str(gama) + ".csv",
    #                      r"C:\data\代码\实验结果\dqn参数\discount/discount_" + str(gama) + "_summary.csv")

    # env = Env(250, 8, 0, 1, 2, 1)
    # sarsa_q = SarsaLambdaTable(lamb=0.5)
    # recording(update_sarsa_lambda_leaning(sarsa_q, env, 5000)).to_csv(
    #     r"C:\data\代码\实验结果\dqn参数\batchsize/sl_new.csv")
    # judge_converaged(r"C:\data\代码\实验结果\dqn参数\batchsize/sl_new.csv",
    #                  r"C:\data\代码\实验结果\dqn参数\batchsize/sl_new_summary.csv")
    # env = Env(250, 8, 0, 1, 2, 2)
    # q_table = Q(epsilon=0.1, learning_rate=0.01, gamma=0.7)
    # recording(update_q_leaning(q_table, env, 5000)).to_csv(
    #     r"C:\data\代码\实验结果\dqn参数\batchsize/q_ad.csv")
    # judge_converaged(r"C:\data\代码\实验结果\dqn参数\batchsize/q_ad.csv",
    #                  r"C:\data\代码\实验结果\dqn参数\batchsize/q_ad_summary.csv")

    # dqn模拟：
    # 工作日
    # for hour in range(7, 24):
    #     env = Env(250, hour, 0, 1, 2, 1)
    #     q_table = Q(epsilon=0.1, learning_rate=0.01, gamma=0.7)
    #     recording(update_q_leaning(q_table, env, 5000)).to_csv(
    #         r"C:\data\代码\实验结果\simulate_each_hour\weekday/dqn_d_" + str(hour) + ".csv")
    #     env = Env(250, hour, 0, 1, 2, 2)
    #     q_table = Q(epsilon=0.1, learning_rate=0.01, gamma=0.7)
    #     recording(update_q_leaning(q_table, env, 5000)).to_csv(
    #         r"C:\data\代码\实验结果\simulate_each_hour\weekday/dqn_ad_" + str(hour) + ".csv")

    # 周末
    # for hour in range(7, 24):
    #     env = Env(250, hour, 0, 0, 2, 1)
    #     q_table = Q(epsilon=0.1, learning_rate=0.01, gamma=0.7)
    #     recording(update_q_leaning(q_table, env, 5000)).to_csv(
    #         r"C:\data\代码\实验结果\simulate_each_hour\weekend/dqn_d_" + str(hour) + ".csv")
    #     env = Env(250, hour, 0, 0, 2, 2)
    #     q_table = Q(epsilon=0.1, learning_rate=0.01, gamma=0.7)
    #     recording(update_q_leaning(q_table, env, 5000)).to_csv(
    #         r"C:\data\代码\实验结果\simulate_each_hour\weekend/dqn_ad_" + str(hour) + ".csv")

    end_time = time_clock.perf_counter()
    print("运行时间为：", round(end_time - start_time), "seconds")
pass
