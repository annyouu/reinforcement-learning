#演習問題9.1
#強化学習その２（Q-learning)迷路探索問題　2021.6.17実施

import copy
import numpy as np
import matplotlib.pyplot as plt

class QLearningAgent:
    """
        Q学習 エージェント
    """

    def __init__(
            self,
            alpha=.2,
            epsilon=.1,
            gamma=.99,
            actions=None,
            observation=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reward_history = []
        self.actions = actions
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = None
        self.previous_action = None
        self.q_values = self._init_q_values()

    def _init_q_values(self):
        """
           Q テーブルの初期化
        """
        q_values = {}
        q_values[self.state] = np.repeat(0.0, len(self.actions))
        return q_values

    def init_state(self):
        """
            状態の初期化
        """
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def act(self):
        # ε-greedy選択
        if np.random.uniform() < self.epsilon:  # random行動
            action = np.random.randint(0, len(self.q_values[self.state]))
        else:   # greedy 行動
            action = np.argmax(self.q_values[self.state])

        self.previous_action = action
        return action

    def observe(self, next_state, reward=None):
        """
            次の状態と報酬の観測
        """
        next_state = str(next_state)
        if next_state not in self.q_values:  # 始めて訪れる状態であれば
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

        if reward is not None:
            self.reward_history.append(reward)
            self.learn(reward)

    def learn(self, reward):
        """
            Q値の更新
        """
        q = self.q_values[self.previous_state][self.previous_action]  # Q(s, a)
        max_q = max(self.q_values[self.state])  # max Q(s')
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        self.q_values[self.previous_state][self.previous_action] = q + \
            (self.alpha * (reward + (self.gamma * max_q) - q))

#未知環境と行動への報酬設定
class GridWorld:

    def __init__(self):

#環境の状態(4種類)
        self.filed_type = {
            "N": 0,  # 通常：通路
            "G": 1,  # ゴール（＋の報酬）
            "W": 2,  # 壁（０，または－の報酬
            "T": 3,  # トラップ　（マイナス報酬）
        }
#行動の候補（４種類）      
        self.actions = {
            "UP": 0,
            "DOWN": 1,
            "LEFT": 2,
            "RIGHT": 3
        }
#未知環境（迷路）
        self.map = [[3, 2, 0, 1],
                    [0, 0, 0, 2],
                    [0, 0, 2, 0],
                    [2, 0, 2, 0],
                    [0, 0, 0, 0]]

        self.start_pos = 0, 4   # エージェントのスタート地点(x, y)
        self.agent_pos = copy.deepcopy(self.start_pos)  # エージェントがいる地点

    def step(self, action):
        """
            行動の実行
            状態, 報酬、ゴールしたかを返却
        """
        to_x, to_y = copy.deepcopy(self.agent_pos)

        # 移動可能かどうかの確認。移動不可能であれば、ポジションはそのままにマイナス報酬
        if self._is_possible_action(to_x, to_y, action) == False:
            return self.agent_pos, -1, False

        if action == self.actions["UP"]:
            to_y += -1
        elif action == self.actions["DOWN"]:
            to_y += 1
        elif action == self.actions["LEFT"]:
            to_x += -1
        elif action == self.actions["RIGHT"]:
            to_x += 1

        is_goal = self._is_end_episode(to_x, to_y)  # エピソードの終了の確認
        reward = self._compute_reward(to_x, to_y)
        self.agent_pos = to_x, to_y
        return self.agent_pos, reward, is_goal

    def _is_end_episode(self, x, y):
        """
            x, yがエピソードの終了かの確認。
        """
        if self.map[y][x] == self.filed_type["G"]:      # ゴール
            return True
        elif self.map[y][x] == self.filed_type["T"]:    # トラップ
            return True
        else:
            return False

    def _is_wall(self, x, y):
        """
            x, yが壁かどうかの確認
        """
        if self.map[y][x] == self.filed_type["W"]:
            return True
        else:
            return False

    def _is_possible_action(self, x, y, action):
        """
            実行可能な行動かどうかの判定
        """
        to_x = x
        to_y = y

        if action == self.actions["UP"]:
            to_y += -1
        elif action == self.actions["DOWN"]:
            to_y += 1
        elif action == self.actions["LEFT"]:
            to_x += -1
        elif action == self.actions["RIGHT"]:
            to_x += 1

        if len(self.map) <= to_y or 0 > to_y:
            return False
        elif len(self.map[0]) <= to_x or 0 > to_x:
            return False
        elif self._is_wall(to_x, to_y):
            return False

        return True

    def _compute_reward(self, x, y):
        if self.map[y][x] == self.filed_type["N"]:
            return 0
        elif self.map[y][x] == self.filed_type["G"]:
            return 100
        elif self.map[y][x] == self.filed_type["T"]:
            return -100

    def reset(self):
        self.agent_pos = self.start_pos
        return self.start_pos


#定数
NB_EPISODE = 100    # エピソード数
EPSILON = .1    # 探索率
ALPHA = .1      # 学習率
GAMMA = .90     # 割引率
ACTIONS = np.arange(4)  # 行動の集合

if __name__ == '__main__':
    grid_env = GridWorld()  # grid worldの環境の初期化
    ini_state = grid_env.start_pos  # 初期状態（エージェントのスタート地点の位置）
    # エージェントの初期化
    agent = QLearningAgent(
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,  # 探索率
        actions=ACTIONS,   # 行動の集合
        observation=ini_state)  # Q学習エージェント
    rewards = []    # 評価用報酬の保存
    is_end_episode = False  # エージェントがゴールしてるかどうか？

    # 実験
    for episode in range(NB_EPISODE):
        episode_reward = []  # 1エピソードの累積報酬
        while(is_end_episode == False):    # ゴールするまで続ける
            action = agent.act()  # 行動選択
            state, reward, is_end_episode = grid_env.step(action)
            agent.observe(state, reward)   # 状態と報酬の観測
            episode_reward.append(reward)
        rewards.append(np.sum(episode_reward))  # このエピソードの平均報酬を与える
        state = grid_env.reset()  # 初期化
        agent.observe(state)    # エージェントを初期位置に
        is_end_episode = False

    # 結果のプロット
    plt.plot(np.arange(NB_EPISODE), rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig("result.jpg")
    plt.show()