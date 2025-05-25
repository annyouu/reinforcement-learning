#演習問題9.2 Sarsa学習による迷路探索問題


import numpy as np
import matplotlib.pyplot as plt
import copy


class EpsGreedyQPolicy:
    def __init__(self, epsilon=.1, decay_rate=1):
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.epsilon:  # random行動
            action = np.random.random_integers(0, nb_actions-1)
        else:   # greedy 行動
            action = np.argmax(q_values)

        return action


class SARSAAgent:
    """
        sarsa
    """
    def __init__(self, alpha=.2, policy=None, gamma=.99, actions=None, observation=None, alpha_decay_rate=None):
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.reward_history = []
        self.actions = actions
        self.alpha_decay_rate = alpha_decay_rate
        self.state = str(observation)
        self.previous_state = None
        self.ini_state = str(observation)   # 初期状態の保存
        self.previous_action_id = None
        self.recent_action_id = 0
        self.q_values = self._init_q_values()
        self.training = True

    def _init_q_values(self):
        """
           Q テーブルの初期化
        """
        q_values = {}
        q_values[self.state] = np.repeat(0.0, len(self.actions))
        return q_values

    def init_state(self):
        """
            状態を初期状態に（再スタート用）
        """
        self.previous_state = None
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def init_policy(self, policy):
        self.policy = policy

    def act(self):
        action = self.actions[self.recent_action_id]
        return action

    def select_action(self):
        action_id = self.policy.select_action(self.q_values[self.state])
        return action_id

    def observe(self, next_state, reward=None):
        """
            次の状態の観測 
        """
        next_state = str(next_state)
        if next_state not in self.q_values: # 始めて訪れる状態であれば
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

        if self.training and reward is not None:
            self.reward_history.append(reward)
            self.learn(reward)

    def learn(self, reward):
        """
            報酬の獲得とQ値の更新 
        """
        self.reward_history.append(reward)
        self.previous_action_id = copy.deepcopy(self.recent_action_id)
        self.recent_action_id = self.select_action()
        self.q_values[self.previous_state][self.previous_action_id] = self._update_q_value(reward)   

    def _update_q_value(self, reward):
        """
            Q値の更新量の計算
        """
        q = self.q_values[self.previous_state][self.previous_action_id] # Q(s, a)
        q2 = self.q_values[self.state][self.recent_action_id] # Q(s', a')saras学習則
       # Q(s, a) = Q(s, a) + alpha*(r+gamma*Q(s', a')-Q(s, a))
    #   q2 = max(self.q_values[self.state]) # Max Q(s', a')　Q学習則
   # Q(s, a) = Q(s, a) + alpha*(r+gamma*Max Q(s', a')-Q(s, a))
        updated_q_value = q + (self.alpha * (reward + (self.gamma * q2) - q)) 

        return updated_q_value

    def update_hyper_parameters(self):
        """
            ハイパーパラメータの更新 
        """
        self.decay_alpha()
        self.policy.decay_epsilon()


class GridWorld:

    def __init__(self):

        self.map = [[0, 2, 0, 1], 
                    [0, 0, 0, 2], 
                    [0, 0, 2, 0], 
                    [0, 2, 0, 0], 
                    [0, 0, 0, 0]]

        self.start_pos = 0, 4   # エージェントのスタート地点(x, y)
        self.agent_pos = copy.deepcopy(self.start_pos)  # エージェントがいる地点

        self.filed_type = {
                "N": 0,  #通常
                "G": 1,  #ゴール
                "W": 2,  #壁
                }

        self.actions = {
            "UP": 0, 
            "DOWN": 1, 
            "LEFT": 2, 
            "RIGHT": 3
            }

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

        is_goal = self._is_goal(to_x, to_y) # ゴールしているかの確認
        reward = self._compute_reward(to_x, to_y)
        self.agent_pos = to_x, to_y
        return self.agent_pos, reward, is_goal

    def _is_goal(self, x, y):
        """
            x, yがゴール地点かの判定 
        """
        if self.map[y][x] == self.filed_type["G"]:
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

    def reset(self):
        self.agent_pos = self.start_pos
        return self.start_pos


if __name__ == '__main__':
    grid_env = GridWorld() # grid worldの環境の初期化
    ini_state = grid_env.start_pos  # 初期状態（エージェントのスタート地点の位置）
    policy = EpsGreedyQPolicy(epsilon=.01, decay_rate=.99) # 方策の初期化。ここではε-greedy
    agent = SARSAAgent(policy=policy, actions=np.arange(4), observation=ini_state)  # sarsa エージェントの初期化
    nb_episode = 100   #エピソード数
    rewards = []    # 評価用報酬の保存
    is_goal = False # エージェントがゴールしてるかどうか？
    for episode in range(nb_episode):
        episode_reward = [] # 1エピソードの累積報酬
        while(is_goal == False):    # ゴールするまで続ける
            action = agent.act()  # 行動選択
            state, reward, is_goal = grid_env.step(action)
            agent.observe(state, reward)   # 状態と報酬の観測
            episode_reward.append(reward)
        rewards.append(np.sum(episode_reward)) # このエピソードの平均報酬を与える
        state = grid_env.reset()    #  初期化
        agent.observe(state)    # エージェントを初期位置に
        is_goal = False

    # 結果のプロット
    plt.plot(np.arange(nb_episode), rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig("result.jpg")
    plt.show()