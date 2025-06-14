import osmnx as ox
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from matplotlib.backend_bases import PickEvent
from matplotlib.collections import PathCollection # PathCollection を直接インポート

# ★★★ 日本語フォントの設定 ★★★
try:
    plt.rcParams['font.family'] = 'Meiryo'
except:
    try:
        plt.rcParams['font.family'] = 'MS Gothic'
    except:
        print("日本語フォントが見つかりません。文字化けする可能性があります。")

# --- 1. OSMデータ取得とグラフ構築 ---

place_name = "Nippon Institute of Technology, Miyashiro, Saitama, Japan"

print(f"OSMデータからグラフをダウンロード中: {place_name}...")
G = ox.graph_from_place(place_name, network_type="walk")
print("グラフのダウンロードが完了しました。")

print(f"ノード数: {len(G.nodes)}")
print(f"エッジ数: {len(G.edges)}")

print("グラフを可視化中...")
# グラフを投影 (Q学習自体は投影されていないGを使用、可視化とクリック選択はG_projを使用)
G_proj = ox.project_graph(G) 

fig, ax = ox.plot_graph(G_proj, figsize=(10, 10), node_size=10, edge_linewidth=0.5, show=False, close=False)

output_dir = "graph_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

graphml_filepath = os.path.join(output_dir, "nippon_institute_of_tech_campus_walk.graphml")
gpkg_filepath = os.path.join(output_dir, "nippon_institute_of_tech_campus_walk_data.gpkg")

print(f"グラフデータを保存中 ({graphml_filepath}, {gpkg_filepath})...")
ox.save_graphml(G, filepath=graphml_filepath) # G (非投影) を保存
ox.save_graph_geopackage(G, filepath=gpkg_filepath) # G (非投影) を保存
print("グラフデータの保存が完了しました。")

# --- 2. ユーザーによるスタートとゴールの選択機能のためのQ学習準備 ---

nodes = list(G.nodes) # G (非投影) のノードを使用
num_states = len(nodes)
node_to_idx = {node_id: i for i, node_id in enumerate(nodes)}
idx_to_node = {i: node_id for i, node_id in enumerate(nodes)}

# --- Q学習とパスプロットの関数として切り出し ---
def run_q_learning_and_plot_path(start_node_id, goal_node_id, G, G_proj, nodes, num_states, node_to_idx, idx_to_node):
    # 報酬関数
    def get_reward(current_node_id, next_node_id, goal_node_id_local):
        # Case 1: ゴールノードに到達した場合
        if next_node_id == goal_node_id_local:
            return 1000000.0 # ゴール到達に非常に大きな正の報酬

        # Case 2: 現在のノードがゴールノードであり、そこから別のノードに移動しようとした場合
        if current_node_id == goal_node_id_local and next_node_id != goal_node_id_local:
            return -10000000.0 # ゴールから離れる行動に対する非常に大きな負の報酬 (ペナルティを極端に増強)

        # Case 3: 通常の移動
        edge_data = G.get_edge_data(current_node_id, next_node_id)
        if edge_data and 0 in edge_data and 'length' in edge_data[0]:
            distance = edge_data[0]['length']
            return -distance * 10.0
        else:
            # Case 4: 無効な移動
            return -20000000.0 


    def get_possible_actions(current_node_id):
        return [node_to_idx[neighbor] for neighbor in G.neighbors(current_node_id)]

    q_table = np.zeros((num_states, num_states))
    
    # ★★★ Qテーブルの初期化 (ゴールから出るQ値を強制的に負に) ★★★
    goal_idx = node_to_idx[goal_node_id]
    for action_idx in range(num_states):
        if action_idx != goal_idx:
            q_table[goal_idx, action_idx] = -100000000.0 


    learning_rate = 0.5 
    discount_factor = 0.995 
    epsilon_initial = 1.0
    
    # エピソード数と減衰率を調整
    num_episodes = 200000
    epsilon_min = 0.001 
    epsilon_decay_rate = (epsilon_min / epsilon_initial)**(1.0 / num_episodes) # ★★★ 計算で減衰率を設定 ★★★

    max_steps_per_episode = 1500 

    print(f"\nQ学習を開始します。")
    print(f"スタートノードID: {start_node_id} (インデックス: {node_to_idx[start_node_id]})")
    print(f"ゴールノードID: {goal_node_id} (インデックス: {node_to_idx[goal_node_id]})")
    print(f"エピソード数: {num_episodes}, 最大ステップ数/エピソード: {max_steps_per_episode}")

    epsilon = epsilon_initial
    for episode in range(num_episodes):
        current_node = start_node_id
        current_state_idx = node_to_idx[current_node]
        
        steps = 0
        
        while steps < max_steps_per_episode:
            possible_action_indices = get_possible_actions(current_node)
            if not possible_action_indices:
                break # 行き止まりの場合
            
            if random.uniform(0, 1) < epsilon:
                # 探索 (ランダム行動)
                next_state_idx = random.choice(possible_action_indices)
            else:
                # 活用 (学習済みQ値に基づく行動)
                q_values_of_possible_actions = q_table[current_state_idx, possible_action_indices]
                max_q_value = np.max(q_values_of_possible_actions)
                best_action_indices = np.where(q_values_of_possible_actions == max_q_value)[0]
                next_state_idx = possible_action_indices[random.choice(best_action_indices)]
                
            next_node = idx_to_node[next_state_idx]
            
            reward = get_reward(current_node, next_node, goal_node_id)
            
            if next_node == goal_node_id:
                max_future_q = 0
            else:
                possible_actions_from_next_state = get_possible_actions(next_node)
                max_future_q = 0
                if possible_actions_from_next_state:
                    max_future_q = np.max(q_table[next_state_idx, possible_actions_from_next_state])
            
            # Q値の更新 (Bellman方程式)
            q_table[current_state_idx, next_state_idx] = (1 - learning_rate) * q_table[current_state_idx, next_state_idx] + \
                                                      learning_rate * (reward + discount_factor * max_future_q)
            
            current_node = next_node
            current_state_idx = next_state_idx
            steps += 1
            
            if current_node == goal_node_id:
                break # ゴールに到達したらこのエピソードを終了
        
        # εを減衰 (指数減衰)
        epsilon = max(epsilon_min, epsilon * epsilon_decay_rate) 
        if (episode + 1) % 1000 == 0: 
            print(f"エピソード {episode + 1}/{num_episodes} 完了, 現在のε: {epsilon:.4f}")

    print("\nQ学習が完了しました！")

    # ★★★ Q値のデバッグ出力 ★★★
    print("\n--- Q-table Debug ---")
    goal_idx = node_to_idx[goal_node_id]
    
    print(f"Q-values from GOAL node ({goal_node_id}):")
    for neighbor_id in G.successors(goal_node_id): 
        neighbor_idx = node_to_idx[neighbor_id]
        q_val = q_table[goal_idx, neighbor_idx]
        print(f"  -> To {neighbor_id}: Q = {q_val:.2f}")

    print("\nQ-values TO GOAL node:")
    for u, v, k, data in G.in_edges(goal_node_id, keys=True, data=True):
        u_idx = node_to_idx[u]
        q_val = q_table[u_idx, goal_idx]
        print(f"  From {u} -> To {goal_node_id}: Q = {q_val:.2f} (Edge Length: {data.get('length', 'N/A'):.2f})")
    
    print("---------------------\n")

    # ★★★ ここにエッジデータ確認コードを組み込みます ★★★
    print("\n--- 特定のエッジデータを確認中 ---")

    # 問題が疑われるノードID
    u_check = 3909304922
    v_check = 5247013971 # これはゴールノードのID

    # エッジが存在するか確認し、そのデータを表示
    if G.has_edge(u_check, v_check):
        print(f"エッジ ({u_check}, {v_check}) のデータ:")
        edge_data = G.get_edge_data(u_check, v_check)
        for key, data in edge_data.items():
            print(f"  キー: {key}, データ: {data}")
            if 'length' in data:
                print(f"    -> length: {data['length']:.2f} m")
            else:
                print(f"    -> WARNING: 'length' 属性が見つかりません！")
    else:
        print(f"エッジ ({u_check}, {v_check}) はグラフ G に存在しません。")

    # 念のため、逆方向のエッジも確認（双方向道路の場合など）
    if G.has_edge(v_check, u_check):
        print(f"\nエッジ ({v_check}, {u_check}) (逆方向) のデータ:")
        edge_data_rev = G.get_edge_data(v_check, u_check)
        for key, data in edge_data_rev.items():
            print(f"  キー: {key}, データ: {data}")
            if 'length' in data:
                print(f"    -> length: {data['length']:.2f} m")
            else:
                print(f"    -> WARNING: 'length' 属性が見つかりません！")
    else:
        print(f"\nエッジ ({v_check}, {u_check}) (逆方向) はグラフ G に存在しません。")

    print("--- エッジデータの確認完了 ---\n")
    # ★★★ エッジデータ確認コード組み込みここまで ★★★


    # 最適経路導出ロジック
    def get_optimal_path(start_node_id_local, goal_node_id_local, G, q_table, node_to_idx, idx_to_node, max_steps=1500):
        current_node = start_node_id_local
        path_nodes = [current_node]
        visited_nodes = {current_node} 
        steps = 0

        while current_node != goal_node_id_local and steps < max_steps:
            current_state_idx = node_to_idx[current_node]
            possible_action_indices = get_possible_actions(current_node)
            
            if not possible_action_indices:
                print(f"DEBUG: Optimal path search - Node {current_node} has no possible actions (dead end). Path incomplete.")
                break 

            q_values_of_possible_actions = q_table[current_state_idx, possible_action_indices]
            
            max_q_value = np.max(q_values_of_possible_actions)
            best_action_indices = np.where(q_values_of_possible_actions == max_q_value)[0]
            next_state_idx = possible_action_indices[random.choice(best_action_indices)]
            
            next_node = idx_to_node[next_state_idx]
            
            if next_node in visited_nodes:
                print(f"DEBUG: Optimal path search - Loop detected: Node {next_node} already in path. Path incomplete.")
                break 

            path_nodes.append(next_node)
            visited_nodes.add(next_node) 
            current_node = next_node
            steps += 1

        if current_node == goal_node_id_local:
            print(f"\nゴールに到達しました！総ステップ数: {steps}")
            total_distance = 0
            for i in range(len(path_nodes) - 1):
                u = path_nodes[i]
                v = path_nodes[i+1]
                edge_data = G.get_edge_data(u, v)
                
                if edge_data and 0 in edge_data and 'length' in edge_data[0]:
                    total_distance += edge_data[0]['length']
                else:
                    print(f"警告: 経路上のノード {u} から {v} へのエッジに'length'属性が見つかりませんでした。距離計算に影響する可能性があります。")
                    return None, 0 

            print(f"経路の総距離: {total_distance:.2f} メートル")
            return path_nodes, total_distance
        else:
            print(f"\nゴールに到達できませんでした。総ステップ数: {steps}")
            if steps >= max_steps:
                print(f"原因: 最大ステップ数 ({max_steps}) に達しました。")
            elif current_node != goal_node_id_local: 
                print(f"原因: ループに陥ったか、行き止まりに到達しました。最終ノード: {current_node}")
            return None, 0

    print("\n学習済み方策に基づき最適な経路を導出中...")
    optimal_path_nodes, total_dist = get_optimal_path(start_node_id, goal_node_id, G, q_table, node_to_idx, idx_to_node)

    if optimal_path_nodes:
        print("最適経路を可視化中...")
        fig_route, ax_route = ox.plot_graph_route(
            G_proj, 
            optimal_path_nodes,
            route_color="r",
            route_linewidth=4,
            node_size=10,
            edge_linewidth=0.5,
            bgcolor='w',
            show=False,
            close=False
        )
        start_point_coords = G_proj.nodes[start_node_id] 
        goal_point_coords = G_proj.nodes[goal_node_id] 
        
        ax_route.scatter(start_point_coords['x'], start_point_coords['y'], c='blue', s=200, zorder=3, label='Start', edgecolors='black')
        ax_route.scatter(goal_point_coords['x'], goal_point_coords['y'], c='green', s=200, zorder=3, label='Goal', edgecolors='black')
        ax_route.legend(loc='best')
        
        plt.title(f"Optimal Path from {start_node_id} to {goal_node_id}\nTotal Distance: {total_dist:.2f} m", fontsize=14)
        plt.show()
        print("最適経路の可視化が完了しました。")
    else:
        print("最適な経路を見つけることができませんでした。Q学習のハイパーパラメータやエピソード数、またはスタート/ゴールノードの設定を確認してください。")

# --- メイン実行フロー ---
selection_mode = 0
start_node_id = None
goal_node_id = None

# マップクリックイベントハンドラ
def on_pick(event):
    global selection_mode
    global start_node_id, goal_node_id

    print(f"Pick event fired: {event.artist}")

    if not hasattr(event, 'mouseevent') or event.mouseevent is None:
        print("警告: 無効なマウスイベントです。スキップします。")
        return
    
    is_node_collection = False
    if isinstance(event.artist, PathCollection):
        is_node_collection = True

    if is_node_collection:
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata
        
        nearest_node = ox.distance.nearest_nodes(G_proj, x, y) 
        
        if nearest_node is None:
            print("エラー: 最も近いノードが見つかりませんでした。")
            return

        if selection_mode == 0:
            start_node_id = nearest_node
            print(f"\nスタートノードを設定: ID = {start_node_id}")
            start_point_coords = G_proj.nodes[start_node_id] 
            ax.scatter(start_point_coords['x'], start_point_coords['y'], c='blue', s=200, zorder=3, label='Start', edgecolors='black')
            selection_mode = 1
            ax.set_title(f"スタートノード選択完了。ゴールノードをクリックしてください。")
            fig.canvas.draw_idle()
        elif selection_mode == 1:
            goal_node_id = nearest_node
            print(f"ゴールノードを設定: ID = {goal_node_id}")
            goal_point_coords = G_proj.nodes[goal_node_id] 
            ax.scatter(goal_point_coords['x'], goal_point_coords['y'], c='green', s=200, zorder=3, label='Goal', edgecolors='black')
            ax.set_title(f"ゴールノード選択完了。Q学習を開始します。")
            fig.canvas.draw_idle()
            
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
            
            run_q_learning_and_plot_path(start_node_id, goal_node_id, G, G_proj, nodes, num_states, node_to_idx, idx_to_node)
    else:
        print(f"クリックされたオブジェクトはノードではありません: {event.artist}。ノードをクリックしてください。")

# インタラクティブな選択のため、プロットしたグラフを表示し、イベントを待機
print("\nスタートノードを選択してください。マップ上のノードをクリックしてください。")
ax.set_title(f"スタートノードをクリックしてください。")

node_collection = None
for collection in reversed(ax.collections):
    if isinstance(collection, PathCollection):
        node_collection = collection
        break

if node_collection:
    node_collection.set_picker(5)
    print(f"ノードコレクションのpickerが設定されました: {node_collection.get_picker()}")
else:
    print("警告: ノードコレクションが見つかりませんでした。クリックイベントが正しく動作しない可能性があります。")

cid = fig.canvas.mpl_connect('pick_event', on_pick)
plt.show()