import random
import re
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset, HeteroData
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import networkx as nx
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from leblanc import leblanc_3

def load_matrix_from_file(filename):
    with open(filename, 'r') as file:
        matrix = [list(map(float, line.split())) for line in file]
    return np.array(matrix)


def generate_matrices(n, max_demand=300, max_capacity=3000, max_time=50, sparsity=0.2):
    D = np.random.randint(0, max_demand + 1, size=(n, n))
    np.fill_diagonal(D, 0)

    T_0 = np.random.uniform(1, max_time, size=(n, n))
    C = np.random.uniform(1, max_capacity, size=(n, n))

    mask = np.random.rand(n, n) < sparsity
    T_0[mask] = 0
    C[mask] = 0

    np.fill_diagonal(T_0, 0)
    np.fill_diagonal(C, 0)

    return T_0, D, C

def generate_connected_graphs():

    # Пример использования
    G, demand_matrix = generate_strongly_connected_tap_graph()
    C, t_0 = get_matrices_from_graph(G)

    return t_0, demand_matrix, C

def get_matrices_from_graph(G):
    n = len(G.nodes)
    capacity_matrix = np.zeros((n, n))
    t0_matrix = np.zeros((n, n))

    for u, v, data in G.edges(data=True):
        capacity_matrix[u, v] = data['capacity']
        t0_matrix[u, v] = data['t0']

    return capacity_matrix, t0_matrix

def generate_strongly_connected_tap_graph(n_nodes=20, min_cap=100, max_cap=500, min_t0=1, max_t0=10, demand_pairs=80):
    # Шаг 1: Создаём минимальное остовное дерево (MST) в виде неориентированного графа
    T = nx.generators.trees.random_tree(n_nodes)
    
    # Шаг 2: Превращаем в сильно связный направленный граф
    G = nx.DiGraph()
    for u, v in T.edges():
        G.add_edge(u, v)
        G.add_edge(v, u)  # Делаем двустороннее ребро для сохранения связности

    # Шаг 3: Добавляем случайное количество дополнительных рёбер
    min_edges = G.number_of_edges()  # Минимальное число рёбер (остовное дерево)
    max_edges = min(n_nodes * (n_nodes - 1), n_nodes * (n_nodes - 1))  # Ограничение сверху
    target_edges = random.randint(min_edges, max_edges)  # Случайное число рёбер

    while G.number_of_edges() < target_edges:
        u, v = np.random.choice(n_nodes, size=2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v)

    # Шаг 4: Назначаем capacity и t0
    for u, v in G.edges():
        G[u][v]['capacity'] = random.randint(min_cap, max_cap)
        G[u][v]['t0'] = random.uniform(min_t0, max_t0)

    # Шаг 5: Генерация demand matrix (случайный спрос между парами узлов)
    demand_matrix = np.zeros((n_nodes, n_nodes))
    nodes = list(G.nodes())
    for _ in range(random.randint((n_nodes ** 2) // 2, n_nodes ** 2)):
        src, dst = np.random.choice(nodes, size=2, replace=False)
        demand_matrix[(src, dst)] = random.randint(10, 500)  # случайный спрос

    # Проверяем, что граф действительно сильно связный
    assert nx.is_strongly_connected(G), "Ошибка: граф не сильно связный!"

    return G, demand_matrix


def read_data(df, index, node_scaler=None, edge_scaler=None, y_scaler=None, n_nodes=20):

    data = df

    row = data.iloc[index]

    d_columns = [f"D_{i}" for i in range(n_nodes ** 2)]
    t_0_columns = [f"T_0_{i}" for i in range(n_nodes ** 2)]
    c_columns = [f"C_{i}" for i in range(n_nodes ** 2)]
    flow_columnns = [f"flow_{i}" for i in range(n_nodes ** 2)]

    d_vector = row[d_columns].values 
    t_0_vector = row[t_0_columns].values
    c_vector = row[c_columns].values
    flow_vector = row[flow_columnns].values

    d_matrix = d_vector.reshape(n_nodes, n_nodes)
    t_0_matrix = t_0_vector.reshape(n_nodes, n_nodes)
    c_matrix = c_vector.reshape(n_nodes, n_nodes)
    flow_matrix = flow_vector.reshape(n_nodes,n_nodes)

    rows, cols = np.where(t_0_matrix > 0)
    
    edge_index = np.vstack((rows, cols))
    y = flow_matrix[rows, cols]
    edge_attr = np.vstack((t_0_matrix[rows, cols], c_matrix[rows, cols])).T

    d_matrix_normalized = node_scaler.transform(d_matrix.reshape((n_nodes ** 2, 1))).reshape(n_nodes, n_nodes) if node_scaler else d_matrix
    
    edge_attr_normalized = edge_scaler.transform(edge_attr) if edge_scaler else edge_attr

    y_normalized = y_scaler.transform(y.reshape(-1, 1)).flatten() if y_scaler else y
    
    return d_matrix_normalized, edge_index, edge_attr_normalized, y_normalized
 

def split_dataset(dataset, test_size=0.2):
    train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=42)
    return train_data, test_data


class CustomHeteroGraphDataset(Dataset):
    def __init__(self, data_path, indices, node_scaler=None, edge_scaler=None, y_scaler=None, n_nodes=20):
        self.df = pd.read_csv(data_path)
        self.indices_ = indices
        self.node_scaler = node_scaler
        self.edge_scaler = edge_scaler
        self.y_scaler = y_scaler
        self.data_path = data_path
        self.n_nodes = n_nodes
        super().__init__()

    def len(self):
        return len(self.indices_)

    def get(self, idx):
        index = self.indices_[idx]
        X, edge_index, edge_attrs, y = read_data(
            self.df, index, self.node_scaler, self.edge_scaler, self.y_scaler, self.n_nodes
        )

        data = HeteroData()
        data['real'].x = torch.tensor(X, dtype=torch.float)  # Пример: [num_nodes, 1]
        data['real'].num_nodes = X.shape[0]
        data['virtual'].num_nodes = X.shape[0]
        data['real'].edge_index = torch.tensor(edge_index, dtype=torch.long)
        data['real'].edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        data['real'].y = torch.tensor(y, dtype=torch.float)
        
        # Виртуальные ребра
        rows, cols = np.where(X > 0)
        virtual_edge_index = np.vstack((rows, cols))
        data['virtual'].edge_index = torch.tensor(virtual_edge_index, dtype=torch.long)
        return data

class CustomGraphDataset(Dataset):
    def __init__(self, src, length, transform=None, pre_transform=None, n_nodes=20):
        self.src = src
        self.length = length
        self.n_nodes = n_nodes
        super().__init__(transform, pre_transform)
    
    def len(self):
        return self.length
    
    def get(self, index):
        X, edge_index, edge_attrs, y = read_data(self.src, index, None, None, None, self.n_nodes)
        data = Data(
            x=torch.tensor(X, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs is not None else None,
            y=torch.tensor(y, dtype=torch.float) if y is not None else None
        )
        
        return data
    
def prepare_scalers(data_path, train_indices, n_nodes=20, scaler_type=StandardScaler):
    
    # Собираем все узловые и реберные признаки из тренировочной выборки
    all_node_features = []
    all_edge_features = []
    all_y = []

    df = pd.read_csv(data_path)

    for idx in train_indices:
        d_matrix, _, edge_attr, y = read_data(df, idx, n_nodes=n_nodes)
        
        # Узловые признаки (пример: D_matrix 20x20 как 400 признаков)
        all_node_features.append(d_matrix.reshape((n_nodes ** 2, 1))) 
        # Реберные признаки (T0 и C)
        all_edge_features.append(edge_attr)
        all_y.append(y)
    
    y_scaler = scaler_type()
    y_scaler.fit(np.concatenate(all_y).reshape(-1, 1))
    
    # Создаем и обучаем скейлеры
    node_scaler = scaler_type()
    node_scaler.fit(np.concatenate(all_node_features))
    
    edge_scaler = scaler_type()
    edge_scaler.fit(np.vstack(all_edge_features))
    
    return node_scaler, edge_scaler, y_scaler




def read_tntp_od_matrix(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    od_matrix = []
    current_origin = None

    for line in lines:
        line = line.strip()
        
        # Пропускаем метаданные
        if line.startswith('<') or not line:
            continue
        
        # Обнаружение нового источника (Origin)
        origin_match = re.match(r'Origin\s+(\d+)', line)
        if origin_match:
            current_origin = int(origin_match.group(1))
            continue
        
        # Обнаружение пар (destination, flow)
        flows = re.findall(r'(\d+)\s*:\s*([\d.]+)', line)
        for dest, flow in flows:
            od_matrix.append((current_origin, int(dest), float(flow)))

    # Создаём DataFrame
    df = pd.DataFrame(od_matrix, columns=['Origin', 'Destination', 'Flow'])
    return df.pivot(index='Origin', columns='Destination', values='Flow').to_numpy()


def read_tntp_cap_t_0(net):
    df = net
    unique_nodes = sorted(set(df["init_node"]).union(set(df["term_node"])))
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}

    # Размер матрицы
    n = len(unique_nodes)

    # Заполняем матрицы NaN
    capacity_matrix = np.full((n, n), 0)
    free_flow_time_matrix = np.full((n, n), 0)

    # Заполняем матрицы
    for _, row in df.iterrows():
        i, j = node_to_idx[row["init_node"]], node_to_idx[row["term_node"]]
        capacity_matrix[i, j] = row["capacity"]
        free_flow_time_matrix[i, j] = row["free_flow_time"]

    return capacity_matrix, free_flow_time_matrix


def read_tntp(net_path, trips_path):
    net = pd.read_csv(net_path, skiprows=8, sep='\t')
    d_matrix = read_tntp_od_matrix(trips_path)
    c_matrix, t_0_matrix = read_tntp_cap_t_0(net)
    return d_matrix, c_matrix, t_0_matrix


def generate_alike_data(net_path, trips_path, n_samples, demand_max, demand_min, cap_max, cap_min, filename, eps=0.1):

    dataset = []

    d_matrix, c_matrix, t_0_matrix = read_tntp(net_path, trips_path)
    for _ in tqdm(range(n_samples)):
        d_factor = np.random.uniform(demand_min, demand_max, d_matrix.shape)
        c_factor = np.random.uniform(cap_min, cap_max, c_matrix.shape)
        
        d_sample = d_matrix * d_factor
        c_sample = c_matrix * c_factor
        
        flow_vector, iterations, function_value = leblanc_3(t_0_matrix, d_sample, c_sample, eps)

        sample = {
            'T_0': t_0_matrix.flatten(),
            'D': d_sample.flatten(),
            'C': c_sample.flatten(),
            'flow_vector': flow_vector.flatten()
        }
        dataset.append(sample)
    
    rows = []

    for sample in dataset:
        row = {
            **{f'T_0_{i}': val for i, val in enumerate(sample['T_0'])},
            **{f'D_{i}': val for i, val in enumerate(sample['D'])},
            **{f'C_{i}': val for i, val in enumerate(sample['C'])},
            **{f'flow_{i}': val for i, val in enumerate(sample['flow_vector'])}
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
        
def read_data_for_mlp(df, index, node_scaler=None, edge_scaler=None, y_scaler=None, n_nodes=20):
    data = df
    row = data.iloc[index]

    # Извлечение названий столбцов
    d_columns = [f"D_{i}" for i in range(n_nodes ** 2)]
    t_0_columns = [f"T_0_{i}" for i in range(n_nodes ** 2)]
    c_columns = [f"C_{i}" for i in range(n_nodes ** 2)]
    flow_columns = [f"flow_{i}" for i in range(n_nodes ** 2)]

    # Получение векторов из строки данных
    d_vector = row[d_columns].values
    t_0_vector = row[t_0_columns].values[row[t_0_columns].values > 0]
    c_vector = row[c_columns].values[row[c_columns].values > 0]
    flow_vector = row[flow_columns].values

    # Нормализация d_vector с использованием node_scaler
    if node_scaler is not None:
        d_vector_normalized = node_scaler.transform(d_vector.reshape(1, -1)).flatten()
    else:
        d_vector_normalized = d_vector.copy()

    # Объединение t_0 и c для применения edge_scaler
    t0_c_matrix = np.column_stack((t_0_vector, c_vector))
    if edge_scaler is not None:
        t0_c_normalized = edge_scaler.transform(t0_c_matrix)
    else:
        t0_c_normalized = t0_c_matrix.copy()
    t_0_normalized = t0_c_normalized[:, 0]
    c_normalized = t0_c_normalized[:, 1]

    # Нормализация flow_vector с использованием y_scaler
    if y_scaler is not None:
        flow_normalized = y_scaler.transform(flow_vector.reshape(-1, 1)).flatten()
    else:
        flow_normalized = flow_vector.copy()

    return d_vector_normalized, t_0_normalized, c_normalized, flow_normalized

class CustomGraphDatasetMLP(Dataset):
    def __init__(self, data_path, indices, node_scaler=None, edge_scaler=None, y_scaler=None, n_nodes=20):
        self.df = pd.read_csv(data_path)
        self.indices_ = indices
        self.node_scaler = node_scaler
        self.edge_scaler = edge_scaler
        self.y_scaler = y_scaler
        self.data_path = data_path
        self.n_nodes = n_nodes
        super().__init__()

    def len(self):
        return len(self.indices_)

    def get(self, idx):
        index = self.indices_[idx]
        d_vector_normalized, t_0_normalized, c_normalized, flow_normalized = read_data_for_mlp(
            self.df, index, self.node_scaler, self.edge_scaler, self.y_scaler, self.n_nodes
        )
        
        return torch.tensor(d_vector_normalized), torch.tensor(t_0_normalized), torch.tensor(c_normalized), torch.tensor(flow_normalized)
    

def generate_alike_data_C(net_path, trips_path, n_samples, cap_max, cap_min, filename, eps=0.1):

    dataset = []

    d_matrix, c_matrix, t_0_matrix = read_tntp(net_path, trips_path)
    for _ in tqdm(range(n_samples)):
        c_factor = np.random.uniform(cap_min, cap_max, c_matrix.shape)
        
        c_sample = c_matrix * c_factor
        
        flow_vector, iterations, function_value = leblanc_3(t_0_matrix, d_matrix, c_sample, eps)

        sample = {
            'T_0': t_0_matrix.flatten(),
            'D': d_matrix.flatten(),
            'C': c_sample.flatten(),
            'flow_vector': flow_vector.flatten()
        }
        dataset.append(sample)
    
    rows = []

    for sample in dataset:
        row = {
            **{f'T_0_{i}': val for i, val in enumerate(sample['T_0'])},
            **{f'D_{i}': val for i, val in enumerate(sample['D'])},
            **{f'C_{i}': val for i, val in enumerate(sample['C'])},
            **{f'flow_{i}': val for i, val in enumerate(sample['flow_vector'])}
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)



def generate_alike_data_num_od_pairs(net_path, trips_path, n_samples, cap_max, cap_min, demand_max, demand_min, filename, od_pairs, fixed=True, eps=0.1):
    dataset = []

    d_matrix, c_matrix, t_0_matrix = read_tntp(net_path, trips_path)
    num_nodes = d_matrix.shape[0]

    all_pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    selected_pairs = np.random.choice(len(all_pairs), od_pairs, replace=False)
    selected_od_pairs = [all_pairs[idx] for idx in selected_pairs]
    
    for _ in tqdm(range(n_samples)):

        d_factor = np.random.uniform(demand_min, demand_max, d_matrix.shape)
        c_factor = np.random.uniform(cap_min, cap_max, c_matrix.shape)
        
        if not fixed:
            selected_pairs = np.random.choice(len(all_pairs), od_pairs, replace=False)
            selected_od_pairs = [all_pairs[idx] for idx in selected_pairs]

        d_sample = d_matrix * d_factor
        c_sample = c_matrix * c_factor
        
        d_filtered = np.zeros_like(d_sample)
        for (i, j) in selected_od_pairs:
            d_filtered[i, j] = d_sample[i, j]


        start = time.time()
        flow_vector, iterations, function_value = leblanc_3(t_0_matrix, d_filtered, c_sample, eps)
        end = time.time()
        print(end - start)
        sample = {
            'T_0': t_0_matrix.flatten(),
            'D': d_filtered.flatten(),
            'C': c_sample.flatten(),
            'flow_vector': flow_vector.flatten()
        }
        dataset.append(sample)
    
    rows = []
    
    for sample in dataset:
        row = {
            **{f'T_0_{i}': val for i, val in enumerate(sample['T_0'])},
            **{f'D_{i}': val for i, val in enumerate(sample['D'])},
            **{f'C_{i}': val for i, val in enumerate(sample['C'])},
            **{f'flow_{i}': val for i, val in enumerate(sample['flow_vector'])}
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)


def generate_n_pairs(data_path, od_pairs, save_path, num_nodes=24):
    df = pd.read_csv(data_path)

    all_pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    selected_pairs = np.random.choice(len(all_pairs), od_pairs, replace=False)
    selected_od_pairs = [all_pairs[idx] for idx in selected_pairs]

    d_columns = [col for col in df.columns if col.startswith("D_")]
    t_columns = [col for col in df.columns if col.startswith("T_0_")]
    c_columns = [col for col in df.columns if col.startswith("C_")]


    dataset = []

    for i, row in tqdm(df.iterrows()):

        D = np.array(row[d_columns]).reshape(num_nodes, num_nodes)
        T = np.array(row[t_columns]).reshape(num_nodes, num_nodes)
        C = np.array(row[c_columns]).reshape(num_nodes, num_nodes)


        d_filtered = np.zeros_like(D)
        for (i, j) in selected_od_pairs:
            d_filtered[i, j] = D[i, j]
        
        flow_vector, iterations, function_value = leblanc_3(T, d_filtered, C, e=0.1)

        sample = {
            'T_0': T.flatten(),
            'D': d_filtered.flatten(),
            'C': C.flatten(),
            'flow_vector': flow_vector.flatten()
        }
        dataset.append(sample)

        
    rows = []
        
    for sample in dataset:
        row = {
            **{f'T_0_{i}': val for i, val in enumerate(sample['T_0'])},
            **{f'D_{i}': val for i, val in enumerate(sample['D'])},
            **{f'C_{i}': val for i, val in enumerate(sample['C'])},
            **{f'flow_{i}': val for i, val in enumerate(sample['flow_vector'])}
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)