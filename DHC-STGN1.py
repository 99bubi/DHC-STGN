import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
import json
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class Config:
    # 数据路径
    DATA_PATH = "./data/solar_radiation.csv"
    SIMILARITY_DIR = "./similarity_matrices"
    OUTPUT_DIR = "./results"
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/metrics/site_metrics", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/predictions", exist_ok=True)
    
    # 模型参数
    WINDOW_SIZE = 30
    PRED_SIZE = 7
    HIDDEN_DIM = 256
    TCN_KERNEL_SIZE = 3
    TCN_NUM_LAYERS = 2
    DROPOUT = 0.1
    
    # 自适应迭代参数
    R_MIN = 1
    R_MAX = 30
    THRESHOLD_HIGH = 0.6
    
    # 注意力参数
    TEMPERATURE = 2.0
    ADJUSTMENT_SCALE = 0.1
    
    # 置信度融合权重（初始权重）
    G_WEIGHT = 0.3
    M_WEIGHT = 0.4
    C_WEIGHT = 0.2
    R_WEIGHT = 0.1
    
    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    PATIENCE = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据划分
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # 测试模式
    TEST_MODE = True
    TEST_SITES = 5


config = Config()
print(f"Using device: {config.DEVICE}")

# ==================== 加载预计算矩阵 ====================
print("Loading pre-computed similarity matrices...")
try:
    g_ij = np.load(f"{config.SIMILARITY_DIR}/mahalanobis_similarity.npy").astype(np.float32)
    m_ij = np.load(f"{config.SIMILARITY_DIR}/dtw_similarity.npy").astype(np.float32)
    r_j = np.load(f"{config.SIMILARITY_DIR}/source_reliability.npy").astype(np.float32)
    c_ij = np.load(f"{config.SIMILARITY_DIR}/cross_interpretability.npy").astype(np.float32)
    print("✓ Similarity matrices loaded")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# ==================== 数据加载 ====================
print("Loading time series data...")
data = pd.read_csv(config.DATA_PATH, header=None)
site_names = data.iloc[:, 0].values.astype(str)
site_coords = data.iloc[:, 1:4].values.astype(np.float32)
temporal_data = data.iloc[:, 4:].values.astype(np.float32)

n_sites, n_timesteps = temporal_data.shape

# 处理缺失值
temporal_data = np.where(np.isnan(temporal_data), 
                         np.nanmean(temporal_data, axis=1, keepdims=True), 
                         temporal_data)

# 标准化
scalers = {}
temporal_data_scaled = np.zeros_like(temporal_data)
for i in range(n_sites):
    scaler = StandardScaler()
    temporal_data_scaled[i] = scaler.fit_transform(temporal_data[i].reshape(-1, 1)).flatten()
    scalers[i] = scaler


class StationDataset(Dataset):
    def __init__(self, data, site_idx, window_size, pred_size, indices):
        self.data = data[site_idx]
        self.window_size = window_size
        self.pred_size = pred_size
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.data[t - self.window_size + 1 : t + 1].copy()
        y = self.data[t + 1 : t + self.pred_size + 1].copy()
        
        if len(x) < self.window_size:
            x = np.pad(x, (self.window_size - len(x), 0), 'edge')
        if len(y) < self.pred_size:
            y = np.pad(y, (0, self.pred_size - len(y)), 'edge')
            
        return torch.FloatTensor(x), torch.FloatTensor(y), torch.LongTensor([t])


class TemporalConvEncoder(nn.Module):
    def __init__(self, hidden_dim=256, kernel_size=3, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.convs.append(nn.Conv1d(1, hidden_dim, kernel_size, padding=kernel_size//2))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x)
            x = self.relu(x)
            x = self.dropout(x)
            
            if residual.shape[1] == x.shape[1]:
                x = x + residual
            
            x = x.transpose(1, 2)
            x = norm(x)
            x = x.transpose(1, 2)
        
        return x.mean(dim=-1)


class ConfidenceModulation(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, h_i, h_j, s_base):
        x = torch.cat([h_i, h_j, s_base], dim=-1)
        return self.net(x).squeeze(-1)


class AdaptiveIterationController:
    def __init__(self, r_min=1, r_max=5, threshold_high=0.6):
        self.r_min = r_min
        self.r_max = r_max
        self.threshold_high = threshold_high
    
    def compute_iterations(self, confidence_dist, device):
        high_conf_mask = confidence_dist > self.threshold_high
        n_high = torch.sum(high_conf_mask).float()
        rho = n_high / (confidence_dist.shape[0] - 1) if confidence_dist.shape[0] > 1 else torch.tensor(0.0, device=device)
        
        eps = 1e-12
        p = torch.clamp(confidence_dist, eps, 1 - eps)
        entropy = -torch.sum(p * torch.log(p))
        max_entropy = torch.log(torch.tensor(confidence_dist.shape[0], dtype=torch.float32, device=device))
        zeta = 1 - (entropy / max_entropy) if max_entropy > 0 else torch.tensor(0.0, device=device)
        
        r_raw = self.r_min + (self.r_max - self.r_min) * rho * zeta
        return int(torch.ceil(torch.clamp(r_raw, self.r_min, self.r_max)).item())


class DHCSTGCN(nn.Module):
    def __init__(self, site_idx, n_sites, hidden_dim, window_size, pred_size):
        super().__init__()
        
        self.site_idx = site_idx
        self.n_sites = n_sites
        self.hidden_dim = hidden_dim
        self.pred_size = pred_size
        self.window_size = window_size
        
        self.tcn_encoder = TemporalConvEncoder(
            hidden_dim=hidden_dim,
            kernel_size=config.TCN_KERNEL_SIZE,
            num_layers=config.TCN_NUM_LAYERS,
            dropout=config.DROPOUT
        )
        
        self.confidence_modulations = nn.ModuleList([
            ConfidenceModulation(hidden_dim) for _ in range(config.R_MAX)
        ])
        
        self.gate_weights = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim) * 0.1) for _ in range(config.R_MAX)
        ])
        self.gate_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(config.R_MAX)
        ])
        
        self.msg_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(config.R_MAX)
        ])
        
        self.grus = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim) for _ in range(config.R_MAX)
        ])
        
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, pred_size)
            ) for _ in range(config.R_MAX)
        ])
        
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim + pred_size * config.R_MAX, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, config.R_MAX),
            nn.Softmax(dim=-1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GRUCell):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def compute_base_confidence(self, g_vec, m_vec, c_vec, r_vec):
        confidence = (config.G_WEIGHT * g_vec + 
                     config.M_WEIGHT * m_vec + 
                     config.C_WEIGHT * c_vec + 
                     config.R_WEIGHT * torch.log(r_vec + 1e-8))
        return F.softmax(confidence * config.TEMPERATURE, dim=0)
    
    def forward(self, x, g_vec, m_vec, c_vec, r_vec):
        batch_size = x.shape[0]
        device = x.device
        
        s_base_vec = self.compute_base_confidence(g_vec, m_vec, c_vec, r_vec)
        s_base = s_base_vec.unsqueeze(0).expand(batch_size, -1)
        
        controller = AdaptiveIterationController(
            r_min=config.R_MIN,
            r_max=config.R_MAX,
            threshold_high=config.THRESHOLD_HIGH
        )
        R = controller.compute_iterations(s_base_vec, device)
        
        h_current = self.tcn_encoder(x)
        h_all = h_current.unsqueeze(1).expand(-1, self.n_sites, -1)
        h_state = h_current
        
        perspectives = []
        
        for r in range(R):
            h_i_expanded = h_state.unsqueeze(1).expand(-1, self.n_sites, -1)
            h_j_expanded = h_all
            
            h_i_flat = h_i_expanded.reshape(-1, self.hidden_dim)
            h_j_flat = h_j_expanded.reshape(-1, self.hidden_dim)
            s_base_flat = s_base.reshape(-1, 1)
            
            delta = self.confidence_modulations[r](h_i_flat, h_j_flat, s_base_flat)
            delta = delta.reshape(batch_size, self.n_sites)
            
            s_refined = s_base + config.ADJUSTMENT_SCALE * torch.tanh(delta)
            
            gate = torch.sigmoid(
                torch.einsum('d,bd->b', self.gate_weights[r], h_state) + self.gate_biases[r]
            )
            s_modulated = s_refined * gate.unsqueeze(-1).expand(-1, self.n_sites)
            
            attention = F.softmax(s_modulated * config.TEMPERATURE, dim=-1)
            
            h_j_transformed = self.msg_transforms[r](h_all.reshape(-1, self.hidden_dim))
            h_j_transformed = h_j_transformed.reshape(batch_size, self.n_sites, self.hidden_dim)
            messages = torch.bmm(attention.unsqueeze(1), h_j_transformed).squeeze(1)
            
            h_state = self.grus[r](messages, h_state)
            pred = self.prediction_heads[r](h_state)
            perspectives.append(pred)
        
        while len(perspectives) < config.R_MAX:
            perspectives.append(torch.zeros(batch_size, self.pred_size, device=device))
        
        perspectives = torch.stack(perspectives, dim=1)
        
        fusion_features = torch.cat([
            perspectives.reshape(batch_size, -1),
            h_state,
        ], dim=-1)
        
        fusion_weights = self.fusion_net(fusion_features)
        final_pred = (perspectives * fusion_weights.unsqueeze(-1)).sum(dim=1)
        
        return final_pred, perspectives, fusion_weights, R


class StationTrainer:
    def __init__(self, site_idx, train_loader, val_loader, test_loader):
        self.site_idx = site_idx
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.g_vec = torch.FloatTensor(g_ij[site_idx]).to(config.DEVICE)
        self.m_vec = torch.FloatTensor(m_ij[site_idx]).to(config.DEVICE)
        self.c_vec = torch.FloatTensor(c_ij[site_idx]).to(config.DEVICE)
        self.r_vec = torch.FloatTensor(r_j).to(config.DEVICE)
        
        self.model = DHCSTGCN(
            site_idx=site_idx,
            n_sites=n_sites,
            hidden_dim=config.HIDDEN_DIM,
            window_size=config.WINDOW_SIZE,
            pred_size=config.PRED_SIZE
        ).to(config.DEVICE)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.EPOCHS, eta_min=1e-6
        )
        self.criterion = nn.MSELoss()
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for x, y, _ in self.train_loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            
            self.optimizer.zero_grad()
            pred, _, _, _ = self.model(x, self.g_vec, self.m_vec, self.c_vec, self.r_vec)
            loss = self.criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self, loader, scaler=None):
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0
        
        for x, y, _ in loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            
            pred, _, _, _ = self.model(x, self.g_vec, self.m_vec, self.c_vec, self.r_vec)
            loss = self.criterion(pred, y)
            total_loss += loss.item()
            
            if scaler is not None:
                pred_np = pred.cpu().numpy()
                y_np = y.cpu().numpy()
                
                pred_inv = scaler.inverse_transform(pred_np)
                y_inv = scaler.inverse_transform(y_np)
                
                all_preds.append(pred_inv)
                all_targets.append(y_inv)
            else:
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        if all_preds:
            predictions = np.vstack(all_preds)
            targets = np.vstack(all_targets)
            rmse = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
            mae = mean_absolute_error(targets.flatten(), predictions.flatten())
            r2 = r2_score(targets.flatten(), predictions.flatten())
            return total_loss / len(loader), rmse, mae, r2, predictions, targets
        
        return total_loss / len(loader), 0, 0, 0, None, None
    
    def train(self):
        for epoch in range(1, config.EPOCHS + 1):
            train_loss = self.train_epoch()
            val_loss, val_rmse, val_mae, val_r2, _, _ = self.evaluate(self.val_loader, scalers[self.site_idx])
            self.scheduler.step()
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                
                torch.save({
                    'model_state': self.best_model_state,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse
                }, f"{config.OUTPUT_DIR}/models/site_{self.site_idx}.pth")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= config.PATIENCE:
                break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        test_loss, test_rmse, test_mae, test_r2, test_preds, test_targets = self.evaluate(
            self.test_loader, scalers[self.site_idx]
        )
        
        # 保存单个站点的预测
        if test_preds is not None:
            np.save(f"{config.OUTPUT_DIR}/predictions/site_{self.site_idx}_pred.npy", test_preds)
            np.save(f"{config.OUTPUT_DIR}/predictions/site_{self.site_idx}_true.npy", test_targets)
        
        return {
            'site_idx': self.site_idx,
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'test_r2': float(test_r2),
            'predictions': test_preds,
            'targets': test_targets
        }


def main():
    valid_start = config.WINDOW_SIZE
    valid_end = n_timesteps - config.PRED_SIZE
    all_indices = list(range(valid_start, valid_end))
    np.random.shuffle(all_indices)
    
    train_size = int(len(all_indices) * config.TRAIN_RATIO)
    val_size = int(len(all_indices) * config.VAL_RATIO)
    
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]
    
    train_sites = range(config.TEST_SITES) if config.TEST_MODE else range(n_sites)
    all_metrics = []
    
    # 用于整体评估
    all_predictions = []
    all_targets = []
    
    for site_idx in tqdm(train_sites, desc="Training"):
        try:
            train_dataset = StationDataset(
                temporal_data_scaled, site_idx, 
                config.WINDOW_SIZE, config.PRED_SIZE, train_indices
            )
            val_dataset = StationDataset(
                temporal_data_scaled, site_idx,
                config.WINDOW_SIZE, config.PRED_SIZE, val_indices
            )
            test_dataset = StationDataset(
                temporal_data_scaled, site_idx,
                config.WINDOW_SIZE, config.PRED_SIZE, test_indices
            )
            
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE * 2)
            test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2)
            
            trainer = StationTrainer(site_idx, train_loader, val_loader, test_loader)
            metrics = trainer.train()
            
            # 收集用于整体评估
            if metrics['predictions'] is not None:
                all_predictions.append(metrics['predictions'])
                all_targets.append(metrics['targets'])
                
                # 保存单个站点指标
                site_metrics = {
                    'site_idx': metrics['site_idx'],
                    'test_rmse': metrics['test_rmse'],
                    'test_mae': metrics['test_mae'],
                    'test_r2': metrics['test_r2']
                }
                all_metrics.append(site_metrics)
            
        except Exception as e:
            print(f"Site {site_idx} failed: {e}")
            continue
    
    # 保存各站点平均性能
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(f"{config.OUTPUT_DIR}/metrics/site_average_metrics.csv", index=False)
        
        print("\n" + "="*50)
        print("AVERAGE PER-SITE PERFORMANCE")
        print("="*50)
        print(f"Sites trained: {len(all_metrics)}")
        print(f"Average RMSE: {metrics_df['test_rmse'].mean():.2f} ± {metrics_df['test_rmse'].std():.2f}")
        print(f"Average MAE: {metrics_df['test_mae'].mean():.2f} ± {metrics_df['test_mae'].std():.2f}")
        print(f"Average R²: {metrics_df['test_r2'].mean():.4f} ± {metrics_df['test_r2'].std():.4f}")
    
    # 整体评估（所有站点的所有预测拼接）
    if all_predictions:
        all_preds_concat = np.vstack(all_predictions)
        all_targets_concat = np.vstack(all_targets)
        
        overall_rmse = np.sqrt(mean_squared_error(
            all_targets_concat.flatten(), 
            all_preds_concat.flatten()
        ))
        overall_mae = mean_absolute_error(
            all_targets_concat.flatten(), 
            all_preds_concat.flatten()
        )
        overall_r2 = r2_score(
            all_targets_concat.flatten(), 
            all_preds_concat.flatten()
        )
        
        print("\n" + "="*50)
        print("OVERALL TEST SET PERFORMANCE")
        print("="*50)
        print(f"Total samples: {len(all_preds_concat)}")
        print(f"Overall RMSE: {overall_rmse:.2f} W/m²")
        print(f"Overall MAE: {overall_mae:.2f} W/m²")
        print(f"Overall R²: {overall_r2:.4f}")
        print("="*50)
        
        # 保存整体结果
        overall_metrics = {
            'overall_rmse': float(overall_rmse),
            'overall_mae': float(overall_mae),
            'overall_r2': float(overall_r2),
            'total_samples': int(len(all_preds_concat)),
            'n_sites': len(all_predictions)
        }
        with open(f"{config.OUTPUT_DIR}/metrics/overall_metrics.json", 'w') as f:
            json.dump(overall_metrics, f, indent=2)
        
        np.save(f"{config.OUTPUT_DIR}/metrics/all_predictions.npy", all_preds_concat)
        np.save(f"{config.OUTPUT_DIR}/metrics/all_targets.npy", all_targets_concat)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    main()
