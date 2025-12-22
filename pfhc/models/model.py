import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialHNNModule(nn.Module):
    def __init__(self, input_dim, hnn_hidden_dim, hnn_layers=2, dropout=0.1):
        super().__init__()
        self.hnn_hidden_dim = hnn_hidden_dim
        
        self.W_v2e = nn.Sequential(
            nn.Linear(input_dim, hnn_hidden_dim),
            nn.LayerNorm(hnn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.W_e = nn.Sequential(
            nn.Linear(input_dim, hnn_hidden_dim),
            nn.LayerNorm(hnn_hidden_dim),
            nn.GELU()
        )
        
        self.W_interference = nn.Sequential(
            nn.Linear(input_dim * 2, hnn_hidden_dim),
            nn.LayerNorm(hnn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.interference_gate = nn.Sequential(
            nn.Linear(hnn_hidden_dim * 2, hnn_hidden_dim),
            nn.Tanh()
        )
        
        self.spatial_fusion = nn.Sequential(
            nn.Linear(hnn_hidden_dim * 2, hnn_hidden_dim),
            nn.LayerNorm(hnn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.temporal_gru = nn.GRU(
            input_size=hnn_hidden_dim,
            hidden_size=hnn_hidden_dim,
            num_layers=max(hnn_layers, 2),
            batch_first=True,
            dropout=dropout if hnn_layers > 1 else 0,
            bidirectional=False
        )
        
        self.final_norm = nn.LayerNorm(hnn_hidden_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, targets_raw, neighbors, masks, lengths=None):
        B, T, N, C = neighbors.shape
        
        valid_mask = (~masks).unsqueeze(-1).float()
        degree = valid_mask.sum(dim=2).clamp(min=1.0)
        
        neighbors_agg = (neighbors * valid_mask).sum(dim=2) / degree
        
        if N > 1:
            neighbor_distances = torch.norm(neighbors - targets_raw.unsqueeze(2), p=2, dim=-1)
            top_k = min(3, N)
            _, top_indices = torch.topk(neighbor_distances, top_k, dim=2, largest=False)
            
            neighbors_2hop = torch.gather(neighbors, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, C))
            neighbors_2hop_mask = torch.gather(valid_mask.squeeze(-1), 2, top_indices)
            
            neighbors_2hop_agg = (neighbors_2hop * neighbors_2hop_mask.unsqueeze(-1)).sum(dim=2) / (neighbors_2hop_mask.sum(dim=2, keepdim=True).clamp(min=1.0))
            
            neighbors_agg = 0.7 * neighbors_agg + 0.3 * neighbors_2hop_agg
        
        z_neighbors = self.W_v2e(neighbors_agg)
        
        z_self = self.W_e(targets_raw)
        
        neighbors_centered = neighbors - neighbors_agg.unsqueeze(2)
        neighbors_var = ((neighbors_centered ** 2) * valid_mask).sum(dim=2) / degree
        
        interference_input = torch.cat([neighbors_agg, neighbors_var], dim=-1)
        z_interference = self.W_interference(interference_input)
        
        gate_input = torch.cat([z_self, z_interference], dim=-1)
        gate = self.interference_gate(gate_input)
        z_gated_interference = gate * z_interference
        
        z_spatial = torch.cat([z_self + z_neighbors, z_gated_interference], dim=-1)
        z_spatial = self.spatial_fusion(z_spatial)
        
        self.temporal_gru.flatten_parameters()
        z_temporal, _ = self.temporal_gru(z_spatial)
        
        z_temporal = z_temporal + z_spatial
        
        return self.final_norm(z_temporal)

class FuzzificationLayer(nn.Module):
    def __init__(self, hnn_hidden_dim, num_rules_K):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hnn_hidden_dim)
        self.centers = nn.Parameter(torch.randn(num_rules_K, hnn_hidden_dim))
        self.bandwidths = nn.Parameter(torch.ones(num_rules_K))

    def forward(self, hnn_features):
        x = self.pre_norm(hnn_features)
        x_exp = x.unsqueeze(2)
        c_exp = self.centers.unsqueeze(0).unsqueeze(0)
        
        dist_sq = torch.sum((x_exp - c_exp)**2, dim=-1)
        bw_sq = (self.bandwidths**2).unsqueeze(0).unsqueeze(0) + 1e-6
        return torch.exp(-dist_sq / (2 * bw_sq))

class ConditionalCausalGRU(nn.Module):
    def __init__(self, num_rules_K, hidden_dim, layers, volatility_window_size, 
                 tau_0=1, delta_tau=5, dropout=0.1):
        super().__init__()
        self.volatility_window_size = volatility_window_size
        self.tau_0 = tau_0
        self.delta_tau = delta_tau
        
        self.register_buffer('gamma_min', torch.tensor(0.0))
        self.register_buffer('gamma_max', torch.tensor(1.0))
        
        self.gru = nn.GRU(
            input_size=num_rules_K, 
            hidden_size=hidden_dim,
            num_layers=max(layers, 2),
            batch_first=True,
            bidirectional=False, 
            dropout=dropout if layers > 1 else 0
        )
        
        self.volatility_gate = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.pi_head = nn.Linear(hidden_dim, num_rules_K)
        
        nn.init.constant_(self.volatility_gate[0].bias, 0.5)
        nn.init.constant_(self.volatility_gate[2].bias, 0.5)

    def calculate_volatility(self, mu_t):
        B, T, K = mu_t.shape
        
        mu_t_perm = mu_t.permute(0, 2, 1) 
        mu_padded = F.pad(mu_t_perm, (self.volatility_window_size - 1, 0))
        windows = mu_padded.unfold(dimension=2, size=self.volatility_window_size, step=1)
        var_t = windows.var(dim=3, unbiased=False).sum(dim=1)
        
        if T > 1:
            diff_t = torch.norm(mu_t[:, 1:, :] - mu_t[:, :-1, :], p=2, dim=-1)
            diff_t = F.pad(diff_t, (1, 0), value=0.0)
        else:
            diff_t = torch.zeros(B, T, device=mu_t.device)
        
        if T > 2:
            diff2_t = torch.abs(diff_t[:, 1:] - diff_t[:, :-1])
            diff2_t = F.pad(diff2_t, (1, 0), value=0.0)
        else:
            diff2_t = torch.zeros(B, T, device=mu_t.device)
        
        gamma_t = (var_t + 0.7 * diff_t + 0.3 * diff2_t).unsqueeze(-1)
        
        if T > 1:
            gamma_smooth = F.avg_pool1d(gamma_t.permute(0, 2, 1), kernel_size=3, padding=1, stride=1).permute(0, 2, 1)
        else:
            gamma_smooth = gamma_t
        
        if self.training:
            with torch.no_grad():
                self.gamma_min = 0.99 * self.gamma_min + 0.01 * gamma_smooth.min()
                self.gamma_max = 0.99 * self.gamma_max + 0.01 * gamma_smooth.max()
        
        gamma_norm = (gamma_smooth - self.gamma_min) / (self.gamma_max - self.gamma_min + 1e-6)
        return torch.clamp(gamma_norm, 0, 1)
    
    def calculate_entropy(self, mu_t):
        mu_safe = torch.clamp(mu_t, min=1e-8)
        entropy = -(mu_safe * torch.log(mu_safe)).sum(dim=-1, keepdim=True)
        entropy_norm = entropy / (torch.log(torch.tensor(mu_t.size(-1), dtype=torch.float32, device=mu_t.device)) + 1e-6)
        return entropy_norm

    def forward(self, mu_t):
        gamma = self.calculate_volatility(mu_t)
        
        entropy = self.calculate_entropy(mu_t)
        
        self.gru.flatten_parameters()
        gru_out, _ = self.gru(mu_t)
        
        gate_input = torch.cat([gru_out, gamma, entropy], dim=-1)
        dynamic_gate = self.volatility_gate(gate_input)
        gated_features = gru_out * dynamic_gate
        
        pi_logits = self.pi_head(gated_features)
        return torch.softmax(pi_logits, dim=-1)

class ProbabilityFuzzyFusionHead(nn.Module):
    def __init__(self, num_rules_K, temperature=1.0):
        super().__init__()
        self.K = num_rules_K
        self.temperature = temperature
        self.register_buffer('I', torch.eye(self.K))

    def forward(self, mu_t, pi_t):
        mu_t_row = mu_t.unsqueeze(2)
        Omega_raw = mu_t_row.expand(-1, -1, self.K, -1) * (1 - self.I)
        Omega = Omega_raw / (Omega_raw.sum(dim=-1, keepdim=True) + 1e-9)
        
        S_mu = torch.diag_embed(mu_t)
        Pi = torch.diag_embed(pi_t)
        
        term1 = self.I - Pi
        term2 = torch.matmul(Pi, Omega)
        R = torch.matmul(S_mu, term1 + term2)
        
        s_t_next = R.sum(dim=-2)
        return F.softmax(s_t_next / self.temperature, dim=-1)

class PFHC_Model(nn.Module):
    def __init__(self, input_dim=6, hnn_hidden_dim=256, hnn_layers=2,
                 num_fuzzy_rules=8, 
                 tcn_hidden_dim=256, tcn_layers=3, 
                 dropout=0.2, kernel_size=None,
                 volatility_window_size=10, tau_0=1, delta_tau=5,
                 temperature=1.0):
        super().__init__()
        
        hnn_layers = hnn_layers if hnn_layers is not None else 2
        
        self.spatial_hnn = SpatialHNNModule(input_dim, hnn_hidden_dim, hnn_layers, dropout)
        self.fuzzification = FuzzificationLayer(hnn_hidden_dim, num_fuzzy_rules)
        self.temporal = ConditionalCausalGRU(
            num_rules_K=num_fuzzy_rules,
            hidden_dim=tcn_hidden_dim,
            layers=tcn_layers,
            volatility_window_size=volatility_window_size,
            tau_0=tau_0, delta_tau=delta_tau,
            dropout=dropout
        )
        self.fusion_head = ProbabilityFuzzyFusionHead(num_fuzzy_rules, temperature)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        
    def forward(self, targets_raw, neighbors, masks, lengths=None):
        hnn_features = self.spatial_hnn(targets_raw, neighbors, masks, lengths)
        mu_t = self.fuzzification(hnn_features)
        pi_t = self.temporal(mu_t)
        preds = self.fusion_head(mu_t, pi_t)
        return torch.clamp(preds, min=1e-6, max=1.0-1e-6)
