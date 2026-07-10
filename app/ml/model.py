# ============================================================================
# StormSentinel Backend — Model Architecture
# EXACT port of StormSentinelNetV2 from v2_06_model_training.py. Must match
# byte-for-byte — any structural difference means the saved state_dict
# won't load correctly.
# ============================================================================

import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim); self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim); self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.GELU(); self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.act(self.bn1(self.fc1(x)))
        out = self.drop(out)
        return self.act(self.bn2(self.fc2(out)) + x)


class HazardHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class StormSentinelNetV2(nn.Module):
    def __init__(self, input_dim, trunk_dim=256, mid_dim=128, dropout=0.3):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Linear(input_dim, trunk_dim), nn.BatchNorm1d(trunk_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.res1 = ResBlock(trunk_dim, dropout)
        self.res2 = ResBlock(trunk_dim, dropout)
        self.mid_block = nn.Sequential(
            nn.Linear(trunk_dim, mid_dim), nn.BatchNorm1d(mid_dim),
            nn.GELU(), nn.Dropout(dropout * 0.6),
        )
        self.res3 = ResBlock(mid_dim, dropout * 0.6)

        self.head_wildfire          = HazardHead(mid_dim, 64)
        self.head_tornado           = HazardHead(mid_dim, 128)
        self.head_hail              = HazardHead(mid_dim, 64)
        self.head_thunderstorm_wind = HazardHead(mid_dim, 64)
        self.head_flash_flood       = HazardHead(mid_dim, 64)
        self.head_heat              = HazardHead(mid_dim, 64)
        self.head_drought           = HazardHead(mid_dim, 64)

    def forward(self, x):
        x = self.res2(self.res1(self.input_block(x)))
        x = self.res3(self.mid_block(x))
        return {
            "wildfire":          self.head_wildfire(x),
            "tornado":           self.head_tornado(x),
            "hail":              self.head_hail(x),
            "thunderstorm_wind": self.head_thunderstorm_wind(x),
            "flash_flood":       self.head_flash_flood(x),
            "heat":              self.head_heat(x),
            "drought":           self.head_drought(x),
        }


HEAD_ORDER = ["wildfire", "tornado", "hail", "thunderstorm_wind", "flash_flood", "heat", "drought"]

# Must match v2_05_feature_engineering.py exactly
US_ONLY_HAZARDS = {"tornado", "hail", "thunderstorm_wind"}
LOW_CONFIDENCE_WILDFIRE_CITIES = ["Kuwait City", "Dubai"]

BEST_THRESHOLDS = {
    "wildfire": 0.65, "tornado": 0.85, "hail": 0.75, "thunderstorm_wind": 0.65,
    "flash_flood": 0.90, "heat": 0.65, "drought": 0.65,
}  # from test_metrics_v2.json — update if retrained
