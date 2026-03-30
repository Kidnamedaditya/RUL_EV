"""
app.py  —  EV Predictive Maintenance API
Run from the project root:
    python app.py
Then open ev_dashboard.html in your browser.
API endpoint: POST /predict  →  returns per-component RUL
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Device ───────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

WINDOW_SIZE     = 20
MAX_RUL         = 300.0
COMPONENT_NAMES = ['Battery', 'Motor', 'Brakes', 'Tires', 'Suspension']
SENSOR_COLS     = [
    'SoC', 'SoH', 'Battery_Voltage', 'Battery_Current',
    'Battery_Temperature', 'Charge_Cycles', 'Motor_Temperature',
    'Motor_Vibration', 'Motor_Torque', 'Motor_RPM', 'Power_Consumption',
    'Brake_Pad_Wear', 'Brake_Pressure', 'Reg_Brake_Efficiency',
    'Tire_Pressure', 'Tire_Temperature', 'Suspension_Load',
    'Ambient_Temperature', 'Ambient_Humidity', 'Load_Weight',
    'Driving_Speed', 'Distance_Traveled', 'Idle_Time', 'Route_Roughness'
]

# ── Model architecture (must match training notebook exactly) ─────────
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None

class GRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = 0.0
    def set_lambda(self, lam): self.lam = lam
    def forward(self, x):
        return GradientReversalFn.apply(x, self.lam)

class CNNFeatureExtractor(nn.Module):
    def __init__(self, n_sensors=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_sensors, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,  128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class ComponentRULHead(nn.Module):
    def __init__(self, feat_dim=128, name=''):
        super().__init__()
        self.name = name
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),       nn.ReLU(),
            nn.Linear(32, 1),        nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class DomainDiscriminator(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),       nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

class MultiComponentDANN(nn.Module):
    def __init__(self, component_names=COMPONENT_NAMES):
        super().__init__()
        self.feature_extractor    = CNNFeatureExtractor()
        self.grl                  = GRL()
        self.domain_discriminator = DomainDiscriminator()
        self.rul_heads = nn.ModuleDict({
            name: ComponentRULHead(name=name)
            for name in component_names
        })
    def forward(self, x):
        features  = self.feature_extractor(x)
        rul_preds = torch.cat(
            [self.rul_heads[n](features) for n in COMPONENT_NAMES], dim=1
        )
        domain_pred = self.domain_discriminator(self.grl(features))
        return rul_preds, domain_pred

# ── Load model & scaler ───────────────────────────────────────────────
print('Loading model...')
model = MultiComponentDANN().to(DEVICE)
model.load_state_dict(torch.load('models/cnn_dann_rul.pth', map_location=DEVICE))
model.eval()
print('Model loaded.')

print('Loading scaler...')
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print('Scaler loaded.')

# ── Flask app ────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow requests from the HTML file opened in browser


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(DEVICE)})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON body with either:
      A) Single reading: {"sensors": {SoC: 0.7, SoH: 0.9, ...}}
         → repeated WINDOW_SIZE times to form a window
      B) Full window:   {"window": [[...], [...], ...]}  (20 × 24 array)

    Returns:
      {"Battery": 241.3, "Motor": 189.7, "Brakes": 156.2,
       "Tires": 278.4, "Suspension": 203.1, "cycles_per_day": 4,
       "days": {"Battery": 60, "Motor": 47, ...}}
    """
    data = request.get_json(force=True)

    # ── Build input window ────────────────────────────────────────────
    if 'window' in data:
        # Full window provided (20 rows × 24 sensors)
        raw = np.array(data['window'], dtype=np.float32)
        if raw.shape != (WINDOW_SIZE, len(SENSOR_COLS)):
            return jsonify({'error': f'window must be shape ({WINDOW_SIZE}, {len(SENSOR_COLS)})'}), 400
    elif 'sensors' in data:
        # Single reading → tile to WINDOW_SIZE rows
        sensors = data['sensors']
        row = np.array([sensors.get(k, 0.0) for k in SENSOR_COLS], dtype=np.float32)
        raw = np.tile(row, (WINDOW_SIZE, 1))   # (20, 24)
    else:
        return jsonify({'error': 'Provide either "sensors" or "window" in the request body'}), 400

    # ── Scale & predict ───────────────────────────────────────────────
    try:
        scaled = scaler.transform(raw).astype(np.float32)          # (20, 24)
        x = torch.tensor(scaled).unsqueeze(0).permute(0, 2, 1).to(DEVICE)  # (1, 24, 20)

        with torch.no_grad():
            rul_norm, _ = model(x)                                  # (1, 5)

        rul_vals = (rul_norm.cpu().numpy()[0] * MAX_RUL).tolist()
        result   = dict(zip(COMPONENT_NAMES, rul_vals))

        # Add days estimate (4 cycles per day assumption)
        cpd = 4
        result['days'] = {k: round(v / cpd) for k, v in result.items()}

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print('\n' + '='*50)
    print('  EV RUL API running at http://localhost:5000')
    print('  Open ev_dashboard.html in your browser')
    print('='*50 + '\n')
    app.run(debug=False, port=5000)
