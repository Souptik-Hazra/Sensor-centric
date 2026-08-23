import os
import torch
import numpy as np
import pandas as pd

try:
    from EquiTrafficAI.gwnet.gwnet_model import GraphWaveNet
except ImportError:
    from gwnet_model import GraphWaveNet

class UniversalPeMSAdapter:
    """
    Universal PeMS Adapter for Graph WaveNet (GWNet) & EquiTraffic-GPT.
    Supports ANY PeMS dataset (PeMS03, PeMS04, PeMS07, PeMS08, METR-LA, SD400, PeMS-BAY, etc.)
    with dynamic node count N, dynamic channel count C, and auto spatial graph generation.
    """
    def __init__(self, dataset_identifier: str = "metr_la", device: str = None):
        self.dataset_id = str(dataset_identifier).lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "..", "data")
        
        self.num_nodes = 207
        self.data_array = None
        self.model = None
        self.load_dataset(self.dataset_id)

    def load_dataset(self, dataset_name: str):
        self.dataset_id = dataset_name.lower()
        
        # Preset Datasets
        if "sd" in self.dataset_id or "400" in self.dataset_id:
            self.num_nodes = 716
            checkpoint = os.path.join(self.base_dir, "gwnet_sd400.pt")
        elif "pems04" in self.dataset_id or "pems4" in self.dataset_id:
            self.num_nodes = 307
            checkpoint = os.path.join(self.base_dir, "gwnet_pems04.pt")
        elif "pems08" in self.dataset_id or "pems8" in self.dataset_id:
            self.num_nodes = 170
            checkpoint = os.path.join(self.base_dir, "gwnet_pems08.pt")
        elif "bay" in self.dataset_id:
            self.num_nodes = 325
            checkpoint = os.path.join(self.base_dir, "gwnet_pems_bay.pt")
        elif "pems03" in self.dataset_id:
            self.num_nodes = 358
            checkpoint = os.path.join(self.base_dir, "gwnet_pems03.pt")
        elif "pems07" in self.dataset_id:
            self.num_nodes = 883
            checkpoint = os.path.join(self.base_dir, "gwnet_pems07.pt")
        else:
            self.num_nodes = 207
            checkpoint = os.path.join(self.base_dir, "gwnet_metr_la.pt")

        # Initialize PyTorch GWNet Model for dynamic node count N
        self.model = GraphWaveNet(
            num_nodes=self.num_nodes,
            in_dim=3,
            out_dim=12,
            residual_channels=32,
            dilation_channels=32,
            skip_channels=64,
            end_channels=128
        ).to(self.device)

        if os.path.exists(checkpoint) and os.path.getsize(checkpoint) > 0:
            try:
                self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
                print(f"[+] Universal PeMS Adapter: Loaded GWNet PyTorch Checkpoint for {self.dataset_id.upper()} ({self.num_nodes} nodes).")
            except Exception as e:
                print(f"[!] Warning: Loaded fresh GWNet weights for {self.dataset_id.upper()}: {e}")
        else:
            print(f"[+] Universal PeMS Adapter: Initialized GWNet Model for {self.dataset_id.upper()} ({self.num_nodes} nodes).")
            
        self.model.eval()

    @staticmethod
    def preprocess_pems_array(data: np.ndarray) -> np.ndarray:
        """
        Ensures data array has shape (T, N, 3) where channels are [speed, time_of_day, day_of_week].
        """
        if data.ndim == 2:
            T, N = data.shape
            speed = data.reshape(T, N, 1)
        elif data.ndim == 3:
            T, N, C = data.shape
            if C >= 3:
                return data[:, :, :3]
            speed = data[:, :, :1]
        else:
            raise ValueError(f"Unsupported PeMS array dimensions: {data.shape}")

        T, N, _ = speed.shape
        tod = (np.arange(T) % 288 / 288.0).reshape(T, 1, 1)
        tod = np.tile(tod, (1, N, 1))
        dow = ((np.arange(T) // 288) % 7 / 7.0).reshape(T, 1, 1)
        dow = np.tile(dow, (1, N, 1))

        return np.concatenate([speed, tod, dow], axis=-1)

    def load_pems_file(self, file_path: str):
        """
        Universal PeMS File Reader (.npz, .h5, .csv, .npy)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PeMS file not found: {file_path}")

        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".npz":
            npz = np.load(file_path)
            keys = list(npz.keys())
            key = "data" if "data" in keys else ("speed" if "speed" in keys else keys[0])
            raw_array = npz[key]
        elif ext == ".npy":
            raw_array = np.load(file_path)
        elif ext == ".csv":
            raw_array = pd.read_csv(file_path).values
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        processed = self.preprocess_pems_array(raw_array)
        self.num_nodes = processed.shape[1]
        self.data_array = processed
        self.load_dataset(f"custom_{self.num_nodes}")
        print(f"[+] Loaded Universal PeMS File {file_path}: Shape {processed.shape}")
        return processed

    def predict_next_15min(self, history_tensor):
        """
        Runs live PyTorch forward pass for Graph WaveNet.
        Input shape: (batch_size, 3, num_nodes, 12)
        Output shape: (batch_size, 12, num_nodes)
        """
        if not isinstance(history_tensor, torch.Tensor):
            history_tensor = torch.tensor(history_tensor, dtype=torch.float32)
            
        history_tensor = history_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(history_tensor)
            
        return output.cpu().numpy()

if __name__ == "__main__":
    adapter = UniversalPeMSAdapter()
    print("Testing Universal PeMS Adapter across multiple dataset shapes...")
    for ds_name in ["metr_la", "sd400", "pems04", "pems08", "pems_bay", "pems03", "pems07"]:
        adapter.load_dataset(ds_name)
        dummy_x = np.random.randn(1, 3, adapter.num_nodes, 12).astype(np.float32)
        preds = adapter.predict_next_15min(dummy_x)
        print(f"  • {ds_name.upper():<10} -> Nodes N: {adapter.num_nodes:<4} | Inference Pass Success: {preds.shape}")
