import argparse
import collections
import math
import os
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

try:
    import cuml
    from cuml.cluster import KMeans as GPUKMeans
    from cuml.cluster import DBSCAN as GPUDBSCAN
    from cuml.manifold import TSNE as GPUTSNE
    from cuml.manifold import UMAP as GPUUMAP

    HAS_CUML = True
except ImportError:
    HAS_CUML = False

IP_PROTO_MAP = {1: "ICMP", 6: "TCP", 17: "UDP", 2: "IGMP", 58: "ICMPv6"}
TCP_FLAGS = ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"]
COMMON_PORTS = {20, 21, 22, 23, 25, 53, 67, 68, 69, 80, 110, 123,
                143, 161, 162, 389, 443, 445, 465, 514, 587, 636,
                873, 993, 995, 1080, 1433, 1521, 2049, 3306, 3389,
                5432, 5900, 6379, 8080, 8443, 8888, 9090, 9200, 27017}
WELL_KNOWN_PORTS = sorted(COMMON_PORTS)


def list_interfaces():
    try:
        from scapy.all import get_if_list
        return get_if_list()
    except ImportError:
        return ["scapy not installed"]


def capture_packets(interface=None, count=5000, timeout=90, bpf_filter=""):
    from scapy.all import sniff, conf
    if interface:
        conf.iface = interface
    print(f"[*] 开始抓包: interface={interface or 'default'}, count={count}, timeout={timeout}s")
    if bpf_filter:
        print(f"[*] BPF filter: {bpf_filter}")
    packets = sniff(count=count, timeout=timeout, filter=bpf_filter, store=True)
    print(f"[*] 抓包完成, 共 {len(packets)} 个包")
    return packets


def load_pcap(filepath):
    from scapy.all import rdpcap
    packets = rdpcap(filepath)
    print(f"[*] 从 {filepath} 加载 {len(packets)} 个包")
    return packets


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _calc_entropy(counts):
    c = np.array(list(counts), dtype=np.float64)
    c = c[c > 0]
    total = np.sum(c)
    if total == 0:
        return 0.0
    p = c / total
    return -np.sum(p * np.log2(p))


def extract_features(packets, window_size=10):
    n = len(packets)
    if n == 0:
        return np.zeros((0, 55)), []
    w = max(2, window_size)
    raw_base = np.zeros((n, 20))
    meta = []
    prev_ts = None
    prev_ip_id = None
    protocol_onehot = np.zeros((n, 5))

    for i, pkt in enumerate(packets):
        info = {"idx": i}
        ts = _safe_float(pkt.time)
        info["timestamp"] = ts
        pkt_len = len(pkt) if pkt else 0
        raw_base[i, 0] = np.log1p(pkt_len)

        ip = pkt.getlayer("IP")
        ip_proto = 0
        if ip:
            raw_base[i, 1] = np.log1p(ip.len if ip.len else pkt_len)
            raw_base[i, 2] = ip.ttl if ip.ttl else 64
            ip_proto = ip.proto if ip.proto else 0
            raw_base[i, 3] = ip_proto
            info["src_ip"] = ip.src
            info["dst_ip"] = ip.dst
            ip_flags = int(getattr(ip, "flags", 0) or 0)
            raw_base[i, 4] = float(ip_flags)
            curr_ip_id = ip.id if ip.id else 0
            if prev_ip_id is not None:
                raw_base[i, 5] = min(abs(curr_ip_id - prev_ip_id), 65535) / 65535.0
            else:
                raw_base[i, 5] = 0.0
            prev_ip_id = curr_ip_id
            raw_base[i, 6] = (ip.frag if ip.frag else 0) / 8192.0
            raw_base[i, 7] = (ip.tos if ip.tos else 0) / 255.0

            if ip_proto == 6:
                protocol_onehot[i, 0] = 1
            elif ip_proto == 17:
                protocol_onehot[i, 1] = 1
            elif ip_proto == 1:
                protocol_onehot[i, 2] = 1
            elif ip_proto == 2:
                protocol_onehot[i, 3] = 1
            elif ip_proto == 58:
                protocol_onehot[i, 4] = 1
        else:
            raw_base[i, 1] = np.log1p(pkt_len)
            raw_base[i, 2] = 64
            raw_base[i, 3] = 0
            raw_base[i, 4] = 0.0
            raw_base[i, 5] = 0.0
            raw_base[i, 6] = 0.0
            raw_base[i, 7] = 0.0
            info["src_ip"] = ""
            info["dst_ip"] = ""

        tcp = pkt.getlayer("TCP")
        udp = pkt.getlayer("UDP")
        sport, dport, flags_val, tcp_win = 0, 0, 0, 0
        if tcp:
            sport = tcp.sport
            dport = tcp.dport
            flags_val = int(tcp.flags) if tcp.flags else 0
            tcp_win = tcp.window if tcp.window else 0
            info["protocol"] = "TCP"
        elif udp:
            sport = udp.sport
            dport = udp.dport
            info["protocol"] = "UDP"
        else:
            info["protocol"] = IP_PROTO_MAP.get(int(ip_proto), f"P{int(ip_proto)}")

        raw_base[i, 8] = np.log1p(sport)
        raw_base[i, 9] = np.log1p(dport)
        raw_base[i, 10] = 1.0 if dport in WELL_KNOWN_PORTS else 0.0
        raw_base[i, 11] = 1.0 if sport in WELL_KNOWN_PORTS else 0.0
        raw_base[i, 12] = float(flags_val) / 255.0
        raw_base[i, 13] = np.log1p(tcp_win)
        info["sport"] = sport
        info["dport"] = dport
        info["tcp_flags"] = int(flags_val)

        raw_layer = pkt.getlayer("Raw")
        payload_len = 0
        if raw_layer:
            payload_len = len(raw_layer.load)
        elif hasattr(pkt, "payload") and not ip:
            try:
                payload_len = len(pkt.payload)
            except Exception:
                payload_len = 0
        raw_base[i, 14] = np.log1p(payload_len)

        if raw_layer and len(raw_layer.load) > 0:
            raw_bytes = bytes(raw_layer.load[:48])
            raw_base[i, 15] = _calc_entropy(collections.Counter(raw_bytes).values()) / 8.0
        else:
            raw_base[i, 15] = 0.0

        raw_base[i, 16] = float(_guess_direction(pkt, info))
        info["direction"] = int(raw_base[i, 16])

        if prev_ts is not None:
            delta_ms = max(0, (ts - prev_ts) * 1000)
            delta_ms = min(delta_ms, 10000)
            raw_base[i, 17] = delta_ms / 10000.0
            raw_base[i, 18] = np.log1p(delta_ms) / np.log1p(10000)
        else:
            raw_base[i, 17] = 0.0
            raw_base[i, 18] = 0.0
        prev_ts = ts

        info["flow_key"] = (info.get("src_ip", ""), info.get("dst_ip", ""),
                            sport, dport, info.get("protocol", ""))
        meta.append(info)

    win_features = np.zeros((n, 19))
    pkt_lens = raw_base[:, 0]
    iats = raw_base[:, 17]
    payloads = raw_base[:, 14]
    tcp_flags_norm = raw_base[:, 12]
    directions = raw_base[:, 16]
    dports = raw_base[:, 9]
    sports = raw_base[:, 8]

    for i in range(n):
        start = max(0, i - w + 1)
        end = i + 1

        seg = pkt_lens[start:end]
        win_features[i, 0] = np.mean(seg)
        win_features[i, 1] = np.std(seg) if len(seg) > 1 else 0
        win_features[i, 2] = np.min(seg)
        win_features[i, 3] = np.max(seg)
        win_features[i, 4] = _safe_skew(seg)
        win_features[i, 5] = _safe_kurtosis(seg)

        seg2 = iats[start:end]
        seg2_nonzero = seg2[seg2 > 0]
        if len(seg2_nonzero) > 0:
            win_features[i, 6] = np.mean(seg2_nonzero)
            win_features[i, 7] = np.std(seg2_nonzero) if len(seg2_nonzero) > 1 else 0
            win_features[i, 8] = np.min(seg2_nonzero)
            win_features[i, 9] = np.max(seg2_nonzero)
            win_features[i, 10] = _safe_skew(seg2_nonzero)
            win_features[i, 11] = _safe_kurtosis(seg2_nonzero)
        else:
            win_features[i, 6:12] = 0.0

        seg3 = payloads[start:end]
        win_features[i, 12] = np.mean(seg3)
        win_features[i, 13] = np.std(seg3) if len(seg3) > 1 else 0
        win_features[i, 14] = np.mean(seg3 > 0)

        seg4 = tcp_flags_norm[start:end]
        win_features[i, 15] = np.mean(seg4 > 0.01)
        win_features[i, 16] = np.mean(seg4 > 0.07)

        seg5 = directions[start:end]
        win_features[i, 17] = np.mean(seg5)
        if len(seg5) > 1:
            win_features[i, 18] = np.sum(np.abs(np.diff(seg5))) / (len(seg5) - 1)
        else:
            win_features[i, 18] = 0.0

    port_features = np.zeros((n, 2))
    for i in range(n):
        start = max(0, i - w + 1)
        end = i + 1
        seg_dports = dports[start:end]
        seg_sports = sports[start:end]
        unique_dports = len(set(np.round(seg_dports, 4)))
        unique_ports = len(set(np.round(seg_sports, 4)))
        port_features[i, 0] = unique_dports / max(1, len(seg_dports))
        port_features[i, 1] = unique_ports / max(1, len(seg_sports))

    flow_features = _compute_flow_features(packets, meta, n)

    features = np.hstack([
        raw_base,
        protocol_onehot,
        win_features,
        port_features,
        flow_features,
    ])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32), meta


def _safe_skew(arr):
    if len(arr) < 3: return 0.0
    std = np.std(arr)
    if std < 1e-10: return 0.0
    mean = np.mean(arr)
    n = len(arr)
    skew = (n / ((n - 1) * (n - 2))) * np.sum(((arr - mean) / std) ** 3)
    return np.clip(float(skew), -5, 5)


def _safe_kurtosis(arr):
    if len(arr) < 4: return 0.0
    std = np.std(arr)
    if std < 1e-10: return 0.0
    mean = np.mean(arr)
    n = len(arr)
    kurt = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((arr - mean) / std) ** 4)
    kurt -= (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return np.clip(float(kurt), -5, 10)


def _compute_flow_features(packets, meta, n):
    flow_features = np.zeros((n, 10))
    flow_packets = defaultdict(list)
    for i, m in enumerate(meta):
        flow_key = m.get("flow_key", ("", "", 0, 0, ""))
        flow_packets[flow_key].append(i)

    for flow_key, indices in flow_packets.items():
        if len(indices) < 1: continue
        flow_pkts = [packets[i] for i in indices]
        pkt_count = len(flow_pkts)
        total_bytes = sum(len(p) for p in flow_pkts)
        timestamps = [_safe_float(p.time) for p in flow_pkts]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.001

        flow_iats = []
        for k in range(1, len(timestamps)):
            dt = max(0, timestamps[k] - timestamps[k - 1]) * 1000
            flow_iats.append(min(dt, 10000))
        flow_iat_mean = np.mean(flow_iats) if flow_iats else 0
        flow_iat_std = np.std(flow_iats) if len(flow_iats) > 1 else 0

        payload_lens = []
        for p in flow_pkts:
            pl = 0
            raw = p.getlayer("Raw")
            if raw:
                pl = len(raw.load)
            elif hasattr(p, "payload") and not p.getlayer("IP"):
                try:
                    pl = len(p.payload)
                except Exception:
                    pl = 0
            payload_lens.append(pl)
        flow_payload_ratio = np.mean([1.0 if pl > 0 else 0.0 for pl in payload_lens])

        syn_count = 0
        psh_count = 0
        dir_values = []
        for p in flow_pkts:
            tcp = p.getlayer("TCP")
            if tcp:
                flags = int(tcp.flags) if tcp.flags else 0
                if flags & 0x02: syn_count += 1
                if flags & 0x08: psh_count += 1
        syn_ratio = syn_count / pkt_count if pkt_count > 0 else 0
        psh_ratio = psh_count / pkt_count if pkt_count > 0 else 0

        for idx in indices:
            dir_values.append(meta[idx].get("direction", 1))
        flow_direction_ratio = np.mean(dir_values) if dir_values else 0.5

        flow_pkt_sizes = [len(p) for p in flow_pkts]
        flow_entropy = _calc_entropy(collections.Counter(flow_pkt_sizes).values())

        for idx in indices:
            flow_features[idx, 0] = np.log1p(pkt_count)
            flow_features[idx, 1] = np.log1p(total_bytes)
            flow_features[idx, 2] = np.log1p(duration * 1000)
            flow_features[idx, 3] = flow_iat_mean / 10000.0
            flow_features[idx, 4] = flow_iat_std / 10000.0
            flow_features[idx, 5] = flow_payload_ratio
            flow_features[idx, 6] = syn_ratio
            flow_features[idx, 7] = psh_ratio
            flow_features[idx, 8] = flow_direction_ratio
            flow_features[idx, 9] = flow_entropy / 8.0 if flow_entropy > 0 else 0.0

    return flow_features


def _guess_direction(pkt, info):
    eth = pkt.getlayer("Ether")
    if eth:
        src_mac = eth.src
        local_prefixes = ["00:15:5d", "00:50:56", "00:0c:29", "08:00:27", "00:1c:42", "0a:00:27"]
        for prefix in local_prefixes:
            if src_mac.startswith(prefix):
                return 0
    return 1


def normalize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1.0
    normalized = (features - mean) / std
    return normalized, mean, std


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class TrafficVAE(nn.Module):
    def __init__(self, input_dim=55, latent_dim=16, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        enc_layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            enc_layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualBlock(hdim, dropout),
            ])
            prev_dim = hdim
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        dec_layers = []
        prev_dim = latent_dim
        for hdim in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualBlock(hdim, dropout),
            ])
            prev_dim = hdim
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu


def vae_loss(reconstructed, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(reconstructed, x, reduction="sum") / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_vae(
        features, epochs=200, batch_size=128, lr=1e-3, latent_dim=16,
        patience=20, device="cpu", beta_max=1.0, beta_warmup_epochs=50,
        verbose=True, weight_decay=1e-5
):
    n = len(features)
    if n < batch_size * 2:
        batch_size = max(1, n // 2)

    split = int(n * 0.8)
    indices = np.random.permutation(n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = torch.tensor(features[train_idx], dtype=torch.float32).to(device)
    X_val = torch.tensor(features[val_idx], dtype=torch.float32).to(device)
    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)

    model = TrafficVAE(input_dim=features.shape[1], latent_dim=latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": []}

    for epoch in range(epochs):
        beta = beta_max * (epoch / max(1, beta_warmup_epochs)) if epoch < beta_warmup_epochs else beta_max
        model.train()
        train_total, train_recon, train_kl = 0.0, 0.0, 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(reconstructed, batch, mu, logvar, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_total += loss.item() * len(batch)
            train_recon += recon_loss.item() * len(batch)
            train_kl += kl_loss.item() * len(batch)

        train_total /= len(train_idx)
        train_recon /= len(train_idx)
        train_kl /= len(train_idx)
        history["train_loss"].append(train_total)
        history["recon_loss"].append(train_recon)
        history["kl_loss"].append(train_kl)
        scheduler.step()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                reconstructed, mu, logvar = model(batch)
                loss, _, _ = vae_loss(reconstructed, batch, mu, logvar, beta=beta)
                val_total += loss.item() * len(batch)
        val_total /= len(val_idx)
        history["val_loss"].append(val_total)

        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(
                f"  Epoch {epoch:3d}/{epochs} | β={beta:.2f} | train={train_total:.6f} | val={val_total:.6f} | recon={train_recon:.4f} | kl={train_kl:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if val_total < best_val_loss - 1e-6:
            best_val_loss = val_total
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose: print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    X_all = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        latent = model.get_latent(X_all).cpu().numpy()

    return model, latent, history


def cluster_kmeans(latent, k=None, k_range=(2, 15)):
    n = len(latent)
    max_k = min(k_range[1], n - 1) if n > 1 else 1
    min_k = max(2, k_range[0])

    KMeansClass = GPUKMeans if HAS_CUML else KMeans

    if k is not None:
        best_k = k
        print(f"[*] K-Means: 使用指定 K={best_k}")
    else:
        best_k = min_k
        best_score = -1
        print(f"\n[*] K-Means 自动搜索最佳 K ({min_k} ~ {max_k}):")
        for cand_k in range(min_k, max_k + 1):
            km = KMeansClass(n_clusters=cand_k, random_state=42, n_init=10)
            labels = km.fit_predict(latent)
            if len(set(labels)) < 2: continue
            sil = silhouette_score(latent, labels)
            db = davies_bouldin_score(latent, labels)
            ch = calinski_harabasz_score(latent, labels)
            score = sil - 0.3 * db + 0.01 * np.log1p(ch) / 1000
            print(f"    K={cand_k:2d} | Sil={sil:.4f} | DB={db:.4f} | CH={ch:.1f} | Score={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = cand_k
    print(f"\n[*] K-Means 最终选择 K={best_k}")
    km = KMeansClass(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(latent)
    scores = _evaluate_clustering(latent, labels)
    return labels, best_k, scores


def cluster_gmm(latent, n_components=None, n_range=(2, 15), covariance_type="full"):
    n = len(latent)
    max_n = min(n_range[1], n - 1, 20)
    min_n = max(2, n_range[0])
    if n_components is not None:
        best_n = n_components
        print(f"[*] GMM: 使用指定 n_components={best_n}")
    else:
        best_n = min_n
        best_bic = float("inf")
        print(f"\n[*] GMM 自动搜索最佳组件数 ({min_n} ~ {max_n}):")
        for cand_n in range(min_n, max_n + 1):
            gmm = GaussianMixture(n_components=cand_n, covariance_type=covariance_type, random_state=42, max_iter=200,
                                  n_init=3, reg_covar=1e-4)
            gmm.fit(latent)
            bic = gmm.bic(latent)
            aic = gmm.aic(latent)
            print(f"    n={cand_n:2d} | BIC={bic:.1f} | AIC={aic:.1f}")
            if bic < best_bic:
                best_bic = bic
                best_n = cand_n
    print(f"\n[*] GMM 最终选择 n_components={best_n}")
    gmm = GaussianMixture(n_components=best_n, covariance_type=covariance_type, random_state=42, max_iter=300, n_init=5,
                          reg_covar=1e-4)
    labels = gmm.fit_predict(latent)
    scores = _evaluate_clustering(latent, labels)
    scores["bic"] = float(gmm.bic(latent))
    scores["aic"] = float(gmm.aic(latent))
    scores["soft_probs"] = gmm.predict_proba(latent)
    return labels, best_n, scores


def cluster_hdbscan(latent, min_cluster_size=None, min_samples=None):
    try:
        import hdbscan
    except ImportError:
        print("[!] hdbscan 未安装, 回退到 DBSCAN")
        return cluster_dbscan(latent)
    n = len(latent)
    if min_cluster_size is None:
        min_cluster_size = max(5, int(n * 0.02))
    if min_samples is None:
        min_samples = max(1, min_cluster_size // 2)
    print(f"\n[*] HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                cluster_selection_epsilon=0.0, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(latent)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"[*] HDBSCAN: {n_clusters} 簇, {n_noise} 噪声点")
    mask = labels != -1
    if mask.sum() > 1 and n_clusters > 1:
        scores = _evaluate_clustering(latent[mask], labels[mask])
    else:
        scores = {"silhouette": 0, "davies_bouldin": 0, "calinski_harabasz": 0}
    scores["n_clusters"] = n_clusters
    scores["n_noise"] = n_noise
    scores["probabilities"] = getattr(clusterer, "probabilities_", None)
    return labels, n_clusters, scores


def cluster_dbscan(latent, eps=None, min_samples=15):
    if eps is None:
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=min(2 * min_samples, len(latent)))
        neigh.fit(latent)
        distances, _ = neigh.kneighbors(latent)
        k_dist = np.sort(distances[:, min_samples - 1])
        eps = float(np.percentile(k_dist, 98))
        if eps < 0.2:
            eps = 0.3
        print(f"[*] DBSCAN 自动调整后 eps = {eps:.4f}, min_samples = {min_samples}")

    DBSCANClass = GPUDBSCAN if HAS_CUML else DBSCAN
    db = DBSCANClass(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(latent)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"[*] DBSCAN: {n_clusters} 簇, {n_noise} 噪声点")
    mask = labels != -1
    if mask.sum() > 1 and n_clusters > 1:
        scores = _evaluate_clustering(latent[mask], labels[mask])
    else:
        scores = {"silhouette": 0, "davies_bouldin": 0, "calinski_harabasz": 0}
    scores["n_clusters"] = n_clusters
    scores["n_noise"] = n_noise
    scores["eps"] = eps
    return labels, n_clusters, scores


def _evaluate_clustering(X, labels):
    n_labels = len(set(labels))
    if n_labels < 2 or n_labels >= len(X):
        return {"silhouette": 0, "davies_bouldin": 0, "calinski_harabasz": 0}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def visualize_clusters(latent, labels, method="tsne", save_path=None, title="Traffic Clusters"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import collections

    n = len(latent)
    if n < 2:
        print("[!] 样本数不足, 跳过可视化")
        return

    if method == "umap":
        try:
            if HAS_CUML:
                reducer = GPUUMAP(n_components=2, random_state=42, n_neighbors=min(15, n - 1), min_dist=0.1)
            else:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, n - 1), min_dist=0.1,
                                    metric="cosine")
            embedding = reducer.fit_transform(latent)
        except ImportError:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            embedding = reducer.fit_transform(latent)
    elif method == "tsne":
        perplexity = min(30, max(5, (n - 1) // 3))
        if HAS_CUML:
            reducer = GPUTSNE(n_components=2, perplexity=perplexity, random_state=42, learning_rate_ratio=None)
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, learning_rate="auto", init="pca",
                           max_iter=1000)
        embedding = reducer.fit_transform(latent)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(latent)

    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)

    fig_width = max(16.0, n_clusters * 0.5)
    fig_height = max(8.0, n_clusters * 0.15)

    palette = sns.color_palette("husl", max(n_clusters, 2))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    ax = axes[0]
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = palette[i] if label != -1 else (0.5, 0.5, 0.5)
        label_name = f"Cluster {label}" if label != -1 else "Noise"
        ax.scatter(embedding[mask, 0], embedding[mask, 1], c=[color], label=label_name, alpha=0.7, s=20,
                   edgecolors="none")

    ax.set_title(f"{title} - {method.upper()} Projection")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    ncol_legend = max(1, n_clusters // 25)
    ax.legend(markerscale=2, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5), ncol=ncol_legend)

    ax = axes[1]
    cluster_counts = collections.Counter(labels)
    sorted_labels = sorted(cluster_counts.keys())
    counts = [cluster_counts[l] for l in sorted_labels]
    bar_labels = [f"C{l}" if l != -1 else "Noise" for l in sorted_labels]
    bar_colors = [palette[i] if l != -1 else (0.5, 0.5, 0.5) for i, l in enumerate(sorted_labels)]

    ax.bar(range(len(sorted_labels)), counts, tick_label=bar_labels, color=bar_colors)
    ax.set_title("Cluster Size Distribution")
    ax.set_ylabel("Number of Packets")
    ax.tick_params(axis='x', rotation=90)

    for i, v in enumerate(counts):
        ax.text(i, v + max(counts) * 0.01, str(v), ha="center", fontsize=8, rotation=90)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[*] 可视化已保存至 {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_ip_cluster_matrix(meta, labels, output_dir="output"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    from collections import defaultdict

    os.makedirs(output_dir, exist_ok=True)

    ip_cluster = defaultdict(lambda: defaultdict(int))
    all_clusters = set()
    for m, label in zip(meta, labels):
        dst_ip = m.get("dst_ip", "unknown") or "unknown"
        ip_cluster[dst_ip][label] += 1
        all_clusters.add(label)

    ip_totals = {ip: sum(clusters.values()) for ip, clusters in ip_cluster.items()}
    all_ips = sorted(ip_totals, key=ip_totals.get, reverse=True)
    sorted_clusters = sorted(all_clusters)

    n_ips = len(all_ips)
    n_clusters = len(sorted_clusters)

    fig_width = max(20.0, n_ips * 0.4 + n_clusters * 0.4)
    fig_height = max(10.0, max(n_clusters, n_ips) * 0.2)

    matrix = np.zeros((n_ips, n_clusters), dtype=int)
    for i, ip in enumerate(all_ips):
        for j, cl in enumerate(sorted_clusters):
            matrix[i, j] = ip_cluster[ip].get(cl, 0)

    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    matrix_cluster_pct = (matrix / col_sums * 100).T

    palette = sns.color_palette("husl", max(n_clusters, 2))
    cluster_colors = {cl: palette[i] if cl != -1 else (0.5, 0.5, 0.5) for i, cl in enumerate(sorted_clusters)}

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    ax = axes[0]
    x = np.arange(n_ips)
    bottom = np.zeros(n_ips)

    for j, cl in enumerate(sorted_clusters):
        color = cluster_colors[cl]
        label_name = f"Cluster {cl}" if cl != -1 else "Noise"
        ax.bar(x, matrix[:, j], bottom=bottom, color=color, label=label_name, width=0.8)
        bottom += matrix[:, j]

    ax.set_xticks(x)
    ax.set_xticklabels(all_ips, rotation=90, ha="center", fontsize=8)
    ax.set_ylabel("Packet Count")
    ax.set_title("Destination IP -> Traffic Cluster Composition")

    ncol_legend_0 = max(1, n_clusters // 20)
    ax.legend(markerscale=1.5, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5), ncol=ncol_legend_0)

    ax = axes[1]
    x = np.arange(n_clusters)
    bottom = np.zeros(n_clusters)

    ip_colors = sns.color_palette("husl", max(n_ips, 2))

    for i, ip in enumerate(all_ips):
        ax.bar(x, matrix_cluster_pct[:, i], bottom=bottom, color=ip_colors[i], label=ip, width=0.8)
        bottom += matrix_cluster_pct[:, i]

    cl_labels = [f"C{c}" if c != -1 else "Noise" for c in sorted_clusters]
    ax.set_xticks(x)
    ax.set_xticklabels(cl_labels, fontsize=8, rotation=90)
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Traffic Cluster -> Destination IP Distribution (100% Stacked)")

    ncol_legend_1 = max(1, n_ips // 20)
    ax.legend(markerscale=1.5, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5), ncol=ncol_legend_1)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "ip_cluster_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[*] IP-簇矩阵图已保存至 {save_path}")


def generate_report(labels, meta, kmeans_scores=None, gmm_scores=None, hdbscan_scores=None):
    n = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    print("\n" + "=" * 70)
    print("            TrafficSniper v2 — 聚类分析报告")
    print("=" * 70)
    print(f"  总样本数        : {n}")
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  发现簇数        : {n_clusters}")
    if -1 in labels:
        print(f"  噪声点          : {int(np.sum(labels == -1))}")
    print(f"\n  簇分布:")
    for label, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        tag = " (Noise)" if label == -1 else ""
        pct = 100 * cnt / n
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        print(f"    Cluster {label:3d}{tag}: {cnt:6d} ({pct:5.1f}%) {bar}")
    print(f"\n  各簇详细分析 (Top 协议 + 端口 + IP):")
    for label in sorted(unique):
        cluster_meta = [m for m, l in zip(meta, labels) if l == label]
        if not cluster_meta: continue
        proto_counter = collections.Counter(m.get("protocol", "?") for m in cluster_meta)
        dport_counter = collections.Counter(m.get("dport", 0) for m in cluster_meta)
        dst_ip_counter = collections.Counter(m.get("dst_ip", "?") for m in cluster_meta)
        top_protos = proto_counter.most_common(3)
        top_ports = dport_counter.most_common(3)
        top_ips = dst_ip_counter.most_common(3)
        protos_str = ", ".join(f"{p}({c})" for p, c in top_protos)
        ports_str = ", ".join(f"{p}({c})" for p, c in top_ports)
        ips_str = ", ".join(f"{ip}({c})" for ip, c in top_ips)
        tag = " (Noise)" if label == -1 else ""
        print(f"    Cluster {label:3d}{tag}:")
        print(f"      proto = [{protos_str}]")
        print(f"      dport = [{ports_str}]")
        print(f"      dst_ip = [{ips_str}]")
    if kmeans_scores:
        print(f"\n  K-Means 评估:")
        _print_scores(kmeans_scores)
    if gmm_scores:
        print(f"\n  GMM 评估:")
        _print_scores(gmm_scores)
        if "bic" in gmm_scores:
            print(f"    BIC                : {gmm_scores['bic']:.1f}")
            print(f"    AIC                : {gmm_scores['aic']:.1f}")
    if hdbscan_scores:
        print(f"\n  HDBSCAN/DBSCAN 评估:")
        _print_scores(hdbscan_scores)
    print("=" * 70)


def _print_scores(scores):
    print(f"    Silhouette        : {scores.get('silhouette', 0):.4f}")
    print(f"    Davies-Bouldin    : {scores.get('davies_bouldin', 0):.4f}")
    print(f"    Calinski-Harabasz : {scores.get('calinski_harabasz', 0):.2f}")


def export_results(labels_dict, latent, meta, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "cluster_results.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["idx", "timestamp", "src_ip", "dst_ip", "sport", "dport", "protocol", "tcp_flags", "direction"]
        for method_name in labels_dict:
            header.append(f"cluster_{method_name}")
        writer.writerow(header)
        for i, m in enumerate(meta):
            row = [
                i, m.get("timestamp", ""), m.get("src_ip", ""), m.get("dst_ip", ""),
                m.get("sport", 0), m.get("dport", 0), m.get("protocol", ""),
                m.get("tcp_flags", 0), m.get("direction", 0),
            ]
            for method_name in labels_dict:
                row.append(int(labels_dict[method_name][i]))
            writer.writerow(row)
    print(f"[*] CSV 结果已导出至 {csv_path}")
    np_path = os.path.join(output_dir, "latent_features.npy")
    np.save(np_path, latent)
    print(f"[*] 潜在特征已导出至 {np_path}")


def run_pipeline(
        interface=None, pcap_file=None, count=5000, timeout=60, bpf_filter="",
        window_size=10, latent_dim=16, vae_epochs=200, vae_lr=1e-3, n_clusters=None,
        hdbscan_min_cluster_size=None, hdbscan_min_samples=None, vis_method="tsne",
        output_dir="output", device="auto"
):
    print("\n" + "*" * 60)
    print("* TrafficSniper v2 — 深度学习无监督流量聚类")
    print("*" * 60)

    if device == "auto":
        if torch.cuda.is_available():
            device_torch = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_torch = "mps"
        else:
            device_torch = "cpu"
    else:
        device_torch = device

    if pcap_file:
        packets = load_pcap(pcap_file)
    else:
        packets = capture_packets(interface=interface, count=count, timeout=timeout, bpf_filter=bpf_filter)
    if len(packets) < 10:
        print(f"[!] 数据包数量不足 ({len(packets)}), 无法进行有意义的聚类")
        return

    print(f"\n[Step 2] 精细化特征提取 (window_size={window_size})...")
    t0 = time.time()
    features, meta = extract_features(packets, window_size=window_size)
    print(f"  提取特征: {features.shape[0]} 条, {features.shape[1]} 维, 耗时 {time.time() - t0:.2f}s")
    features_norm, feat_mean, feat_std = normalize_features(features)

    print(f"\n[Step 3] 训练深度 VAE (latent_dim={latent_dim}, epochs={vae_epochs})...")
    t0 = time.time()

    if device_torch == "cuda":
        print(f"  PyTorch VAE 使用 GPU: {torch.cuda.get_device_name(0)}")
    elif device_torch == "mps":
        print("  PyTorch VAE 使用 Apple Silicon GPU (MPS)")
    else:
        print("  PyTorch VAE 使用 CPU")

    model, latent, history = train_vae(
        features_norm, epochs=vae_epochs, lr=vae_lr, latent_dim=latent_dim, device=device_torch
    )
    print(f"  VAE 训练完成, 潜在特征维度: {latent.shape[1]}, 耗时 {time.time() - t0:.2f}s")
    print(f"  最终 train_loss={history['train_loss'][-1]:.6f}, val_loss={history['val_loss'][-1]:.6f}")

    print(f"\n[Step 4] 无监督聚类 (K-Means + GMM + HDBSCAN)...")
    if HAS_CUML:
        print("  检测到 NVIDIA RAPIDS (cuML)，机器学习聚类与降维将使用 GPU 加速！")
    else:
        print("  未检测到 cuML，聚类与降维将回退至 CPU 运算。")

    all_labels = {}
    all_scores = {}

    print("\n  --- K-Means ---")
    km_labels, best_k, km_scores = cluster_kmeans(latent, k=n_clusters)
    all_labels["kmeans"] = km_labels
    all_scores["kmeans"] = km_scores

    print("\n  --- GMM (Gaussian Mixture Model) ---")
    gmm_labels, gmm_n, gmm_scores = cluster_gmm(latent)
    all_labels["gmm"] = gmm_labels
    all_scores["gmm"] = gmm_scores

    print("\n  --- HDBSCAN ---")
    hdb_labels, hdb_n, hdb_scores = cluster_hdbscan(
        latent, min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples
    )
    all_labels["hdbscan"] = hdb_labels
    all_scores["hdbscan"] = hdb_scores

    print(f"\n[Step 5] 可视化...")
    os.makedirs(output_dir, exist_ok=True)
    for method_name, method_labels in all_labels.items():
        if method_name == "gmm": continue
        vis_path = os.path.join(output_dir, f"clusters_{method_name}.png")
        visualize_clusters(
            latent, method_labels, method=vis_method, save_path=vis_path,
            title=f"Traffic Clusters ({method_name.upper()})"
        )

    best_method = "kmeans"
    best_sil = all_scores["kmeans"].get("silhouette", 0)
    for m in ["gmm", "hdbscan"]:
        sil = all_scores[m].get("silhouette", 0)
        if sil > best_sil:
            best_sil = sil
            best_method = m
    print(f"[*] IP-矩阵图使用最佳方法: {best_method} (Silhouette={best_sil:.4f})")
    visualize_ip_cluster_matrix(meta, all_labels[best_method], output_dir=output_dir)

    print(f"\n[Step 6] 生成报告...")
    generate_report(
        all_labels["kmeans"], meta, kmeans_scores=all_scores["kmeans"],
        gmm_scores=all_scores["gmm"], hdbscan_scores=all_scores["hdbscan"]
    )
    export_results(all_labels, latent, meta, output_dir=output_dir)
    model_path = os.path.join(output_dir, "vae_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[*] VAE 模型已保存至 {model_path}")
    print("\n[✓] TrafficSniper v2 分析完成!")

    return {
        "features": features, "latent": latent, "labels": all_labels, "scores": all_scores,
        "model": model, "meta": meta, "history": history
    }


def main():
    parser = argparse.ArgumentParser()
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument("-i", "--interface", type=str, default=None)
    src_group.add_argument("-f", "--pcap", type=str, default=None)
    parser.add_argument("-c", "--count", type=int, default=None)
    parser.add_argument("-t", "--timeout", type=int, default=None)
    parser.add_argument("--bpf", type=str, default="")
    parser.add_argument("-w", "--window", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--vae-epochs", type=int, default=200)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("-k", "--n-clusters", type=int, default=None)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=None)
    parser.add_argument("--hdbscan-min-samples", type=int, default=None)
    parser.add_argument("--vis-method", type=str, default="tsne", choices=["tsne", "umap", "pca"])
    parser.add_argument("-o", "--output-dir", type=str, default="output")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.count is None:
        user_count = input("请输入需要抓取的包数 (直接回车默认 5000): ")
        args.count = int(user_count) if user_count.strip() else 5000

    if args.timeout is None:
        user_timeout = input("请输入抓包超时时间/秒 (直接回车默认 60): ")
        args.timeout = int(user_timeout) if user_timeout.strip() else 60

    run_pipeline(
        interface=args.interface, pcap_file=args.pcap, count=args.count, timeout=args.timeout,
        bpf_filter=args.bpf, window_size=args.window, latent_dim=args.latent_dim,
        vae_epochs=args.vae_epochs, vae_lr=args.vae_lr, n_clusters=args.n_clusters,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size, hdbscan_min_samples=args.hdbscan_min_samples,
        vis_method=args.vis_method, output_dir=args.output_dir, device=args.device,
    )


if __name__ == "__main__":
    main()