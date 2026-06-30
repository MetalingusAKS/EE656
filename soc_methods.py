
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from skimage import data as skdata
from skimage.transform import resize

from soc_core import normalize, run_imc_pass, soc_cluster, all_indices



def imc1_cluster(X, nk):

    X_norm = normalize(X)
    idx, cc, _, _ = run_imc_pass(X_norm, nk, np.ones(nk))
    return idx, cc * (X.max(0) - X.min(0) + 1e-10) + X.min(0)


def imc2_cluster(X, nk):

    X_norm = normalize(X)
    beta = np.array([nk / (m + 1) for m in range(nk)], dtype=float)
    idx, cc, _, _ = run_imc_pass(X_norm, nk, beta)
    return idx, cc * (X.max(0) - X.min(0) + 1e-10) + X.min(0)


def kmeans_cluster(X, nk):

    km = KMeans(n_clusters=nk, n_init=10, random_state=42)
    idx = km.fit_predict(X)
    return idx, km.cluster_centers_



def find_optimum_k(X, k_range=range(2, 8)):

    X_norm = normalize(X)
    gsi_map = {}
    for k in k_range:
        try:
            idx, _, _, _ = run_imc_pass(X_norm, k, np.ones(k))
            if len(np.unique(idx)) >= 2:
                sw = silhouette_samples(X_norm, idx)
                gsi_map[k] = float(sw.mean())
        except Exception:
            pass
    best_k = max(gsi_map, key=gsi_map.get) if gsi_map else list(k_range)[0]
    return best_k, gsi_map



def segment_image(X_flat, labels, shape):

    out = np.zeros_like(X_flat, dtype=np.float32)
    for lbl in np.unique(labels):
        mask = labels == lbl
        out[mask] = X_flat[mask].mean(0)
    return np.clip(out.reshape(shape[0], shape[1], 3), 0, 255).astype(np.uint8)



def run_full_analysis():

    img = skdata.rocket()
    img = resize(img, (48, 48), anti_aliasing=True)
    X = (img * 255).astype(np.float32).reshape(-1, 3)
    shape = img.shape[:2]
    print(f"Image: {shape}, pixels={len(X)}")

    # ── Step 1: Find optimum k ────────────────────────────────────────────────
    print("\nFinding optimum k (SOC, Section III-D)...")
    best_k, gsi_by_k = find_optimum_k(X, range(2, 8))
    print(f"  {gsi_by_k}\n  Optimum k={best_k}")

    # ── Step 2: Run all methods at best_k ────────────────────────────────────
    print(f"\nRunning all methods at k={best_k}...")
    soc_idx, soc_cc, soc_hist = soc_cluster(X, best_k)
    imc1_idx, imc1_cc         = imc1_cluster(X, best_k)
    imc2_idx, imc2_cc         = imc2_cluster(X, best_k)
    km_idx, km_cc             = kmeans_cluster(X, best_k)

    methods = {
        'SOC':     (soc_idx,  soc_cc),
        'IMC-1':   (imc1_idx, imc1_cc),
        'IMC-2':   (imc2_idx, imc2_cc),
        'K-Means': (km_idx,   km_cc),
    }

    # ── Step 3: Validity indices at best_k ───────────────────────────────────
    rows = []
    for name, (idx, cc) in methods.items():
        GSI, PI, SI, DI = all_indices(X, idx, cc)
        rows.append({'Method': name, 'GSI↑': GSI, 'PI↓': PI, 'SI↓': SI, 'DI↑': DI})
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))

    # ── Step 4: Scan k=2..7 for all methods ──────────────────────────────────
    print("\nScanning k=2..7 for all methods...")
    scan = []
    for k in range(2, 8):
        try:
            s_idx, s_cc, _   = soc_cluster(X, k)
            i1_idx, i1_cc    = imc1_cluster(X, k)
            i2_idx, i2_cc    = imc2_cluster(X, k)
            km_idx2, km_cc2  = kmeans_cluster(X, k)

            s_g,  s_p,  s_s,  s_d  = all_indices(X, s_idx,   s_cc)
            i1_g, i1_p, i1_s, i1_d = all_indices(X, i1_idx,  i1_cc)
            i2_g, i2_p, i2_s, i2_d = all_indices(X, i2_idx,  i2_cc)
            km_g, km_p, km_s, km_d = all_indices(X, km_idx2, km_cc2)

            scan.append({
                'k': k,
                'SOC_GSI': s_g,  'SOC_PI': s_p,  'SOC_SI': s_s,  'SOC_DI': s_d,
                'IMC1_GSI': i1_g,'IMC1_PI': i1_p,'IMC1_SI': i1_s,'IMC1_DI': i1_d,
                'IMC2_GSI': i2_g,'IMC2_PI': i2_p,'IMC2_SI': i2_s,'IMC2_DI': i2_d,
                'KM_GSI':  km_g, 'KM_PI':  km_p, 'KM_SI':  km_s, 'KM_DI':  km_d,
            })
        except Exception as e:
            print(f"  k={k} err: {e}")

    df_scan = pd.DataFrame(scan)
    print(df_scan[['k', 'SOC_GSI', 'IMC1_GSI', 'IMC2_GSI', 'KM_GSI']].round(4).to_string(index=False))

    return X, shape, img, methods, best_k, gsi_by_k, df, df_scan, soc_hist
