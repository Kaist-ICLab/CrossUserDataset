import os
import time
import warnings
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import cloudpickle
import detectshift as ds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import ray
import seaborn as sns
import shap
import scipy.stats as st
import torch

from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from xgboost import XGBClassifier
from otdd.pytorch.distance import DatasetDistance
from scipy.spatial.distance import jensenshannon
from scipy import stats
from scipy.stats import wilcoxon, binomtest, spearmanr, pearsonr

__all__ = [
    "load",
    "dump",
    "select_top_xgb_features",
    "save_correlation_heatmaps",
    "compute_pairwise_wasserstein_matrix",
    "compute_pairwise_label_shift_matrix",
    "compute_pairwise_conditional_matrix",
    "compute_pairwise_concept_shift_matrix",
    "plot_distance_matrix_and_save",
    "create_train_test_splits",
    "normalize_data",
    "filter_users",
    "evaluate_users_with_fixed_splits",
    "wilcoxon_greater",
    "sign_test_greater",
    "fdr_bh_series"
]

DEFAULT_TZ = pytz.FixedOffset(540)  # GMT+09:00; Asia/Seoul

PATH_DATA = "/var/nfs_share/D#1"
PATH_ESM = os.path.join(PATH_DATA, 'EsmResponse.csv')
PATH_PARTICIPANT = os.path.join(PATH_DATA, 'UserInfo.csv')
PATH_SENSOR = os.path.join(PATH_DATA, 'Sensor')
PATH_RESULTS = '/var/nfs_share/Stress_Detection_D-1/Results'

PATH_INTERMEDIATE = '/var/nfs_share/Stress_Detection_D-1/Intermediate'
RANDOM_STATE = 42


seed = RANDOM_STATE
DATA_TYPES = {
    'Acceleration': 'ACC',
    'AmbientLight': 'AML',
    'Calorie': 'CAL',
    'Distance': 'DST',
    'EDA': 'EDA',
    'HR': 'HRT',
    'RRI': 'RRI',
    'SkinTemperature': 'SKT',
    'StepCount': 'STP',
    'UltraViolet': 'ULV',
    'ActivityEvent': 'ACE',
    'ActivityTransition': 'ACT',
    'AppUsageEvent': 'APP',
    'BatteryEvent': 'BAT',
    'CallEvent': 'CAE',
    'Connectivity': 'CON',
    'DataTraffic': 'DAT',
    'InstalledApp': 'INS',
    'Location': 'LOC',
    'MediaEvent': 'MED',
    'MessageEvent': 'MSG',
    'WiFi': 'WIF',
    'ScreenEvent': 'SCR',
    'RingerModeEvent': 'RNG',
    'ChargeEvent': 'CHG',
    'PowerSaveEvent': 'PWS',
    'OnOffEvent': 'ONF'
}


def load(path: str):
    """Load an object from a pickle file.
    
    Args:
        path (str): Path to the pickle file.
        
    Returns:
        The unpickled object.
    """
    with open(path, mode='rb') as f:
        return cloudpickle.load(f)

    
def dump(obj, path: str):
    """Dump an object to a pickle file.
    
    Args:
        obj: Object to pickle.
        path (str): Path to save the pickle file.
    """
    with open(path, mode='wb') as f:
        cloudpickle.dump(obj, f)
        
    
def log(msg: any):
    """Log a message with timestamp.
    
    Args:
        msg: Message to log.
    """
    print('[{}] {}'.format(datetime.now().strftime('%y-%m-%d %H:%M:%S'), msg))


def summary(x):
    """Generate summary statistics for numeric or categorical data.
    
    Args:
        x: Data to summarize (array-like).
        
    Returns:
        dict: Summary statistics including counts, means, ranges for numeric data
              or value counts and cardinality for categorical data.
    """
    x = np.asarray(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        n = len(x)
        if x.dtype.kind.isupper() or x.dtype.kind == 'b': 
            cnt = pd.Series(x).value_counts(dropna=False)
            card = len(cnt)
            cnt = cnt[:20]                
            cnt_str = ', '.join([f'{u}:{c}' for u, c in zip(cnt.index, cnt)])
            if card > 30:
                cnt_str = f'{cnt_str}, ...'
            return {
                'n': n,
                'cardinality': card,
                'value_count': cnt_str
            }
        else: 
            x_nan = x[np.isnan(x)]
            x_norm = x[~np.isnan(x)]
            
            tot = np.sum(x_norm)
            m = np.mean(x_norm)
            me = np.median(x_norm)
            s = np.std(x_norm, ddof=1)
            l, u = np.min(x_norm), np.max(x)
            conf_l, conf_u = st.t.interval(0.95, len(x_norm) - 1, loc=m, scale=st.sem(x_norm))
            n_nan = len(x_nan)
            
            return {
                'n': n,
                'sum': tot,
                'mean': m,
                'SD': s,
                'med': me,
                'range': (l, u),
                'conf.': (conf_l, conf_u),
                'nan_count': n_nan
            }
        
def _load_data(name: str) -> Optional[pd.DataFrame]:
    """Load sensor data files from all participants.
    
    Args:
        name (str): Name of the sensor data type to load.
        
    Returns:
        Optional[pd.DataFrame]: Combined dataframe with pcode and timestamp index.
    """
    paths = [
        (d, os.path.join(PATH_SENSOR, d, f'{name}.csv'))
        for d in os.listdir(PATH_SENSOR)
        if d.startswith('P')
    ]
    return pd.concat(
        filter(
            lambda x: len(x.index), 
            [
                pd.read_csv(p).assign(pcode=pcode)
                for pcode, p in paths
                if os.path.exists(p)
            ]
        ), ignore_index=True
    ).assign(
        timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='ms', utc=True).dt.tz_convert(DEFAULT_TZ)
    ).set_index(
        ['pcode', 'timestamp']
    )

@contextmanager
def on_ray(*args, **kwargs):
    """Context manager for Ray initialization and cleanup.
    
    Args:
        *args: Arguments to pass to ray.init().
        **kwargs: Keyword arguments to pass to ray.init().
    """
    try:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(*args, **kwargs)
        yield None
    finally:
        ray.shutdown()
       
        
transform = {
    'GAME': 'ENTER',
    'GAME_TRIVIA': 'ENTER',
    'GAME_CASINO': 'ENTER',
    'GAME-ACTION': 'ENTER',
    'GAME_SPORTS': 'ENTER',
    'GAME_PUZZLE': 'ENTER',
    'GAME_SIMULATION': 'ENTER',
    'GAME_STRATEGY': 'ENTER',
    'GAME_ROLE_PLAYING': 'ENTER',
    'GAME_ACTION': 'ENTER',
    'GAME_ARCADE': 'ENTER',
    'GAME_RACING': 'ENTER',
    'GAME_CASUAL': 'ENTER',
    'GAME_MUSIC': 'ENTER',
    'GAME_CARD': 'ENTER',
    'GAME_ADVENTURE': 'ENTER',
    'GAME_BOARD': 'ENTER',
    'GAME_EDUCATIONAL': 'ENTER',
    'GAME_RACING': 'ENTER',
    'PHOTOGRAPHY': 'ENTER',
    'ENTERTAINMENT': 'ENTER',
    'SPORTS': 'ENTER',
    'MUSIC_AND_AUDIO': 'ENTER',
    'COMICS': 'ENTER',
    'VIDEO_PLAYERS_AND_EDITORS': 'ENTER',
    'VIDEO_PLAYERS': 'ENTER',
    'ART_AND_DESIGN': 'ENTER',
    'TRAVEL_AND_LOCAL': 'INFO',
    'FOOD_AND_DRINK': 'INFO',
    'NEWS_AND_MAGAZINES': 'INFO',
    'MAPS_AND_NAVIGATION': 'INFO',
    'WEATHER': 'INFO',
    'HOUSE_AND_HOME': 'INFO',
    'BOOKS_AND_REFERENCE': 'INFO',
    'SHOPPING': 'INFO',
    'LIBRARIES_AND_DEMO': 'INFO',
    'BEAUTY': 'INFO',
    'AUTO_AND_VEHICLES': 'INFO',
    'LIFESTYLE': 'INFO',
    'PERSONALIZATION': 'SYSTEM',
    'TOOLS': 'SYSTEM',
    'COMMUNICATION': 'SOCIAL',
    'SOCIAL': 'SOCIAL',
    'DATING': 'SOCIAL',
    'PARENTING':'SOCIAL',
    'FINANCE': 'WORK',
    'BUSINESS': 'WORK',
    'PRODUCTIVITY': 'WORK',
    'EDUCATION': 'WORK',
    'HEALTH_AND_FITNESS': 'HEALTH',
    'MEDICAL': 'HEALTH',
    'SYSTEM': 'SYSTEM',
    'MISC': 'SYSTEM', # ABC logger
     None: 'UNKNOWN',
    'UNKNOWN':'UNKNOWN'
}


param = {
    "predictor": 'cpu_predictor',
    "early_stopping_rounds": 200,
    "reg_alpha": 0,
    "colsample_bytree": 1,
    "colsample_bylevel": 1,
    "scale_pos_weight": 1,
    "learning_rate": 0.3,
    "min_child_weight": 1,
    "subsample": 1,
    "reg_lambda": 1.72,
    "reg_alpha":0,
    "seed": seed,
    "objective": 'binary:logistic',
    "max_depth": 6,
    "gamma": 0,
    'eval_metric': 'auc',
    'verbosity': 0,
    'tree_method': 'hist',
}

def select_top_xgb_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_k: int = 10,
    random_state=42
) -> List[str]:
    """Select top features using SHAP values from an XGBoost classifier.

    Args:
        X_train (pd.DataFrame): Training features dataframe.
        y_train (pd.Series): Training target labels series.
        X_test (pd.DataFrame): Test features dataframe.
        y_test (pd.Series): Test target labels series.
        top_k (int): Number of top features to select.
        random_state: Random seed for reproducibility.

    Returns:
        List[str]: List of selected feature names.

    Raises:
        ValueError: If y_train or y_test contains fewer than two classes.
    """
    if len(np.unique(y_train)) < 2:
        raise ValueError("y_train must contain at least two classes")
    if len(np.unique(y_test)) < 2:
        raise ValueError("y_test must contain at least two classes")

    model = XGBClassifier(
        n_estimators=500, max_depth=4, random_state=random_state,
        eval_metric='auc', early_stopping_rounds=100, n_jobs= -1,
        tree_method='hist', gpu_id=-1  # Force CPU usage
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list): 
        shap_values = np.array(shap_values).mean(axis=0)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_k:]

    return X_train.columns[top_indices].tolist()

def save_correlation_heatmaps(
    corr_maps: Dict[Any, pd.DataFrame],
    common_feats: List[str],
    feature_imps: Dict[Any, np.ndarray],
    shap_imps: Dict[Any, np.ndarray],
    out_dir: str = 'correlation_heatmaps'
):
    """Generate and save heatmap PNG files for each user based on correlation matrices.

    Args:
        corr_maps (Dict[Any, pd.DataFrame]): Mapping from user IDs to their feature correlation matrices.
        common_feats (List[str]): List of common feature names.
        feature_imps (Dict[Any, np.ndarray]): Mapping from user IDs to their normalized XGBoost feature importances.
        shap_imps (Dict[Any, np.ndarray]): Mapping from user IDs to their normalized SHAP importances.
        out_dir (str, optional): Directory to save the heatmap PNG files. Defaults to 'correlation_heatmaps'.
    """
    os.makedirs(out_dir, exist_ok=True)
    for user, corr in corr_maps.items():
        labels = [
            f"{feat}\n({g:.2f}/{s:.2f})"
            for feat, g, s in zip(common_feats,
                                   feature_imps[user],
                                   shap_imps[user])
        ]
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr, cmap='coolwarm', center=0,
            annot=True, fmt='.2f',
            xticklabels=labels, yticklabels=labels
        )
        plt.title(f'Feature Corr Heatmap (User {user})')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/Correlation_{user}.png")
        plt.close()
        
def _wass_pair(i, j, data, groups, feats):
    """Compute Wasserstein distance between two users' feature distributions.
    
    Args:
        i: First user ID.
        j: Second user ID.
        data: Feature dataframe.
        groups: User group identifiers.
        feats: List of features to compare.
        
    Returns:
        float: Mean Wasserstein distance across features.
    """
    xi = data.loc[groups == i, feats]
    xj = data.loc[groups == j, feats]
    dists = [wasserstein_distance(xi[col], xj[col]) for col in feats]
    return np.mean(dists)

def compute_pairwise_wasserstein_matrix(data, groups, feats, n_jobs=1, cache_path=None):
    """Compute pairwise Wasserstein distance matrix between users.
    
    Args:
        data: Feature dataframe.
        groups: User group identifiers.
        feats: List of features to compare.
        n_jobs (int): Number of parallel jobs.
        cache_path: Path to cache the matrix.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: User IDs and distance matrix.
    """
    user_ids = np.unique(groups)
    n = len(user_ids)
    mat = np.zeros((n, n))
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(_wass_pair)(ui, uj, data, groups, feats)
        for idx, ui in enumerate(user_ids)
        for uj in user_ids[idx+1:]
    )
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            mat[i, j] = mat[j, i] = results[k]
            k += 1
    if cache_path:
        np.save(cache_path, mat)
    return user_ids, mat

def compute_pairwise_label_shift_matrix(labels, groups, n_jobs=1, cache_path=None):
    """Compute pairwise label distribution shift matrix between users.
    
    Args:
        labels: Target labels array.
        groups: User group identifiers.
        n_jobs (int): Number of parallel jobs.
        cache_path: Path to cache the matrix.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: User IDs and label shift matrix.
    """
    user_ids = np.unique(groups)
    max_label = labels.max()
    probs = {}
    for u in user_ids:
        y_u = labels[groups == u]
        counts = np.bincount(y_u, minlength=max_label+1)
        probs[u] = counts / counts.sum()
    n = len(user_ids)
    mat = np.zeros((n, n))
    for i, ui in enumerate(user_ids):
        for j in range(i+1, n):
            uj = user_ids[j]
            mat[i, j] = mat[j, i] = np.linalg.norm(probs[ui] - probs[uj])
    if cache_path:
        np.save(cache_path, mat)
    return user_ids, mat

def calculate_weighted_otdd_distance(
    features_u: np.ndarray,
    labels_u: np.ndarray,
    features_v: np.ndarray,
    labels_v: np.ndarray,
    importances_u: np.ndarray,
    importances_v: np.ndarray,
    ot_params: Dict[str, Any]
) -> float:
    """
    Calculate the OTDD distance between two user datasets with weighted features.

    Args:
        features_u: Feature matrix for user U.
        labels_u: Label array for user U.
        features_v: Feature matrix for user V.
        labels_v: Label array for user V.
        importances_u: Feature importances for user U.
        importances_v: Feature importances for user V.
        ot_params: Parameters for the DatasetDistance calculation.

    Returns:
        OTDD distance as a float.
    """
    weights = importances_u
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(weights) / (len(weights) * 4)
    scale = np.sqrt(weights)[None, :]
    Xu = (features_u * scale).astype(np.float32)
    Xv = (features_v * scale).astype(np.float32)
    Xu += np.random.normal(0, 1e-3, Xu.shape)
    Xv += np.random.normal(0, 1e-3, Xv.shape)
    yu = labels_u.astype(np.int64)
    yv = labels_v.astype(np.int64)
    ds_u = TensorDataset(torch.from_numpy(Xu), torch.from_numpy(yu))
    ds_v = TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv))
    loader_u = DataLoader(ds_u, batch_size=len(Xu), shuffle=False)
    loader_v = DataLoader(ds_v, batch_size=len(Xv), shuffle=False)
    ot = DatasetDistance(loader_u, loader_v, **ot_params)
    return ot.distance().item()

def _compute_otdd_for_pair(
    i: int,
    j: int,
    user_ids: np.ndarray,
    data: Dict[Any, Tuple[np.ndarray, np.ndarray]],
    feature_list: List[str],
    user_importances: Dict[Any, np.ndarray],
    ot_params: Dict[str, Any]
) -> Tuple[int, int, float]:
    """
    Compute OTDD distance for a single pair of users.

    Args:
        i: Index of first user in user_ids.
        j: Index of second user in user_ids.
        user_ids: Array of unique user identifiers.
        data: Mapping from user ID to (features, labels).
        feature_list: List of features to include.
        user_importances: Mapping from user ID to feature importances array.
        ot_params: Parameters for the DatasetDistance calculation.

    Returns:
        Tuple of (i, j, distance).
    """
    u_id = user_ids[i]
    v_id = user_ids[j]
    Xu, yu = data[u_id]
    Xv, yv = data[v_id]
    imp_u = user_importances.get(u_id, np.ones(len(feature_list)) / len(feature_list))
    imp_v = user_importances.get(v_id, np.ones(len(feature_list)) / len(feature_list))
    dist = calculate_weighted_otdd_distance(
        Xu, yu, Xv, yv, imp_u, imp_v, ot_params
    )
    return i, j, dist

def compute_pairwise_conditional_matrix(
    features: pd.DataFrame,
    labels: np.ndarray,
    group_indices: np.ndarray,
    feature_list: List[str],
    user_importances: Dict[Any, np.ndarray],
    device: str = "cpu",
    n_jobs: int = -1,
    cache_path: str = "otdd_w.npy"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise OTDD distances between users and return user IDs with the distance matrix.
    Results are cached to disk to avoid recomputation.

    Args:
        features: Dataframe of all user features.
        labels: Array of all labels corresponding to features.
        group_indices: Array of user identifiers for each row in features.
        feature_list: List of features to include.
        user_importances: Mapping from user ID to feature importances array.
        device: PyTorch device for OTDD computation.
        n_jobs: Number of parallel jobs for distance computation.
        cache_path: File path for caching the distance matrix.

    Returns:
        Tuple containing array of unique user IDs and the distance matrix.
    """
    user_ids = np.unique(group_indices)
    if os.path.exists(cache_path):
        return user_ids, np.load(cache_path)

    data = {
        u: (
            features.loc[group_indices == u, feature_list].values,
            labels[group_indices == u]
        )
        for u in user_ids
    }

    ot_params = {
        "inner_ot_method": "exact",
        "debiased_loss": True,
        "p": 2,
        "λ_x": 0,
        "λ_y": 1,
        "entreg": 1e-5,
        "device": device
    }

    pairs = [(i, j) for i in range(len(user_ids)) for j in range(i + 1, len(user_ids))]
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(_compute_otdd_for_pair)(
            i, j, user_ids, data, feature_list, user_importances, ot_params
        )
        for i, j in pairs
    )

    U = len(user_ids)
    D = np.zeros((U, U), dtype=float)
    for i, j, dist in results:
        D[i, j] = D[j, i] = dist

    if np.isnan(D).any():
        max_val = np.nanmax(D)
        D[np.isnan(D)] = max_val * 10

    np.save(cache_path, D)
    return user_ids, D

def plot_distance_matrix_and_save(mat, out_dir, title, fname_prefix, users=None):
    """Generate and save distance matrix visualizations.
    
    Args:
        mat: Distance matrix.
        out_dir (str): Output directory.
        title (str): Plot title.
        fname_prefix (str): Filename prefix.
        users: Optional user list for filtering.
    """
    flat = mat[np.triu_indices_from(mat, k=1)]
    plt.figure(figsize=(6, 4))
    sns.histplot(flat, bins=50, kde=True)
    plt.title(f"{title} Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{fname_prefix}_hist.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, square=True, cbar_kws={'label':'Distance'}, cmap="viridis")
    plt.title(f"{title} Matrix")
    plt.xlabel("User Index")
    plt.ylabel("User Index")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{fname_prefix}_matrix.png")
    plt.close()

def _train_concept_shift_model(X_src, y_src, X_tgt, y_tgt, random_state=42):
    """Train a model to learn P(Y|X) for source and target distributions.
    
    Args:
        X_src: Source features.
        y_src: Source labels.
        X_tgt: Target features.
        y_tgt: Target labels.
        random_state: Random seed.
    
    Returns:
        Tuple of predicted probability distributions for both domains.
    """
    X_combined = np.vstack([X_src, X_tgt])
    y_combined = np.hstack([y_src, y_tgt])
    
    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        eval_metric='auc',
        tree_method='hist',
        n_jobs = -1,
        random_state=random_state,
        gpu_id=-1  # Force CPU usage
    )
    model.fit(X_combined, y_combined)
    
    prob_src = model.predict_proba(X_src)
    prob_tgt = model.predict_proba(X_tgt)
    
    return prob_src, prob_tgt

def _compute_concept_shift_pair(i, j, user_ids, data, feats, random_state=42):
    """Compute concept shift between two users using Jensen-Shannon divergence.
    
    Args:
        i: Index of first user.
        j: Index of second user.
        user_ids: Array of user identifiers.
        data: Feature dataframe.
        feats: List of features to use.
        random_state: Random seed.
        
    Returns:
        float: Jensen-Shannon divergence between users.
    """
    ui, uj = user_ids[i], user_ids[j]
    
    try:
        Xi = data.loc[data.index.get_level_values(0) == ui, feats].values
        yi = data.loc[data.index.get_level_values(0) == ui, 'label'].values
        Xj = data.loc[data.index.get_level_values(0) == uj, feats].values
        yj = data.loc[data.index.get_level_values(0) == uj, 'label'].values
        
        if len(Xi) < 10 or len(Xj) < 10 or len(np.unique(yi)) < 2 or len(np.unique(yj)) < 2:
            return 1.0
        
        prob_i, prob_j = _train_concept_shift_model(Xi, yi, Xj, yj, random_state)
        
        avg_prob_i = np.mean(prob_i, axis=0)
        avg_prob_j = np.mean(prob_j, axis=0)
        
        eps = 1e-8
        avg_prob_i = np.clip(avg_prob_i, eps, 1-eps)
        avg_prob_j = np.clip(avg_prob_j, eps, 1-eps)
        
        avg_prob_i = avg_prob_i / np.sum(avg_prob_i)
        avg_prob_j = avg_prob_j / np.sum(avg_prob_j)
        
        js_dist = jensenshannon(avg_prob_i, avg_prob_j)
        
        if np.isnan(js_dist) or np.isinf(js_dist):
            return 1.0
            
        return js_dist
        
    except Exception as e:
        print(f"Error computing concept shift for users {ui}-{uj}: {e}")
        return 1.0

def compute_pairwise_concept_shift_matrix(data, groups, labels, feats, n_jobs=1, cache_path=None, random_state=42):
    """Compute pairwise concept shift matrix using Jensen-Shannon divergence on P(Y|X).
    
    Args:
        data: DataFrame with features.
        groups: Array of user identifiers.
        labels: Array of labels.
        feats: List of feature names to use.
        n_jobs (int): Number of parallel jobs.
        cache_path: Path to cache results.
        random_state: Random seed.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: User IDs and concept shift matrix.
    """
    if cache_path and os.path.exists(cache_path):
        user_ids = np.unique(groups)
        return user_ids, np.load(cache_path)
    
    user_ids = np.unique(groups)
    n = len(user_ids)
    
    combined_data = data[feats].copy()
    combined_data['label'] = labels
    combined_data.index = pd.MultiIndex.from_arrays([groups, combined_data.index], names=['user', 'sample'])
    
    results = Parallel(n_jobs=-1 ,backend='loky')(
        delayed(_compute_concept_shift_pair)(i, j, user_ids, combined_data, feats, random_state)
        for i in range(n)
        for j in range(i+1, n)
    )
    
    mat = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            mat[i, j] = mat[j, i] = results[k]
            k += 1
    
    if cache_path:
        np.save(cache_path, mat)
    
    return user_ids, mat

def create_train_test_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Create random 80/20 train/test splits for each user.
    
    Uses random shuffled splitting where 80% of each user's data is randomly selected
    for training and the remaining 20% for testing, with stratification to preserve class balance.
    
    Args:
        X (pd.DataFrame): Full feature dataframe.
        y (np.ndarray): Full target labels.
        groups (np.ndarray): User group identifiers.
        test_size (float): Fraction for test split.
        random_state (int): Random seed for reproducible shuffled splits.
        
    Returns:
        Tuple: (X_train_all, y_train_all, groups_train_all, X_test_all, y_test_all, groups_test_all)
    """
    user_ids = np.unique(groups)
    
    X_train_parts = []
    y_train_parts = []
    groups_train_parts = []
    X_test_parts = []
    y_test_parts = []
    groups_test_parts = []
    
    for user in user_ids:
        user_mask = (groups == user)
        X_user = X.loc[user_mask]
        y_user = y[user_mask]
        
        if len(np.unique(y_user)) < 2:
            print(f"Skipping user {user}: insufficient class diversity")
            continue
            
        try:
            # Random shuffle split: 80/20 with shuffle=True
            X_train, X_test, y_train, y_test = train_test_split(
                X_user, y_user,
                test_size=test_size,
                random_state=random_state,
                stratify=y_user,
                shuffle=True
            )
            
            # Check if both splits have at least one sample of each class
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print(f"Skipping user {user}: shuffle split results in insufficient class diversity")
                continue
            
            X_train_parts.append(X_train)
            y_train_parts.append(y_train)
            groups_train_parts.append(np.full(len(y_train), user))
            
            X_test_parts.append(X_test)
            y_test_parts.append(y_test)
            groups_test_parts.append(np.full(len(y_test), user))
            
        except Exception as e:
            print(f"Error splitting user {user}: {e}")
            continue
    
    if not X_train_parts:
        raise ValueError("No valid users found for train/test split")
    
    X_train_all = pd.concat(X_train_parts, ignore_index=True)
    y_train_all = np.concatenate(y_train_parts)
    groups_train_all = np.concatenate(groups_train_parts)
    
    X_test_all = pd.concat(X_test_parts, ignore_index=True)
    y_test_all = np.concatenate(y_test_parts)
    groups_test_all = np.concatenate(groups_test_parts)
    
    print(f"Training data: {len(X_train_all)} samples from {len(np.unique(groups_train_all))} users")
    print(f"Test data: {len(X_test_all)} samples from {len(np.unique(groups_test_all))} users")
    
    return X_train_all, y_train_all, groups_train_all, X_test_all, y_test_all, groups_test_all

def normalize_data(
    X_train: pd.DataFrame,
    groups_train: np.ndarray,
    X_test: pd.DataFrame,
    groups_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize features using user-wise normalization.
    
    Args:
        X_train (pd.DataFrame): Training data (80% from each user).
        groups_train (np.ndarray): User groups for training data.
        X_test (pd.DataFrame): Test data (20% from each user).
        groups_test (np.ndarray): User groups for test data.
        numeric_cols (List[str]): List of numeric columns to normalize.
        categorical_cols (pd.DataFrame): Categorical columns dataframe.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Normalized training and test data.
    """
    X_numeric_train = X_train[numeric_cols]
    train_vars = X_numeric_train.var()
    valid_features = train_vars[train_vars > 0].index.tolist()
    
    user_ids = np.unique(groups_train)
    train_normalized_parts = []
    test_normalized_parts = []
    
    for user in user_ids:
        user_train_mask = (groups_train == user)
        X_user_train = X_train.loc[user_train_mask, valid_features]
        
        if len(X_user_train) == 0:
            continue
            
        scaler = StandardScaler()
        X_user_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_user_train),
            columns=X_user_train.columns,
            index=X_user_train.index
        )
        train_normalized_parts.append(X_user_train_scaled)
        
        user_test_mask = (groups_test == user)
        X_user_test = X_test.loc[user_test_mask]
        
        if len(X_user_test) == 0:
            continue
            
        common_features = [f for f in valid_features if f in X_user_test.columns]
        
        if len(common_features) == 0:
            continue
            
        X_user_test_common = X_user_test[common_features].copy()
        
        common_numeric_cols = [col for col in common_features if col in valid_features]
        
        if len(common_numeric_cols) > 0:
            user_train_numeric = X_user_train_scaled[common_numeric_cols]
            user_test_numeric = X_user_test_common[common_numeric_cols]
            
            X_user_test_common[common_numeric_cols] = scaler.transform(user_test_numeric)
        
        test_normalized_parts.append(X_user_test_common)
    
    X_train_normalized = pd.concat(train_normalized_parts)
    X_test_normalized = pd.concat(test_normalized_parts) if test_normalized_parts else pd.DataFrame()
    
    if not categorical_cols.empty:
        cat_train_mask = categorical_cols.index.isin(X_train.index)
        cat_filtered = categorical_cols.loc[cat_train_mask]
        
        X_train_final = pd.concat([
            X_train_normalized.reset_index(drop=True),
            cat_filtered.reset_index(drop=True)
        ], axis=1)
        
        if not X_test_normalized.empty:
            cat_test_mask = categorical_cols.index.isin(X_test.index)
            cat_test_filtered = categorical_cols.loc[cat_test_mask]
            
            X_test_final = pd.concat([
                X_test_normalized.reset_index(drop=True),
                cat_test_filtered.reset_index(drop=True)
            ], axis=1)
        else:
            X_test_final = pd.DataFrame()
    else:
        X_train_final = X_train_normalized.reset_index(drop=True)
        X_test_final = X_test_normalized.reset_index(drop=True) if not X_test_normalized.empty else pd.DataFrame()
    
    problematic_chars = ['[', ']', '<', '>', '{', '}', '(', ')', ',']
    
    cols_to_drop_train = []
    for col in X_train_final.columns:
        if any(char in str(col) for char in problematic_chars):
            cols_to_drop_train.append(col)
    
    if cols_to_drop_train:
        print(f"Dropping {len(cols_to_drop_train)} columns with problematic characters from training data: {cols_to_drop_train[:5]}...")
        X_train_final = X_train_final.drop(columns=cols_to_drop_train)
    
    if not X_test_final.empty:
        cols_to_drop_test = []
        for col in X_test_final.columns:
            if any(char in str(col) for char in problematic_chars):
                cols_to_drop_test.append(col)
        
        if cols_to_drop_test:
            print(f"Dropping {len(cols_to_drop_test)} columns with problematic characters from test data: {cols_to_drop_test[:5]}...")
            X_test_final = X_test_final.drop(columns=cols_to_drop_test)
    
    # print(f"Normalized features ({len(valid_features)}): {valid_features}")
    print(f"Normalized features ({len(valid_features)})")
    print(f"Final normalized training shape: {X_train_final.shape}")
    print(f"Final normalized test shape: {X_test_final.shape}")
    
    return X_train_final, X_test_final

def filter_users(
    X_train_normalized: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test_normalized: pd.DataFrame,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    pra_threshold: float = 0.65,
    random_state: int = 42
) -> Tuple[List, pd.DataFrame, np.ndarray, np.ndarray]:
    """Filter users based on performance using proper train/test splits.
    
    Args:
        X_train_normalized (pd.DataFrame): Normalized training data (80%).
        y_train (np.ndarray): Training labels (80%).
        groups_train (np.ndarray): Training groups (80%).
        X_test_normalized (pd.DataFrame): Normalized test data (20%).
        y_test (np.ndarray): Test labels (20%).
        groups_test (np.ndarray): Test groups (20%).
        pra_threshold (float): PRAUC threshold for filtering.
        random_state (int): Random seed.
        
    Returns:
        Tuple: (valid_users, filtered_X_train, filtered_y_train, filtered_groups_train)
    """
    def check_user(user):
        train_idx = (groups_train == user)
        X_u_train = X_train_normalized.loc[train_idx]
        y_u_train = y_train[train_idx]
        
        test_idx = (groups_test == user)
        X_u_test = X_test_normalized.loc[test_idx]
        y_u_test = y_test[test_idx]
        
        if (len(np.unique(y_u_train)) < 2 or len(X_u_train) < 10 or 
            len(np.unique(y_u_test)) < 2 or len(X_u_test) < 5):
            return None
            
        model = XGBClassifier(
            n_estimators=500,
            max_depth=3,
            eval_metric='auc',
            tree_method='hist',
            early_stopping_rounds=100,
            random_state=random_state,
            gpu_id=-1  # Force CPU usage
        )
        
        try:
            model.fit(X_u_train, y_u_train, eval_set=[(X_u_test, y_u_test)], verbose=False)
            
            proba = model.predict_proba(X_u_test)[:, 1]
            prauc = average_precision_score(y_u_test, proba)
            
            if prauc >= pra_threshold:
                return user
            else:
                print(f"User {user} removed (PRAUC = {prauc:.3f})")
                return None
        except Exception as e:
            print(f"User {user} failed: {e}")
            return None

    common_users = np.intersect1d(groups_train, groups_test)
    results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
        delayed(check_user)(u) for u in common_users
    )
    valid_users = [u for u in results if u is not None]

    mask = np.isin(groups_train, valid_users)
    X_filtered = X_train_normalized.loc[mask].reset_index(drop=True)
    y_filtered = y_train[mask]
    groups_filtered = groups_train[mask]

    print(f"Remaining users after filtering: {len(valid_users)}, samples: {len(y_filtered)}")
    return valid_users, X_filtered, y_filtered, groups_filtered

def evaluate_users_with_fixed_splits(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    common_feats: List[str],
    numeric_cols: List[str],
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """Evaluate users using pre-split train/test data.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (np.ndarray): Training labels.
        groups_train (np.ndarray): Training user groups.
        X_test (pd.DataFrame): Test features.
        y_test (np.ndarray): Test labels.
        groups_test (np.ndarray): Test user groups.
        common_feats (List[str]): List of common features to use.
        numeric_cols (List[str]): List of numeric columns for normalization.
        random_state (int): Random seed.
        n_jobs (int): Number of parallel jobs.
        
    Returns:
        Tuple: (common_scores, full_scores, user_imp, user_imp_map, corr_maps)
    """
    from sklearn.preprocessing import StandardScaler
    
    def _process_user(user):
        train_mask = (groups_train == user)
        X_train_user = X_train.loc[train_mask]
        y_train_user = y_train[train_mask]
        
        test_mask = (groups_test == user)
        X_test_user = X_test.loc[test_mask]
        y_test_user = y_test[test_mask]
        
        if len(np.unique(y_train_user)) < 2 or len(np.unique(y_test_user)) < 2:
            return None
        if len(X_train_user) < 5 or len(X_test_user) < 2:
            return None
            
        common_cols = X_train_user.columns.intersection(X_test_user.columns)
        
        if len(common_cols) == 0:
            return None
            
        X_train_final = X_train_user[common_cols]
        X_test_common = X_test_user[common_cols]
        
        if len(X_test_common) == 0 or X_test_common.shape[1] == 0:
            return None
            
        from sklearn.preprocessing import StandardScaler
        
        try:
            numeric_test_cols = X_test_common.select_dtypes(include=[np.number]).columns
            
            if len(numeric_test_cols) > 0:
                test_scaler = StandardScaler()
                X_test_numeric = X_test_common[numeric_test_cols]
                
                if X_test_numeric.var().sum() > 0:
                    X_test_scaled = pd.DataFrame(
                        test_scaler.fit_transform(X_test_numeric),
                        columns=numeric_test_cols,
                        index=X_test_numeric.index
                    )
                    
                    non_numeric_cols = X_test_common.columns.difference(numeric_test_cols)
                    if len(non_numeric_cols) > 0:
                        X_test_final = pd.concat([
                            X_test_scaled,
                            X_test_common[non_numeric_cols]
                        ], axis=1)
                    else:
                        X_test_final = X_test_scaled
                else:
                    X_test_final = X_test_common
            else:
                X_test_final = X_test_common
                
        except Exception as e:
            X_test_final = X_test_common
        
        problematic_chars = ['[', ']', '<', '>', '{', '}', '(', ')', ',']
        cols_to_drop = []
        for col in X_train_final.columns:
            if any(char in str(col) for char in problematic_chars):
                cols_to_drop.append(col)
        
        if cols_to_drop:
            X_train_final = X_train_final.drop(columns=cols_to_drop)
            X_test_final = X_test_final.drop(columns=cols_to_drop)
        
        available_common_feats = [f for f in common_feats if f in X_train_final.columns]
        if not available_common_feats:
            return None
            
        X_train_common = X_train_final[available_common_feats]
        X_test_common = X_test_final[available_common_feats]
        
        def fit_and_predict(X_tr, y_tr, X_te, y_te):
            # Ensure consistent feature ordering
            X_te = X_te[X_tr.columns]
            
            model = XGBClassifier(
                n_estimators=500,
                max_depth=3,
                eval_metric='auc',
                tree_method='hist',
                early_stopping_rounds=100,
                random_state=random_state,
                gpu_id=-1  # Force CPU usage
            )
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            proba = model.predict_proba(X_te)[:, 1]
            return model, proba

        m_c, pred_c = fit_and_predict(X_train_common, y_train_user, X_test_common, y_test_user)
        sc_c = {
            'auroc': roc_auc_score(y_test_user, pred_c),
            'prauc': average_precision_score(y_test_user, pred_c)
        }

        m_f, pred_f = fit_and_predict(X_train_final, y_train_user, X_test_final, y_test_user)
        sc_f = {
            'auroc': roc_auc_score(y_test_user, pred_f),
            'prauc': average_precision_score(y_test_user, pred_f)
        }

        imp = m_c.feature_importances_
        explainer = shap.TreeExplainer(m_c)
        shap_vals = explainer.shap_values(X_train_common)
        shap_imp = np.abs(shap_vals).mean(axis=0)

        corr = X_train_common.corr()

        norm_imp = imp / imp.sum() if imp.sum() > 0 else np.ones_like(imp) / len(imp)
        norm_shap = shap_imp / shap_imp.sum() if shap_imp.sum() > 0 else np.ones_like(shap_imp) / len(shap_imp)

        return user, sc_c, sc_f, norm_imp, norm_shap, corr

    train_users = set(np.unique(groups_train))
    test_users = set(np.unique(groups_test))
    common_users = list(train_users.intersection(test_users))
    
    print(f"Evaluating {len(common_users)} users with consistent train/test splits")
    
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(_process_user)(u) for u in common_users
    )

    common_scores = {}
    full_scores = {}
    user_imp = {}
    user_imp_map = {}
    corr_maps = {}

    for res in results:
        if res is None:
            continue
        user, sc_c, sc_f, imp_arr, shap_arr, corr = res
        common_scores[user] = sc_c
        full_scores[user] = sc_f
        user_imp[user] = imp_arr
        user_imp_map[user] = shap_arr
        corr_maps[user] = corr

    return common_scores, full_scores, user_imp, user_imp_map, corr_maps

def fdr_bh_series(p):
    """Benjamini–Hochberg q-values (Series 입력)."""
    p = pd.Series(p, dtype=float)
    mask = p.notna()
    m = int(mask.sum())
    q = pd.Series(np.nan, index=p.index, dtype=float)
    if m == 0:
        return q
    order = p[mask].sort_values().index
    ranks = np.arange(1, m+1)
    adj = p[mask].loc[order].values * m / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q.loc[order] = np.minimum(adj, 1.0)
    return q

def wilcoxon_greater(x):
    x = np.asarray(list(x), dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 10 or np.allclose(x, 0):
        return np.nan
    try:
        return wilcoxon(x, alternative='greater').pvalue
    except ValueError:
        return np.nan

def sign_test_greater(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan
    k_pos = int((x > 0).sum())
    return binomtest(k_pos, n, p=0.5, alternative='greater').pvalue