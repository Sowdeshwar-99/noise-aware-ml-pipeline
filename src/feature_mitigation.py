import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time


#FEATURE Noise Mitigation (Bayesian Ridge + PCA Denoising)
def mitig_feat(
    in_path,
    noise_tag="20"
):

    print(f"\n Running FEATURE mitigation for {noise_tag}% noise...")

    t0 = time.time()

    print(" Loading dataset...")
    df_noisy = pd.read_parquet(in_path)

    num_cols = [
        c for c in df_noisy.columns
        if c not in ["yield", "DOY", "zadok_stage"]
    ]

    print(f" Loaded {len(num_cols)} numeric feature columns for mitigation.")
    print(f" Total rows: {len(df_noisy):,}")

    #BayesianRidge Iterative Imputation
    print("\n Step 1/3: Performing multivariate imputation (Bayesian Ridge)...")

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=10,
        random_state=42,
        tol=1e-3,
        initial_strategy='median'
    )

    X_imputed = imputer.fit_transform(df_noisy[num_cols])

    print(" Imputation completed successfully.")

    # PCA Denoising
    print("\n Step 2/3: PCA reconstruction for denoising (k=10)...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_imputed)

    k = min(10, len(num_cols))

    pca = PCA(n_components=k, random_state=42)

    X_pca = pca.fit_transform(X_scaled)

    X_reconstructed = pca.inverse_transform(X_pca)

    X_denoised = scaler.inverse_transform(X_reconstructed)

    print(" PCA denoising complete.")

    # Combining and saving
    print("\n  Combining mitigated dataset...")

    df_mitigated = pd.DataFrame(X_denoised, columns=num_cols)

    df_mitigated["yield"] = df_noisy["yield"].values

    for col in ["DOY", "zadok_stage"]:
        if col in df_noisy.columns:
            df_mitigated[col] = df_noisy[col].values

    print(f"Columns saved: {len(df_mitigated.columns)}")

    print(f" Total runtime: {time.time() - t0:.2f} sec")

    return df_mitigated
