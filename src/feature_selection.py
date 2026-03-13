from pyspark.sql.types import NumericType
from pyspark.sql.functions import corr
import pandas as pd
import matplotlib.pyplot as plt

#Keep only numeric values
def get_numeric_features(df):
    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType) and f.name != "yield"]
    return numeric_cols


# Correlation analysis
def compute_feature_correlations(df, numeric_cols):
    correlations = []
    for c in numeric_cols:
        corr_val = df.select(corr("yield", c)).collect()[0][0]
        if corr_val is not None:
            correlations.append((c, corr_val))

    corr_df = pd.DataFrame(correlations, columns=["feature", "correlation"])
    corr_df["abs_corr"] = corr_df["correlation"].abs()
    return corr_df


#top features
def get_top_features(corr_df, top_n=15):
    top_features = corr_df.sort_values("abs_corr", ascending=False).head(top_n)
    return top_features


#correlation barplot
def plot_top_features(top_features):
    top_sorted = top_features.sort_values("abs_corr", ascending=True)

    plt.figure(figsize=(8,6))
    plt.barh(top_sorted["feature"], top_sorted["abs_corr"],
             color="skyblue", edgecolor="black")

    plt.xlabel("Absolute Correlation with Yield", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title("Top Features Most Correlated with Yield", fontsize=14, weight="bold")

    for i, v in enumerate(top_sorted["abs_corr"]):
        plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=10)

    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


#adding yield
def select_top_features(df, feature_list):
    df_selected = df.select(feature_list + ["yield"])
    return df_selected
