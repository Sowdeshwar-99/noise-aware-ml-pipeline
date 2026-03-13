from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
import numpy as np
import time


#Label Noise Mitigation (MAD based residual analysis)
def mitig_labl(
    spark,
    df_feat,
    noise_tag="20"
):

    print(f"\nRunning LABEL mitigation for {noise_tag}% noise...")

    t0 = time.time()

    TARGET = "yield"

    if "features_pca" in df_feat.columns:

        df_vec = df_feat.select(
            F.col("features_pca").alias("features"),
            TARGET
        )

    else:

        feat_cols = [
            c for (c, t) in df_feat.dtypes
            if t in ["double", "float", "int", "bigint"] and c != TARGET
        ]

        assembler = VectorAssembler(
            inputCols=feat_cols,
            outputCol="features",
            handleInvalid="keep"
        )

        df_vec = assembler.transform(df_feat).select("features", TARGET)

    total_rows = df_vec.count()

    print(f" Total rows for yield correction: {total_rows:,}")

    #Training random forest on a small sample
    print("\nTraining Random Forest model (3% sample)...")

    df_train = df_vec.sample(False, 0.03, 42)

    train_rows = df_train.count()

    print(f"Training sample size: {train_rows:,}")

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol=TARGET,
        numTrees=80,
        maxDepth=8,
        maxBins=32,
        subsamplingRate=0.7,
        featureSubsetStrategy="sqrt",
        seed=42
    )

    rf_model = rf.fit(df_train)

    print("Random Forest trained successfully.")

    #predicting yields and computing residuals
    print("\n Predicting yields for all rows...")

    df_pred = rf_model.transform(df_vec).withColumnRenamed(
        "prediction",
        "y_pred"
    )

    df_pred = df_pred.withColumn(
        "resid",
        F.col(TARGET) - F.col("y_pred")
    )

    # compute MAD threshold on 1% of the sample
    resids = np.array(
        df_pred.select("resid")
        .sample(False, 0.01, 42)
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    mad = np.median(np.abs(resids - np.median(resids))) + 1e-12

    threshold = 2.5 * mad

    print(f" Residual MAD threshold: {threshold:.4f}")

    #Flag and correct noisy yields
    df_flagged = df_pred.withColumn(
        "is_noisy",
        F.abs(F.col("resid")) > threshold
    )

    noisy_cnt = df_flagged.filter("is_noisy").count()

    print(f"Flagged {noisy_cnt:,} noisy rows")

    df_flagged = df_flagged.withColumn(
        "yield_mitigated",
        F.when(F.col("is_noisy"), F.col("y_pred"))
        .otherwise(F.col(TARGET))
    )

    df_flagged = df_flagged.withColumn(
        "yield_mitigated",
        F.greatest(F.col("yield_mitigated"), F.lit(0.0))
    )

    return df_flagged
