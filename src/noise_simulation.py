from pyspark.sql import functions as F
from pyspark.sql.functions import col, rand, randn, when, lit, exp, log, greatest
from pyspark.sql.types import NumericType


# Row-level noise simulation
def sim_noise(
    df_input,
    noise_tag="20",
    total_noise_fraction=0.20,
    label_ratio=0.5,
    mcar_level=0.10,
    gaussian_level_signed=0.25,
    relative_level_nonneg=0.20,
    logit_level_b01=0.10,
    mul_sigma_clean=0.10,
    mul_sigma_corrupt=0.50,
    outlier_scale=3.0,
    frac_zero=0.4,
    frac_outlier=0.3,
    seed=42
):

    df = df_input

    num_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, NumericType)
        and f.name not in ["yield", "DOY", "zadok_stage"]
    ]

    #Get min/max for feature-type classification
    summary = df.select(
        *[F.min(c).alias(f"{c}_min") for c in num_cols],
        *[F.max(c).alias(f"{c}_max") for c in num_cols]
    ).collect()[0].asDict()

    bounded01, nonneg, signed = [], [], []

    for c in num_cols:
        lo, hi = summary[f"{c}_min"], summary[f"{c}_max"]

        if lo >= 0 and hi <= 1.05:
            bounded01.append(c)
        elif lo >= 0:
            nonneg.append(c)
        else:
            signed.append(c)

    # Flagging required amount of rows
    is_noisy = rand(seed) < total_noise_fraction
    u = rand(seed + 1)

    is_label_noise = is_noisy & (u < label_ratio)
    is_feature_noise = is_noisy & (u >= label_ratio)

    # Applying label noise to half of the flagged rows
    def lognormal_mult(seed, sigma):
        return exp(randn(seed=seed) * lit(sigma))

    base_mult = lognormal_mult(seed + 2, mul_sigma_clean)
    heavy_mult = lognormal_mult(seed + 3, mul_sigma_corrupt)

    y_base = col("yield") * base_mult

    label_noise_yield = (
        when(is_label_noise & (rand(seed + 4) < frac_zero), 0.0)
        .when(is_label_noise & (rand(seed + 5) < frac_outlier), y_base * outlier_scale)
        .when(is_label_noise, col("yield") * heavy_mult)
        .otherwise(y_base)
    )

    label_noise_yield = greatest(label_noise_yield, lit(0.0))

    df = df.withColumn("yield", label_noise_yield)

    # applying feature noise to half of the flagged rows
    for c in signed:
        std_val = df.select(F.stddev(col(c))).collect()[0][0]

        if std_val and std_val > 0:
            noise = randn(seed=seed + 6) * (gaussian_level_signed * std_val)

            df = df.withColumn(
                c,
                when(is_feature_noise, col(c) + noise).otherwise(col(c))
            )

    for c in nonneg:

        jitter = F.abs(
            col(c) * (1 + randn(seed=seed + 7) * relative_level_nonneg)
        )

        df = df.withColumn(
            c,
            when(is_feature_noise, jitter).otherwise(col(c))
        )

    def logit(x):
        return log((x + 1e-6) / (1 - x + 1e-6))

    def inv_logit(z):
        return 1 / (1 + exp(-z))

    for c in bounded01:

        jitter = inv_logit(
            logit(col(c)) + randn(seed + 8) * logit_level_b01
        )

        df = df.withColumn(
            c,
            when(is_feature_noise, jitter).otherwise(col(c))
        )

    for c in num_cols:

        df = df.withColumn(
            c,
            when(
                is_feature_noise & (rand(seed + 9) < mcar_level),
                lit(None)
            ).otherwise(col(c))
        )

    return df
