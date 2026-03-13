from src.spark_setup import create_spark
from src.data_preprocessing import load_parquet_dataset, clean_dataset
from src.feature_selection import (
    get_numeric_features,
    compute_feature_correlations,
    get_top_features,
    select_top_features
)
from src.noise_simulation import sim_noise
from src.feature_mitigation import mitig_feat
from src.label_mitigation import mitig_labl
from src.model_training import (
    prepare_features,
    train_lightgbm,
    evaluate_model,
    split_data
)

import pandas as pd


def main():

    #Spark session
    spark = create_spark()

    #Load dataset
    data_path = "data"

    df = load_parquet_dataset(spark, data_path)

    #Cleaning dataset
    df_clean = clean_dataset(df)

    #Feature selection
    numeric_cols = get_numeric_features(df_clean)

    corr_df = compute_feature_correlations(df_clean, numeric_cols)

    top_features = get_top_features(corr_df, top_n=15)

    feature_list = top_features["feature"].tolist()

    df_selected = select_top_features(df_clean, feature_list)

    #Noise simulation
    df_noisy = sim_noise(df_selected, noise_tag="20", total_noise_fraction=0.20)

    #Convert to pandas for feature mitigation
    df_noisy_pd = df_noisy.toPandas()

    #Feature mitigation
    df_mitigated_pd = mitig_feat(df_noisy_pd)

    #Back to Spark
    df_mitigated = spark.createDataFrame(df_mitigated_pd)

    #Label mitigation
    df_final = mitig_labl(spark, df_mitigated)

    #Model training
    df_prepared = prepare_features(df_final)

    train_df, test_df = split_data(df_prepared)

    model = train_lightgbm(train_df)

    metrics = evaluate_model(model, test_df)

    print("\nFinal Model Performance")

    print(metrics)


if __name__ == "__main__":
    main()
