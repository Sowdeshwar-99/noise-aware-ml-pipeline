from synapse.ml.lightgbm import LightGBMRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F


def prepare_features(df):

    feat_cols = [
        c for (c, t) in df.dtypes
        if t in ["double", "float", "int", "bigint"]
        and c not in ["yield", "yield_mitigated", "DOY", "zadok_stage"]
    ]

    assembler = VectorAssembler(
        inputCols=feat_cols,
        outputCol="features",
        handleInvalid="keep"
    )

    if "yield_mitigated" in df.columns:
        target = "yield_mitigated"
    else:
        target = "yield"

    df_prepared = assembler.transform(df).select(
        "features",
        F.col(target).alias("label")
    )

    return df_prepared


def train_lightgbm(train_df):

    lgbm = LightGBMRegressor(
        featuresCol="features",
        labelCol="label",
        objective="regression",
        learningRate=0.05,
        numLeaves=64,
        maxDepth=-1,
        numIterations=500,
        baggingFraction=0.8,
        featureFraction=0.8,
        seed=42
    )

    model = lgbm.fit(train_df)

    return model


def evaluate_model(model, test_df):

    evaluator = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction"
    )

    preds = model.transform(test_df)

    rmse = evaluator.evaluate(preds, {evaluator.metricName: "rmse"})
    mae = evaluator.evaluate(preds, {evaluator.metricName: "mae"})
    r2 = evaluator.evaluate(preds, {evaluator.metricName: "r2"})

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

    return metrics


def split_data(df):

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    return train_df, test_df
