from pyspark.sql import functions as F


def load_parquet_dataset(spark, data_path):
    """
    Loads all parquet files from the specified directory.
    """
    df = spark.read.parquet(f"{data_path}/*.parquet")

    print("Total rows loaded:", df.count())
    df.printSchema()

    return df


def clean_dataset(df):
    """
    Basic dataset cleaning:
    - Removes rows where yield <= 0
    """

    df_clean = df.filter(df["yield"] > 0)

    print("Rows after removing zero yields:", df_clean.count())

    return df_clean


def show_summary_statistics(df):
    """
    Prints summary statistics for key agronomic variables.
    """

    df.describe(["yield", "rain", "maxt", "mint", "biomass"]).show()


def check_missing_values(df):
    """
    Displays number of missing values per column.
    """

    missing = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    )

    missing.show(truncate=False)
