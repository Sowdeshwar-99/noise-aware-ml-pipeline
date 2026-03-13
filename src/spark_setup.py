from pyspark.sql import SparkSession

def create_spark():

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("CapstoneNoiseSimulation_LightGBM")
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.3")
        .config("spark.driver.memory", "45g")
        .config("spark.executor.memory", "16g")
        .config("spark.sql.shuffle.partitions", "12")
        .getOrCreate()
    )

    return spark
