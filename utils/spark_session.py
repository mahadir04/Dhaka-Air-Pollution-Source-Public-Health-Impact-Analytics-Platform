"""
Reusable Spark session factory.

Every script in the project should call get_spark() instead of building
its own SparkSession — this guarantees consistent settings and avoids
accidental multiple-session issues.
"""

from pyspark.sql import SparkSession
from utils.config import SPARK_APP_NAME, SPARK_DRIVER_MEMORY, SPARK_LOG_LEVEL
import os
import sys

# Ensure HADOOP_HOME is set and its bin directory is in PATH on Windows
if sys.platform == "win32":
    hadoop_home = os.environ.get("HADOOP_HOME", r"E:\DA_project\hadoop")
    os.environ["HADOOP_HOME"] = hadoop_home
    hadoop_bin = os.path.join(hadoop_home, "bin")
    if hadoop_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

# Ensure Spark uses the same Python executable for workers to avoid connection timeouts
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

_spark = None


def get_spark(app_name: str | None = None) -> SparkSession:
    """Return a project-wide SparkSession (singleton).

    Parameters
    ----------
    app_name : str, optional
        Override the default application name from config.

    Returns
    -------
    SparkSession
    """
    global _spark
    if _spark is None or _spark._jsc.sc().isStopped():
        _spark = (
            SparkSession.builder
            .appName(app_name or SPARK_APP_NAME)
            .master("local[*]")
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
            .config("spark.sql.parquet.compression.codec", "snappy")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        _spark.sparkContext.setLogLevel(SPARK_LOG_LEVEL)
    return _spark


def stop_spark() -> None:
    """Stop the active SparkSession, if any."""
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None
