

"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile
# import awswrangler as wr

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import sys
import subprocess



logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

#logger.info("Installing aws-wrangler")
# subprocess.check_call([
#    sys.executable, "-m", "pip", "install", "-r",
#    "/opt/ml/processing/requirements.txt",
# ])

# Since we get a headerless CSV file we specify the column names here.
# feature_columns_names = [
#     "sex",
#     "length",
#     "diameter",
#     "height",
#     "whole_weight",
#     "shucked_weight",
#     "viscera_weight",
#     "shell_weight",
# ]
label_column = "target"

# feature_columns_dtype = {
#     "sex": str,
#     "length": np.float64,
#     "diameter": np.float64,
#     "height": np.float64,
#     "whole_weight": np.float64,
#     "shucked_weight": np.float64,
#     "viscera_weight": np.float64,
#     "shell_weight": np.float64,
# }
label_column_dtype = {"target": int}


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    # key = "/".join(input_data.split("/")[3:])
    key = "features/train/feature_train.csv"
    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/train-dataset.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    # s3://px-data-ml-ops-data/features/telesales_gsa/
    # df = wr.s3.read_parquet(input_data)


    logger.debug("Reading downloaded data.")
    df = pd.read_csv(fn)
    os.unlink(fn)

    logger.debug("Defining transformers.")
    # Due to the fact that we have a ready to use feature vector we don't need to much preprocessing

    # numeric_features = list(feature_columns_names)
    # numeric_features.remove("sex")
    # numeric_transformer = Pipeline(
    #     steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    # )

    # categorical_features = ["sex"]
    # categorical_transformer = Pipeline(
    #     steps=[
    #         ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    #         ("onehot", OneHotEncoder(handle_unknown="ignore")),
    #     ]
    # )

    # preprocess = ColumnTransformer(
    #     transformers=[
    #         ("num", numeric_transformer, numeric_features),
    #         ("cat", categorical_transformer, categorical_features),
    #     ]
    # )

    logger.info("Applying transforms.")
    df.drop(['customer_key', 'campaing_id'], axis=1)
    y = df.pop("target")
    # we are not doing any transformation
    #X_pre = preprocess.fit_transform(df)
    # we are just conveting to numpy array
    X_pre = df.to_numpy()
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)