import collections
import os
import resource
import sys
import time

import pandas as pd
import tensorflow as tf

from recommender_system.recsys_core.recsys_utils import print_with_date, read_metadata_from_file
from recommender_system.recsys_tf.recsys_tfrs_model import TfrsModelMaker

AM_AWS_REGION = "us-east-1"
AM_S3_ENDPOINT = f"s3.{AM_AWS_REGION}.amazonaws.com"
AM_S3_USE_HTTPS = "0"
AM_S3_VERIFY_SSL = "0"
TF_LOG_LEVEL = "3"
NUM_TRAIN_EPOCHS = 3


def get_inputs(in_path_data):
    items_path = os.path.join(in_path_data, "items")
    users_path = os.path.join(in_path_data, "users")
    events_path = os.path.join(in_path_data, "events")

    metadata_path = os.path.join(os.path.join(in_path_data, "metadata"), "metadata.csv")
    metadata_dict = read_metadata_from_file(metadata_path)
    num_songs = int(metadata_dict["num_songs"])
    num_users = int(metadata_dict["num_users"])
    num_events = int(metadata_dict["num_events"])

    return items_path, users_path, events_path, num_songs, num_users, num_events


def get_test_users(in_path_test_user_list):
    print(">> Getting the test users...")
    dtypes = collections.OrderedDict({"user_id": str, "url_slug": str, "user_name": str})
    df = pd.read_csv(in_path_test_user_list, names=list(dtypes.keys()), dtype=dtypes, encoding="utf-8")
    # tuples of (user id, slug, user name)
    return list(df.itertuples(index=False, name=None))


def main(args):
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = TF_LOG_LEVEL
    os.environ["AWS_REGION"] = AM_AWS_REGION
    os.environ["S3_ENDPOINT"] = AM_S3_ENDPOINT
    os.environ["S3_USE_HTTPS"] = AM_S3_USE_HTTPS
    os.environ["S3_VERIFY_SSL"] = AM_S3_VERIFY_SSL

    if "." not in sys.path:
        sys.path.insert(0, ".")

    # Using allow_soft_placement=True allows TF to fall back to CPU when no GPU implementation is available.
    tf.config.set_soft_device_placement(True)

    start_time = time.time()

    print()
    print_with_date(">> Running the TFRS based song recommender...")

    s3_in_path_data = args[1]
    s3_in_path_test_users = args[2]

    (items_path, users_path, events_path, num_songs, num_users, num_events,) = get_inputs(s3_in_path_data)

    # Load the intermediary parquet data into TF datasets
    model_maker = TfrsModelMaker(items_path, users_path, events_path, num_songs, num_users, num_events)
    # Build the model
    model, train_events_ds, test_events_ds = model_maker.create_model()
    # Train and evaluate the model
    model_maker.train_and_evaluate(model, NUM_TRAIN_EPOCHS, train_events_ds, test_events_ds)

    # Get recomms for test users
    # TODO keep this as test option o/w use users from users_path
    input_users = get_test_users(s3_in_path_test_users)
    model_maker.generate_recommendations(model, model_maker.items_ds, input_users)

    # TODO option to delete the input data prepared for the TFRS based recsys, from S3

    elapsed_time = time.time() - start_time
    str_elapsed_time = time.strftime("%H : %M : %S", time.gmtime(elapsed_time))

    print()
    print_with_date(">> TFRS based song recommender done. Duration: {}.".format(str_elapsed_time))
    print()


if __name__ == "__main__":
    main(sys.argv)
