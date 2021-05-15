import os
import re
import tempfile
from abc import ABC
from typing import Dict, Text

import boto3
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_recommenders as tfrs
import urllib3

from recommender_system.recsys_core.recsys_utils import print_with_date

# from tqdm.keras import TqdmCallback

#
# Create/initialize this early to avoid RuntimeError "Collective ops must be configured at program startup"
# described here: https://github.com/tensorflow/tensorflow/issues/34568
#
# strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.get_strategy()
strategy = tf.distribute.MultiWorkerMirroredStrategy()


batch_size_items = 100
batch_size_train = 100
batch_size_test = 50


class TfrsModel(tfrs.Model, ABC):
    def __init__(self, user_model, item_model, loss_task):
        super().__init__()
        self.item_model: tf.keras.Model = item_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = loss_task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # Pick out the item features and pass them into the item model, getting embeddings back.
        item_embeddings = self.item_model(features["item_id"])

        # The task computes the loss and the metrics.
        # return self.task(user_embeddings, item_embeddings, compute_metrics=False)
        return self.task(user_embeddings, item_embeddings, compute_metrics=not training)


class TfrsModelMaker(object):
    def __init__(self, items_path, users_path, events_path, num_items, num_users, num_events):
        self.items_path = items_path
        self.users_path = users_path
        self.events_path = events_path
        self.num_items = num_items
        self.num_users = num_users
        self.num_events = num_events

        print()
        print(">> Initializing TfrsModelMaker...")
        print(">> items_path  : {}".format(items_path))
        print(">> users_path  : {}".format(users_path))
        print(">> events_path : {}".format(events_path))
        print(">> num_items   : {}".format(num_items))
        print(">> num_users   : {}".format(num_users))
        print(">> num_events  : {}".format(num_events))
        print()

        # Turn off the many Unverified HTTPS request warnings during file downloads.
        urllib3.disable_warnings()
        self.items_ds, self.events_ds = self._load_tf_datasets()
        self.test_events_ds, self.train_events_ds = self._prepare_data()

    def create_model(self):
        embedding_dimension = 32

        print(">> Strategy: {}".format(strategy))
        print(">> Number of devices: {}".format(strategy.num_replicas_in_sync))
        gpus = tf.config.list_physical_devices("GPU")
        print(">> GPU's: {}".format(gpus))

        with strategy.scope():
            user_ids_filepath = self.get_s3_filepaths(self.users_path, ".csv")[0]
            item_ids_filepath = self.get_s3_filepaths(self.items_path, ".csv")[0]

            # The query tower
            u_lookup = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=user_ids_filepath)
            # u_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(
            #     vocabulary=user_ids_filepath, mask_token=None
            # )
            user_model = tf.keras.Sequential(
                [
                    u_lookup,
                    # We add an additional embedding to account for unknown tokens.
                    tf.keras.layers.Embedding(u_lookup.vocab_size() + 1, embedding_dimension),
                ]
            )

            # The candidate tower
            c_lookup = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=item_ids_filepath)
            # c_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(
            #     vocabulary=item_ids_filepath, mask_token=None
            # )
            item_model = tf.keras.Sequential(
                [
                    c_lookup,
                    # We add an additional embedding to account for unknown tokens.
                    tf.keras.layers.Embedding(c_lookup.vocab_size() + 1, embedding_dimension),
                ]
            )

            # Metrics
            cands = self.items_ds.map(item_model)
            metrics = tfrs.metrics.FactorizedTopK(candidates=cands)

            # Loss
            task = tfrs.tasks.Retrieval(metrics=metrics)

            # cached_train_event_ds = self.train_events_ds.batch(8192).cache()
            # cached_test_event_ds = self.test_events_ds.batch(4096).cache()

            # cached_train_event_ds = self.train_events_ds.cache()
            # cached_test_event_ds = self.test_events_ds.cache()

            # per_replica_batch_size = 64
            # global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
            #
            # print()
            # print(">> BATCH SIZE: {}".format(global_batch_size))
            # print()
            #
            # cached_train_event_ds = self.train_events_ds.batch(global_batch_size).cache()
            # cached_test_event_ds = self.test_events_ds.batch(global_batch_size).cache()

            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

            opt_train_events_ds = self.train_events_ds.batch(batch_size_train).with_options(options)
            opt_test_events_ds = self.test_events_ds.batch(batch_size_test).with_options(options)

            model = TfrsModel(user_model, item_model, task)

            # https://github.com/tensorflow/recommenders/issues/269
            # Adagrad for embeddings isn't implemented on GPUs.
            # model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))

        return model, opt_train_events_ds, opt_test_events_ds

    @staticmethod
    def train_and_evaluate(model, num_epochs, train_events_ds, test_events_ds):
        print_with_date(">> Training the model...")

        # Train the model
        model.fit(train_events_ds, epochs=num_epochs)

        # sz = sum([tf.size(x).numpy() for x in model.variables])
        # print(">> SZ: " + str(sz))

        # model.save("/home/ec2-user/tfrs_proto/tfrs.model")

        # turn off keras progress (verbose=0) and use tqdm instead. For the callback:
        # verbose=2 means separate progressbars for epochs and batches
        # 1 means clear batch bars when done
        # 0 means only show epochs (never show batch bars)
        # model.fit(model.cached_train_event_ds, epochs=num_epochs, verbose=0, callbacks=[TqdmCallback(verbose=2)])

        print_with_date(">> Training of the model: done.")

        # Evaluate the model
        print_with_date(">> Evaluating the model...")
        eval_results = model.evaluate(test_events_ds, return_dict=True)
        print_with_date(">> Evaluation of the model: done.")

        print()
        print(f">> Eval results (epochs={num_epochs}):")
        print(str(eval_results))
        print()

    def _load_tf_datasets(self):
        print(">> Loading TF datasets from S3...")

        # options = tf.data.Options()
        # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        temp_dirpath = tempfile.mkdtemp()

        print(">> --- Loading the ITEMS dataset from {}...".format(self.items_path))
        local_file_list = self.download_files(self.items_path, temp_dirpath, "items", ".csv")
        items_ds = tf.data.experimental.make_csv_dataset(
            local_file_list,
            column_names=["item_id"],
            batch_size=batch_size_items,
            num_parallel_reads=50,
            sloppy=True,
            num_epochs=1,
        )
        items_ds = items_ds.map(lambda item: item["item_id"])
        print(">> --- ITEMS dataset: loaded.")

        print(">> --- Loading the EVENTS dataset from {}...".format(self.events_path))
        # Load the events
        events_filepaths = self.get_s3_filepaths(self.events_path, ".parquet")
        print(">> Events filepaths: " + str(events_filepaths))
        events_columns = ["user_id", "item_id"]
        events_ds = self.load_dataset("events", events_filepaths, events_columns)
        events_ds = events_ds.map(lambda event: {"item_id": event["item_id"], "user_id": event["user_id"]})
        # events_ds = events_ds.with_options(options)
        print(">> --- EVENTS dataset: loaded")

        print(">> Loading TF datasets from S3: done.")

        return items_ds, events_ds

    @staticmethod
    def load_dataset(ds_name, files, columns):
        print(f">> Loading {files[0]} for {ds_name}...")
        dataset = tfio.IODataset.from_parquet(files[0], columns=columns)

        for file_name in files[1:]:
            print(f">> Loading {file_name} for {ds_name}...")
            ds = tfio.IODataset.from_parquet(file_name, columns=columns)
            dataset = dataset.concatenate(ds)

        return dataset

    @staticmethod
    def download_files(s3_dirpath, temp_dirpath, subdir_name, postfix):
        s3_file_list = TfrsModelMaker.get_s3_filepaths(s3_dirpath, postfix)

        bucket_name, path = TfrsModelMaker.get_s3_uri_parts(s3_dirpath)
        s3 = boto3.resource("s3", verify=False)
        bucket = s3.Bucket(bucket_name)

        local_fpaths = []
        local_subdir_path = os.path.join(temp_dirpath, subdir_name)
        os.mkdir(local_subdir_path)

        for s3_filepath in s3_file_list:
            _, path = TfrsModelMaker.get_s3_uri_parts(s3_filepath)
            target_fpath = os.path.join(local_subdir_path, os.path.basename(s3_filepath))
            print(">> downloading {} ({}, {}) to {}...".format(s3_filepath, bucket_name, path, target_fpath))
            bucket.download_file(path, target_fpath)
            local_fpaths.append(target_fpath)

        return local_fpaths

    @staticmethod
    def get_s3_filepaths(s3_dir_uri, postfix):
        bucket_name, dirpath = TfrsModelMaker.get_s3_uri_parts(s3_dir_uri)
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(bucket_name)

        fpaths = []
        parent_path = f"s3://{bucket_name}"
        for object_summary in bucket.objects.filter(Prefix=dirpath):
            if not postfix or object_summary.key.endswith(postfix):
                fpaths.append(os.path.join(parent_path, object_summary.key))
        return fpaths

    @staticmethod
    def get_s3_uri_parts(s3_uri):
        matches = re.search("(.*)://([^/]*)/(.*)", s3_uri)
        bucket_name = matches.group(2)
        path = matches.group(3)
        return bucket_name, path

    def _prepare_data(self):
        print(">> Preparing data...")
        tf.random.set_seed(42)

        size_80_percent = int(self.num_events * 0.8)
        size_20_percent = self.num_events - size_80_percent

        # (data is pre-shuffled)
        train_events_ds = self.events_ds.take(size_80_percent)
        test_events_ds = self.events_ds.skip(size_80_percent).take(size_20_percent)

        print(">> Data preparation: done.")

        return test_events_ds, train_events_ds

    @staticmethod
    def generate_recommendations(model, items_ds, input_users):
        # Making predictions

        print()
        print(">> Generating recs - 1...")
        print()

        # Create a model that takes in raw query features, and
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model, k=20)

        print(">> Generating recs - 2...")

        # recommends items out of the entire items dataset.
        index.index(items_ds.map(model.item_model), items_ds)

        print(">> Generating recs - 3...")

        print()
        print("*" * 80)
        print()
        for (user_id, url_slug, user_name) in input_users:
            # Get recommendations for the given user
            ratings_for_user, item_ids_for_user = index(tf.constant([user_id]))
            print()
            print(">> Recommendations for user %s - %s - %s:", user_id, user_name, url_slug)
            ratings_arr = ratings_for_user[0]
            item_ids_arr = item_ids_for_user[0]
            for idx in range(0, len(ratings_arr)):
                print(f"\t{ratings_arr[idx]} -- {item_ids_arr[idx].numpy().decode('utf-8')}")
            print()
            print("*" * 80)
            print()
