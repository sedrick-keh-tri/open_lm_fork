import glob
import argparse
import collections
import enum
import gzip
import hashlib
import io
import json
import os
import sys
import random
import resource
import tarfile
import time
import traceback
import glob
from enum import Enum
from io import BytesIO
from typing import BinaryIO, List

import boto3
import fsspec
import jsonlines
import numpy as np
import pandas as pd
import psutil
import ray
import webdataset as wds
import zstandard as zstd
from botocore.exceptions import (
    IncompleteReadError,
    ReadTimeoutError,
    ResponseStreamingError,
)
from braceexpand import braceexpand
from loguru import logger
from ray._private.internal_api import memory_summary
from ray.data._internal.util import _check_pyarrow_version
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask
from ray.runtime_context import RuntimeContext
from tqdm import tqdm
from transformers import GPTNeoXTokenizerFast

import logging


import yaml
import pathlib

# Initialize an empty dictionary for sampling frequencies

DIR = pathlib.Path(__file__).parent.absolute()


def load_from_yaml(filename):
    SAMPLING_FREQUENCIES = {}

    with open(filename, "r") as file:
        data = yaml.safe_load(file)

    # Dynamically create the Sources enum based on YAML file
    Sources = enum.Enum("Sources", {item["source"]: index for index, item in enumerate(data["sources"])})

    # Add get_source and get_sampling_frequency methods to Sources
    def get_source_dynamic(self, key):
        for item in data["sources"]:
            if any(marker in key for marker in item["markers"]):
                return Sources[item["source"]]
        return Sources.UNKNOWN

    def get_sampling_frequency_dynamic(self, key):
        return SAMPLING_FREQUENCIES[self.get_source(key)]

    Sources.get_source = classmethod(get_source_dynamic)
    Sources.get_sampling_frequency = classmethod(get_sampling_frequency_dynamic)

    # Load sampling frequencies
    for key, value in data["sampling_frequencies"].items():
        source = Sources[key]
        SAMPLING_FREQUENCIES[source] = value
    return Sources, SAMPLING_FREQUENCIES


class RawFileType(enum.Enum):
    JSONL = 1
    ZSTD_JSONL_COMPRESSED = 2
    GZIP_JSONL_COMPRESSED = 3
    TAR = 4
    UNKNOWN = -1


def jsonl_file_reader(fh: BinaryIO, content_key: str):
    with io.TextIOWrapper(fh, encoding="utf-8") as text_reader:
        with jsonlines.Reader(text_reader) as jsonl_reader:
            for item in jsonl_reader:
                yield item[content_key]


def zstd_compressed_reader(fh: BinaryIO, content_key: str):
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(fh) as reader:
        for item in jsonl_file_reader(reader, content_key=content_key):
            yield item


def gzip_compressed_reader(fh: BinaryIO, content_key: str):
    with gzip.open(fh, "rb") as f_in:
        with jsonlines.Reader(f_in) as jsonl_reader:
            for item in jsonl_reader:
                yield item[content_key]


def tar_reader(fh: BinaryIO, content_key: str):
    """
    content_key: where in the tarfile to find the text/tokens. Options:
        "txt" - read text file as string
        "json:key" - read json[key] as string
        "npy" - read numpy array as tokens
    """
    content_ext = content_key.split(":")[0]
    buffer = io.BytesIO(fh.read())
    with tarfile.open(fileobj=buffer, mode="r") as tar:
        samples = []
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(f".{content_ext}"):
                with tar.extractfile(member) as fileobj:
                    if fileobj:  # Ensure fileobj is not None
                        if content_ext == "txt":
                            content = fileobj.read().decode("utf-8")
                        elif content_ext == "json":
                            json_dict, json_key = json.load(fileobj), content_key.split(":")[1]
                            content = json_dict[json_key]
                        elif content_ext == "npy":
                            token_array = np.load(io.BytesIO(fileobj.read()), allow_pickle=True)
                            content = token_array.reshape(-1).tolist()
                        else:
                            raise ValueError(f"Unsupported content key extension: {content_key}")

                        yield content


def get_reader(file_type, content_key: str):
    if file_type == RawFileType.JSONL:
        return lambda x: jsonl_file_reader(x, content_key=content_key)
    if file_type == RawFileType.ZSTD_JSONL_COMPRESSED:
        return lambda x: zstd_compressed_reader(x, content_key=content_key)
    if file_type == RawFileType.GZIP_JSONL_COMPRESSED:
        return lambda x: gzip_compressed_reader(x, content_key=content_key)
    if file_type == RawFileType.TAR:
        return lambda x: tar_reader(x, content_key=content_key)
    else:
        raise Exception("Unsupported filetype")


def get_raw_filetype(key: str):
    if key.endswith(".jsonl") or key.endswith(".json"):
        return RawFileType.JSONL
    elif key.endswith(".jsonl.zst") or key.endswith(".json.zst"):
        return RawFileType.ZSTD_JSONL_COMPRESSED
    elif key.endswith(".jsonl.gz") or key.endswith(".json.gz"):
        return RawFileType.GZIP_JSONL_COMPRESSED
    elif key.endswith(".tar"):
        return RawFileType.TAR
    else:
        logger.warning(f"Unknown filetype: {key}")
        return RawFileType.UNKNOWN


@ray.remote
class GlobalCounter:
    def __init__(self):
        self.value = 0
        self.token_count = 0

    def increment(self):
        self.value += 1
        return self.value

    def increment_token_count(self, num_tokens):
        self.token_count += num_tokens
        return self.token_count

    def get_counter(self):
        return self.value

    def get_token_counter(self):
        return self.token_count


def preprocess(
    key: str,
    fh: BinaryIO,
    seed: int,
    content_key: str,
    seqlen: int = 8192,
    tokenizer=None,
    do_sample: bool = False,
    sources: enum.Enum = None,
    source_counter: GlobalCounter = None,
):
    tokenizer_fn, vocab_size = tokenizer
    rng = random.Random(hash(key) + seed)
    EOT = SpecialTokens.END_OF_TEXT.value % (vocab_size + len(SpecialTokens))
    PAD = SpecialTokens.PAD.value % (vocab_size + len(SpecialTokens))
    if do_sample:
        assert sources is not None
        sample_freq = sources.get_sampling_frequency(key)
    buffer = []
    try:
        file_type = get_raw_filetype(key)
        if file_type == RawFileType.UNKNOWN:
            return []
        file_reader = get_reader(file_type, content_key)
        pbar = tqdm(file_reader(fh))
        pbar.set_description(key)
        for string in pbar:
            tokens = tokenizer_fn(string)
            tokens.append(EOT)
            buffer += tokens
            while len(buffer) >= seqlen:
                if do_sample:
                    local_sample_freq = sample_freq
                    # This code does the following
                    # yield a int(sample_freq) copies of buffer[:seqlen]
                    # then yield 1 more sample with Pr[sample_freq - int(sample_freq)]
                    # in expectation we will yield sample_freq copies of buffer[:seqlen]
                    while local_sample_freq > 1:
                        if source_counter is not None:
                            ray.get(source_counter.increment_token_count.remote(seqlen))
                        yield buffer[:seqlen]
                        local_sample_freq -= 1
                    if rng.random() < local_sample_freq:
                        if source_counter is not None:
                            ray.get(source_counter.increment_token_count.remote(seqlen))
                        yield buffer[:seqlen]
                    buffer = buffer[seqlen:]
                else:
                    if source_counter is not None:
                        ray.get(source_counter.increment_token_count.remote(seqlen))
                    yield buffer[:seqlen]
                    buffer = buffer[seqlen:]
        if len(buffer) > 0:
            if source_counter is not None:
                ray.get(source_counter.increment_token_count.remote(len(buffer)))
            yield buffer + [PAD] * (seqlen - len(buffer))

    except (IncompleteReadError, ReadTimeoutError, ResponseStreamingError) as e:
        logger.error(f"There was an incomplete read error: {e} for key {key}")
        return []


def process_keys(data, tokenizer, seqlen, seed, content_key, do_sample, sources=None, source_counters=None):
    path = data["path"]

    if path.startswith("s3"):
        s3_client = boto3.client("s3")
        bucket, key = parse_s3_path(path)
        response = s3_client.get_object(Bucket=bucket, Key=key)
        fh = response["Body"]
    else:
        key = path
        fh = open(path, "rb")

    try:
        # select a counter
        if sources is not None and source_counters is not None:
            source_counter = source_counters[sources.get_source(key)]
        else:
            source_counter = None
        # Process the file stream (either S3 or local)
        token_buffers = preprocess(
            key,
            fh,
            seqlen=seqlen,
            seed=seed,
            tokenizer=tokenizer,
            content_key=content_key,
            do_sample=do_sample,
            sources=sources,
            source_counter=source_counter,
        )

        # Ensure that all operations on the file handle are done within this block
        for token_buffer in token_buffers:
            yield {"tokens": token_buffer}
    finally:
        # Close the file handle/stream after all operations are done
        fh.close()


class SpecialTokens(Enum):
    END_OF_TEXT = 0
    PAD = -1
    END_OF_DOCUMENT = -2


def parse_s3_path(s3_path):
    """
    Extract the bucket and key from an S3 path.

    Args:
    - s3_path (str): The S3 path in the format "s3://bucket/key"

    Returns:
    - tuple: A tuple containing the bucket and key
    """
    if not s3_path.startswith("s3://"):
        raise ValueError("Invalid S3 path format. Must start with 's3://'")

    s3_parts = s3_path[5:].split("/", 1)
    bucket = s3_parts[0]

    if len(s3_parts) > 1:
        key = s3_parts[1]
    else:
        key = ""
    return bucket, key


def add_hash(item, column="tokens"):
    item["hash"] = hash(str(item[column]))
    return item


def map_write_wds(batch, batch_size, folder, counter):
    tar_index = ray.get(counter.increment.remote())
    digits = 8  # default to 8
    # Format tar index with the determined number of leading zeros
    tar_index_str = f"{tar_index:0{digits}}"
    # Create tar file name
    tar_name = f"{tar_index_str}.tar"
    token_count = 0
    # Write the batch to a tarball using webdataset's TarWriter
    bio = io.BytesIO()
    with wds.TarWriter(bio) as sink:
        for i in range(len(batch["tokens"])):
            tokens = [int(x) for x in batch["tokens"][i]]
            token_count += len(tokens)
            json_string = json.dumps(tokens)
            uid = hashlib.md5(json_string.encode()).hexdigest()
            sample = {"__key__": uid, "json.gz": json_string}
            sink.write(sample)
    bio.seek(0)
    token_count = ray.get(counter.increment_token_count.remote(token_count))
    write_to_location(folder, tar_name, bio)

    return_dict = {"shard": [tar_name.split(".")[0]], "num_sequences": [len(batch["tokens"])]}

    return return_dict


def write_to_location(folder, tar_name, bio):
    path = f"{folder}/{tar_name}"

    # Check if the path is an S3 path
    if path.startswith("s3://"):
        s3 = boto3.client("s3")

        # Properly extract bucket and key from the S3 path
        s3_path_parts = path[5:].split("/")
        bucket = s3_path_parts[0]
        key = "/".join(s3_path_parts[1:])

        try:
            s3.put_object(Bucket=bucket, Key=key, Body=bio.getvalue())
        except Exception as e:
            assert False, f"bucket is {bucket} key is {key} and {e}"

    else:
        # Create directory if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        try:
            with open(path, "wb") as f:
                f.write(bio.getvalue())
        except Exception as e:
            assert False, f"error is {path} and {e}"


def load_tokenizer(tokenizer):
    if tokenizer == "EleutherAI/gpt-neox-20b":
        enc = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
        return (lambda x: enc(x).input_ids, enc.vocab_size)
    else:
        raise ValueError(f"Unknown Tokenizer: {tokenizer}")


def glob_files(path, suffix=".jsonl"):
    """
    Glob files based on a given path and suffix.
    Supports both local and S3 paths.

    :param path: path to glob. Can be local or S3 (e.g., s3://bucket-name/path/)
    :param suffix: suffix of files to match. Defaults to ".jsonl"
    :return: list of file paths matching the pattern
    """
    if path.startswith("s3://"):
        # Use boto3 for S3 paths
        s3 = boto3.client("s3")
        bucket_name, prefix = path[5:].split("/", 1)

        # Ensure the prefix ends with a '/'
        if not prefix.endswith("/"):
            prefix += "/"

        # List the objects in the bucket with the given prefix
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        all_files = [f"s3://{bucket_name}/{obj['Key']}" for objects in pages for obj in objects.get("Contents", [])]

        # Filter out the files based on the suffix
        matching_files = [f for f in all_files if f.endswith(suffix)]

    else:
        # Use glob for local paths
        search_pattern = f"{path.rstrip('/')}/**/*{suffix}"
        matching_files = glob.glob(search_pattern, recursive=True)
        print("matching files with suffix: ", suffix)
        print(matching_files)

    return matching_files


def write_manifest(jsonl_lines, args):
    "Write manifest to provided output path."

    output_path = os.path.join(args.output.strip("/"), "manifest.jsonl")

    if output_path.startswith("s3://"):
        # Use boto3 for S3 paths
        s3_client = boto3.client("s3")
        jsonl_content = "\n".join(json.dumps(record) for record in jsonl_lines) + "\n"  # Add a newline at the end
        bucket_name, s3_key = output_path[5:].split("/", 1)
        response = s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=jsonl_content)
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            logging.warning(
                "Failed to write manifest. Please manually include manifest by running "
                "open_lm.utils.make_manifest on the tokenized data."
            )
    else:
        with open(output_path, "w") as f:
            for item in jsonl_lines:
                json.dump(item, f)
                f.write("\n")


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input path", type=str, required=True)
    parser.add_argument(
        "--output",
        help="output path",
        type=str,
        required=True
        # e.g s3://dcnlp-data/rpj_tokenized_upsampled_eleutherai_deduplicated/
    )
    parser.add_argument("--content_key", type=str, default="text")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--vocab_size", type=int, default=None)  # for pre-tokenized data, don't load tokenizer
    parser.add_argument("--wds_chunk_size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--subfraction", type=float, default=None)
    parser.add_argument("--ray_address", type=str, default=None)
    parser.add_argument("--block_size", type=str, default="10MB")
    parser.add_argument("--force_parallelism", type=int, default=None)
    parser.add_argument("--force_num_cores", type=int, default=None)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--ray_spill_location", type=str, default="/tmp/ray_spill")
    parser.add_argument("--default_dataset_yaml", type=str, default=(DIR.parent / "metadata" / "rpj_lm_data.yaml"))
    parser.add_argument(
        "--ray_dashboard_host", type=str, default="127.0.0.1"
    )  # default is localhost; for slurm jobs do 0.0.0.0

    args = parser.parse_args(args)
    if args.do_sample:
        Sources, SAMPLING_FREQUENCIES = load_from_yaml(args.default_dataset_yaml)
        logger.info(f"SOURCES:\n {Sources}")
        logger.info(f"SAMPLING_FREQUENCIES:\n{SAMPLING_FREQUENCIES}")
    else:
        Sources, SAMPLING_FREQUENCIES = None, None
    # configure remote spilling
    creds = {k: v for k, v in os.environ.items() if k.startswith("AWS")}
    runtime_env = {"env_vars": creds}

    if args.force_num_cores is not None:
        num_cores = args.force_num_cores
    else:
        num_cores = os.cpu_count()

    print(f"num cores = {num_cores}")
    if args.ray_address is None:
        ray.init(
            runtime_env=runtime_env,
            _temp_dir=args.ray_spill_location,
            dashboard_host=args.ray_dashboard_host,
        )
    else:
        ray.init(
            args.ray_address,
            runtime_env=runtime_env,
            _temp_dir=args.ray_spill_location,
            dashboard_host=args.ray_dashboard_host,
        )
    num_nodes = len(ray.nodes())
    input_folders = args.input.split(",")
    input_paths = []
    for inp_folder in input_folders:
        input_paths += glob_files(inp_folder, suffix=".json")
        input_paths += glob_files(inp_folder, suffix=".jsonl")
        input_paths += glob_files(inp_folder, suffix=".zst")
        input_paths += glob_files(inp_folder, suffix=".jsonl.gz")
        input_paths += glob_files(inp_folder, suffix=".tar")
        input_paths += glob_files(inp_folder, suffix=".gz")
    rng = random.Random(args.seed)
    rng.shuffle(input_paths)  # shuffle before selecting subsets
    if args.subset is not None:
        input_paths = input_paths[: args.subset]
    if args.subfraction is not None:
        input_paths = input_paths[: int(args.subfraction * len(input_paths))]
    print("Files considered: \n", input_paths)
    print(f"num files ={len(input_paths)}")
    num_files = len(input_paths)

    output_path = args.output
    seqlen = args.seqlen + 1
    wds_chunk_size = args.wds_chunk_size
    content_key = args.content_key
    if args.force_parallelism is not None:
        parallelism = args.force_parallelism
    else:
        parallelism = num_cores * num_nodes
    ctx = DataContext.get_current()
    ctx.use_push_based_shuffle = True
    ctx.execution_options.resource_limits.object_store_memory = float("inf")
    ray.data.DataContext.get_current().execution_options.verbose_progress = True
    start_time = time.time()
    tokenizer = load_tokenizer(args.tokenizer) if args.vocab_size is None else (lambda x: x, args.vocab_size)
    logger.info(f"Total number of keys = {len(input_paths)}")
    df = pd.DataFrame(input_paths, columns=["path"])
    ds = ray.data.from_pandas(pd.DataFrame(input_paths, columns=["path"])).repartition(parallelism)

    # dictionary with counters to keep track of the tokens for each source
    if Sources is not None:
        source_counters = {source: GlobalCounter.remote() for source in Sources}
    else:
        source_counters = None

    ds = ds.flat_map(
        lambda x: process_keys(
            x,
            tokenizer=tokenizer,
            seqlen=seqlen,
            seed=args.seed,
            content_key=content_key,
            do_sample=args.do_sample,
            sources=Sources,
            source_counters=source_counters,
        )
    )
    ds = ds.map(add_hash)
    tokenize_context_end = time.time()
    # sorting by hash is a random shuffle
    if not args.no_shuffle:
        ds = ds.sort(key="hash")
    else:
        ds = ds.repartition(num_cores * num_nodes, shuffle=False)
    counter = GlobalCounter.remote()
    ds = ds.map_batches(
        map_write_wds,
        batch_size=wds_chunk_size,
        fn_kwargs={
            "batch_size": wds_chunk_size,
            "folder": args.output.rstrip("/"),
            "counter": counter,
        },
        batch_format="pandas",
    )

    # Sort by shard name
    ds = ds.repartition(1)
    ds = ds.sort(key="shard")
    jsonl_lines = ds.take_all()
    write_manifest(jsonl_lines, args)

    end_time = time.time()
    duration = end_time - start_time
    final_token_count = ray.get(counter.increment_token_count.remote(0))
    print("==== Token count summary ====")
    print(f"Tokenize + Shuffle script Finished in: {duration}")
    print(f"Final Token Count: {final_token_count}")
    if Sources is not None:
        for source, counter in source_counters.items():
            token_count = ray.get(counter.increment_token_count.remote(0))
            print(f"Source: {source}, Token count: {token_count}")

    print("==== Driver memory summary ====")
    maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1e3)
    print(f"max: {maxrss / 1e9}/GB")
    process = psutil.Process(os.getpid())
    rss = int(process.memory_info().rss)
    print(f"rss: {rss / 1e9}/GB")
    try:
        print(memory_summary(stats_only=True))
    except Exception:
        print("Failed to retrieve memory summary")
        print(traceback.format_exc())
    print("")


if __name__ == "__main__":
    main(sys.argv[1:])
