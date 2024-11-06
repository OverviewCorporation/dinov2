import logging
import os
import shutil
from typing import Tuple

import tensorrt as trt

log = logging.getLogger(__name__)


EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
MB = 1 << 20


def _load_trt_cache(cache_file: str) -> Tuple[trt.ITimingCache, float]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # You have to create a network and a parser. If not it Seg faults
    network = builder.create_network(EXPLICIT_BATCH)
    trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    for flag in ["FP16"]:
        config.set_flag(getattr(trt.BuilderFlag, flag))

    with open(cache_file, 'rb') as init_f:
        timing_cache = config.create_timing_cache(init_f.read())
        config.set_timing_cache(timing_cache, ignore_mismatch=True)
        size = os.path.getsize(cache_file) / MB
        return timing_cache, size


def _combine_caches(
    cache_file: str,
    timing_cache: trt.ITimingCache,
    other_timing_cache: trt.ITimingCache,
) -> Tuple[bool, float]:
    with open(cache_file, 'rb+') as cache_f:
        combined_successfully = timing_cache.combine(
            other_timing_cache, ignore_mismatch=True
        )
        cache_f.write(timing_cache.serialize())

        curr_size = os.path.getsize(cache_file) / MB
        return combined_successfully, curr_size


def _merge_trt_caches(init_cache_file: str, cache_file: str) -> None:
    if not os.path.exists(cache_file) or os.path.getsize(cache_file) > 100 * MB:
        # if cache file does not exist or if the cache is too big (>100mb), copy the init cache file
        shutil.copyfile(init_cache_file, cache_file)
        size = os.path.getsize(cache_file) / MB
        log.info(f"TRT cache initialized successfully. Size: {size:.2f}MB.")
    else:
        # if the cache file exists and it's not too big, combine it with the init cache
        log.info("Start combining caches")

        init_timing_cache, init_size = _load_trt_cache(init_cache_file)
        log.info(
            f"Finished loading init cache. TRT init cache size: {init_size:.2f}MB."
        )

        curr_timing_cache, curr_size = _load_trt_cache(cache_file)
        log.info(
            f"Finished loading current cache. TRT current cache size: {curr_size:.2f}MB."
        )

        combined_successfully, new_size = _combine_caches(
            cache_file, curr_timing_cache, init_timing_cache
        )
        log.info(
            f"TRT cache was combined = {combined_successfully}. New size: {new_size:.2f}MB."
        )


def init_trt_cache() -> None:
    try:
        base_cache_path = "/app/timing_caches"
        base_init_cache_path = "/app/timing_caches_init"
        os.makedirs(base_cache_path, exist_ok=True)

        cache_filename = "trt.cache"
        cache_file = os.path.join(base_cache_path, cache_filename)
        init_cache_file = os.path.join(base_init_cache_path, cache_filename)
        _merge_trt_caches(init_cache_file, cache_file)
    except Exception:
        log.exception("Error while initializing TRT cache")


def onnx2trt(
    onnx_path: str,
    trt_path: str,
    timing_cache_path: str,
    flags: list = ["FP16"],
    max_memory_gb: float = 0.5,
) -> None:
    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    for flag in flags:
        config.set_flag(getattr(trt.BuilderFlag, flag))
    config.max_workspace_size = int(max_memory_gb * (1 << 30))

    network = builder.create_network(EXPLICIT_BATCH)

    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(onnx_path)

    if not os.path.exists(timing_cache_path):
        open(timing_cache_path, 'w').close()

    with open(timing_cache_path, 'rb') as f:
        timing_cache = config.create_timing_cache(f.read())
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    engine = builder.build_serialized_network(network, config)

    with open(trt_path, 'wb') as f:
        f.write(engine)

    with open(timing_cache_path, 'wb') as f:
        timing_cache.combine(config.get_timing_cache(), ignore_mismatch=True)
        f.write(timing_cache.serialize())
