#!/usr/bin/env python3
"""
CPU/loopback benchmark of the model fetcher against an in-process S3 (moto). It does NOT measure internet speed; it proves the
implementation is correct under concurrency and shows that SHA-256 verification is not the bottleneck (hash MB/s vs a 1 Gbit/s link).
  python tools/model_fetch_bench.py [--files 4] [--mb 96]
"""
import argparse, hashlib, os, pathlib, sys, tempfile, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import boto3
from moto.server import ThreadedMotoServer

from scripts.ops import model_fetch

ap = argparse.ArgumentParser(); ap.add_argument("--files", type=int, default=4); ap.add_argument("--mb", type=int, default=96)
a = ap.parse_args()
srv = ThreadedMotoServer(port=0, verbose=False); srv.start()
host, port = srv.get_host_and_port()
s3 = boto3.client("s3", endpoint_url=f"http://{host}:{port}", aws_access_key_id="x", aws_secret_access_key="y", region_name="us-east-1")
s3.create_bucket(Bucket="bench")
tmp = pathlib.Path(tempfile.mkdtemp())
datas = {f"m{i}": os.urandom(a.mb << 20) for i in range(a.files)}
for k, d in datas.items():
    s3.put_object(Bucket="bench", Key=f"models/{k}", Body=d)
total = a.files * a.mb
t0 = time.time()
for k in datas:                                                            # old behaviour: one by one, default transfer config, no hash
    s3.download_file("bench", f"models/{k}", str(tmp / f"seq_{k}"))
t_seq = time.time() - t0
items = [model_fetch.Item(k, f"models/{k}", tmp / "par" / f"{k}.pt", False, hashlib.sha256(d).hexdigest(), len(d)) for k, d in datas.items()]
t0 = time.time(); res = model_fetch.fetch_all(items, client=s3, bucket="bench"); t_par = time.time() - t0
assert all(r.status == "verified" for r in res)
t0 = time.time(); h = model_fetch.sha256_file(items[0].local); t_h = time.time() - t0
t0 = time.time(); res2 = model_fetch.fetch_all(items, client=s3, bucket="bench"); t_cached = time.time() - t0
assert all(r.status == "cached" for r in res2)
print(f"{a.files} files x {a.mb} MB = {total} MB over loopback (not an internet measurement)")
print(f"sequential boto3 default, no hash : {t_seq:6.2f} s  ({total/t_seq:7.0f} MB/s)")
print(f"parallel + SHA-256 verified       : {t_par:6.2f} s  ({total/t_par:7.0f} MB/s)")
print(f"SHA-256 alone                     : {a.mb/t_h:7.0f} MB/s  (a 1 Gbit/s link is ~125 MB/s -> hashing is {a.mb/t_h/125:.1f}x faster than the wire)")
print(f"second start, files cached+sidecar: {t_cached:6.3f} s  (no download, no re-hash)")
srv.stop()
