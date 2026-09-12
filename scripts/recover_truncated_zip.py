"""Recover intact local entries from a ZIP missing its central directory."""
import argparse
import pathlib
import struct
import zlib

LOCAL_HEADER = struct.Struct("<4s5H3I2H")


def recover(source, destination):
    data = pathlib.Path(source).read_bytes()
    destination = pathlib.Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    offset = 0
    recovered = 0
    while offset + LOCAL_HEADER.size <= len(data):
        header = LOCAL_HEADER.unpack_from(data, offset)
        if header[0] != b"PK\x03\x04":
            break
        _, _, flags, method, _, _, crc, compressed_size, _, name_size, extra_size = header
        name_start = offset + LOCAL_HEADER.size
        name_end = name_start + name_size
        payload_start = name_end + extra_size
        payload_end = payload_start + compressed_size
        if payload_end > len(data) or not name_size:
            break
        name = pathlib.PurePosixPath(data[name_start:name_end].decode("utf-8"))
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Unsafe archive path: {name}")
        payload = data[payload_start:payload_end]
        if method == 0:
            content = payload
        elif method == 8:
            content = zlib.decompress(payload, -15)
        else:
            raise ValueError(f"Unsupported compression method {method} for {name}")
        if (zlib.crc32(content) & 0xFFFFFFFF) != crc:
            break
        target = destination.joinpath(*name.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        recovered += 1
        offset = payload_end
    print(f"Recovered {recovered} entries from {source}")
    print(f"Output: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    recover(args.source, args.destination)
