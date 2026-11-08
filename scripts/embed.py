#!/usr/bin/env python3
"""Writes a binary file as a C byte array: embed.py INPUT OUTPUT SYMBOL"""

import sys

src, dst, name = sys.argv[1:4]
with open(src, "rb") as f:
    data = f.read()

with open(dst, "w") as f:
    f.write("#include <stddef.h>\n\n")
    # cuModuleLoadData wants an aligned image.
    f.write(f"const unsigned char {name}[] __attribute__((aligned(16))) = {{\n")
    for i in range(0, len(data), 16):
        f.write("  " + ", ".join(str(b) for b in data[i : i + 16]) + ",\n")
    f.write("};\n")
    f.write(f"const size_t {name}_size = {len(data)};\n")
