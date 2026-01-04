#!/usr/bin/env python3
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NIF_PATHS = [
    os.path.join(ROOT, "native", "generated_nif.c"),
    os.path.join(ROOT, "native", "fp_job_nif.c"),
]
OUT_PATH = os.path.join(ROOT, "lib", "merkle_db", "fp_signatures_generated.ex")


ARG_PATTERNS = [
    ("enif_inspect_binary", "binary", r"enif_inspect_binary\([^,]+,\s*argv\[(\d+)\],\s*&([A-Za-z0-9_]+)\)"),
    ("enif_get_uint64", "u64", r"enif_get_uint64\([^,]+,\s*argv\[(\d+)\],\s*[^&]*&([A-Za-z0-9_]+)\)"),
    ("enif_get_int64", "i64", r"enif_get_int64\([^,]+,\s*argv\[(\d+)\],\s*[^&]*&([A-Za-z0-9_]+)\)"),
    ("enif_get_double", "f64", r"enif_get_double\([^,]+,\s*argv\[(\d+)\],\s*[^&]*&([A-Za-z0-9_]+)\)"),
    ("enif_get_int", "i32", r"enif_get_int\([^,]+,\s*argv\[(\d+)\],\s*[^&]*&([A-Za-z0-9_]+)\)"),
    ("enif_get_uint", "u32", r"enif_get_uint\([^,]+,\s*argv\[(\d+)\],\s*[^&]*&([A-Za-z0-9_]+)\)"),
    ("enif_get_atom", "atom", r"enif_get_atom\([^,]+,\s*argv\[(\d+)\],\s*([A-Za-z0-9_]+)"),
]

RESOURCE_PATTERN = r"enif_get_resource\([^,]+,\s*argv\[(\d+)\],\s*RES_TYPE_([A-Za-z0-9_]+)"


def to_snake(name):
    out = []
    for idx, ch in enumerate(name):
        if ch.isupper() and idx > 0 and name[idx - 1].islower():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def clean_arg_name(name):
    for prefix in ("bin_", "val_", "size_", "out_bin_", "ptr_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def parse_registry(text):
    registry = {}
    nif_map = {}
    pattern = re.compile(r'\{"([^"]+)",\s*(\d+),\s*nif_([A-Za-z0-9_]+)([^}]*)\}')
    for match in pattern.finditer(text):
        name = match.group(1)
        arity = int(match.group(2))
        nif_name = match.group(3)
        dirty = "ERL_NIF_DIRTY_JOB_CPU_BOUND" in match.group(4)
        registry[name] = {"arity": arity, "dirty": dirty}
        nif_map[nif_name] = name
    return registry, nif_map


def split_top_level_args(value):
    args = []
    depth = 0
    current = []
    for ch in value:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            arg = "".join(current).strip()
            if arg:
                args.append(arg)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def infer_expr_type(expr, block_lines):
    if "enif_make_tuple" in expr:
        return parse_tuple_return(expr, block_lines)
    if "enif_make_binary" in expr:
        return "binary"
    if "enif_make_double" in expr:
        return "f64"
    if "enif_make_uint64" in expr:
        return "u64"
    if "enif_make_int64" in expr:
        return "i64"
    if "enif_make_int(" in expr:
        return "i32"
    if "enif_make_uint(" in expr:
        return "u32"
    if "enif_make_atom" in expr:
        if "\"true\"" in expr and "\"false\"" in expr:
            return "bool"
        return "atom"
    if "enif_make_resource" in expr:
        res_type = resource_type_from_block(block_lines)
        if res_type:
            return "resource:" + res_type
        return "resource"
    if "enif_make_map" in expr:
        return "map"
    return "term"


def parse_tuple_return(expr, block_lines):
    match = re.search(r"enif_make_tuple(\d+)\((.*)\)", expr)
    if not match:
        return "tuple"
    args = split_top_level_args(match.group(2))
    if args and args[0].strip() == "env":
        args = args[1:]
    types = [infer_expr_type(arg, block_lines) for arg in args]
    return "tuple<" + ",".join(types) + ">"


def resource_type_from_block(block_lines):
    for line in block_lines:
        match = re.search(r"RES_TYPE_([A-Za-z0-9_]+)", line)
        if match:
            return match.group(1)
    return None


def parse_return_type(block_lines):
    for line in reversed(block_lines):
        stripped = line.strip()
        if stripped.startswith("return ") and "badarg" not in stripped:
            expr = stripped[len("return "):].strip()
            if expr.endswith(";"):
                expr = expr[:-1]
            return infer_expr_type(expr, block_lines)
    return "term"


def parse_args(block_lines, arity):
    arg_map = {}

    for _, arg_type, pattern in ARG_PATTERNS:
        for line in block_lines:
            match = re.search(pattern, line)
            if match:
                idx = int(match.group(1))
                name = match.group(2)
                if idx not in arg_map:
                    arg_map[idx] = {"name": clean_arg_name(name), "type": arg_type}

    for line in block_lines:
        match = re.search(RESOURCE_PATTERN, line)
        if match:
            idx = int(match.group(1))
            res_type = match.group(2)
            if idx not in arg_map:
                arg_map[idx] = {
                    "name": to_snake(res_type),
                    "type": "resource:" + res_type,
                }

    args = []
    for idx in range(arity):
        if idx in arg_map:
            args.append(arg_map[idx])
        else:
            args.append({"name": "arg{}".format(idx + 1), "type": "term"})
    return args


def parse_functions(text, nif_map, registry):
    signatures = {}
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = re.match(r"static ERL_NIF_TERM nif_([A-Za-z0-9_]+)\(", line)
        if not match:
            idx += 1
            continue
        nif_name = match.group(1)
        block = []
        brace_depth = 0
        while idx < len(lines):
            line = lines[idx]
            block.append(line)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth == 0 and line.strip().endswith("}"):
                break
            idx += 1
        name = nif_map.get(nif_name, nif_name)
        arity = registry.get(name, {}).get("arity", 0)
        args = parse_args(block, arity) if arity else []
        returns = parse_return_type(block)
        signatures[name] = {
            "name": name,
            "arity": arity,
            "args": args,
            "returns": returns,
            "source": "generated_nif.c",
        }
        idx += 1
    return signatures


def apply_manual_entries(signatures, registry):
    manual = {
        "fp_job_start": {
            "args": [
                {"name": "op_name", "type": "atom_or_binary"},
                {"name": "args", "type": "list"},
                {"name": "opts", "type": "term"},
            ],
            "returns": "resource:FPJob",
            "source": "fp_job_nif.c",
        },
        "fp_job_status": {
            "args": [{"name": "job", "type": "resource:FPJob"}],
            "returns": "map",
            "source": "fp_job_nif.c",
        },
        "fp_job_result": {
            "args": [{"name": "job", "type": "resource:FPJob"}],
            "returns": "tuple",
            "source": "fp_job_nif.c",
        },
        "fp_job_cancel": {
            "args": [{"name": "job", "type": "resource:FPJob"}],
            "returns": "atom",
            "source": "fp_job_nif.c",
        },
    }

    for name, info in manual.items():
        if name in registry and name not in signatures:
            signatures[name] = {
                "name": name,
                "arity": registry[name]["arity"],
                "args": info["args"],
                "returns": info["returns"],
                "source": info["source"],
            }
    return signatures


def build_output(signatures, registry):
    entries = []
    for name in sorted(registry.keys()):
        sig = signatures.get(name)
        if not sig:
            sig = {
                "name": name,
                "arity": registry[name]["arity"],
                "args": [{"name": "arg{}".format(i + 1), "type": "term"} for i in range(registry[name]["arity"])],
                "returns": "term",
                "source": "registry",
            }
        sig["dirty"] = registry[name]["dirty"]
        entries.append(sig)
    return entries


def render_elixir(signatures):
    lines = []
    lines.append("# AUTO-GENERATED by scripts/gen_fp_signatures.py. DO NOT EDIT.")
    lines.append("defmodule MerkleDb.FPSignaturesGenerated do")
    lines.append("  @moduledoc false")
    lines.append("")
    lines.append("  @signatures [")
    for sig in signatures:
        lines.append("    %{" )
        lines.append('      name: "{}",'.format(sig["name"]))
        lines.append("      arity: {},".format(sig["arity"]))
        lines.append("      dirty: {},".format("true" if sig.get("dirty") else "false"))
        lines.append('      returns: "{}",'.format(sig["returns"]))
        lines.append('      source: "{}",'.format(sig.get("source", "unknown")))
        lines.append("      args: [")
        for arg in sig["args"]:
            lines.append('        %{name: "{}", type: "{}"},'.format(arg["name"], arg["type"]))
        lines.append("      ]")
        lines.append("    },")
    lines.append("  ]")
    lines.append("")
    lines.append("  def all, do: @signatures")
    lines.append("end")
    lines.append("")
    return "\n".join(lines)


def main():
    for path in NIF_PATHS:
        if not os.path.exists(path):
            print("Missing file: {}".format(path), file=sys.stderr)
            return 1

    with open(NIF_PATHS[0], "r", encoding="utf-8") as handle:
        text = handle.read()
    registry, nif_map = parse_registry(text)
    signatures = parse_functions(text, nif_map, registry)

    if os.path.exists(NIF_PATHS[1]):
        with open(NIF_PATHS[1], "r", encoding="utf-8") as handle:
            job_text = handle.read()
        signatures.update(parse_functions(job_text, nif_map, registry))

    signatures = apply_manual_entries(signatures, registry)
    output = build_output(signatures, registry)
    rendered = render_elixir(output)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(rendered)

    print("Wrote {}".format(OUT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
