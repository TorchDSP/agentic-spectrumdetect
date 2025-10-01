#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python version of tye_sp_ad_retune (C++ implementation).

This script:
1. Parses command‑line arguments.
2. Listens for an advertisement (JSON) on a UDP port.
3. Sends a retune request (JSON) to the advertised address/port.
4. Waits for a retune status reply and reports success/failure.

The behaviour mirrors the original C++ program, including timeout handling
and console output formatting.
"""

import argparse
import json
import select
import socket
import struct
import sys
import time

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
THIS_DFLT_AD_PORT = 61111          # default advertisement listening port
RECV_TIMEOUT_SEC = 2               # seconds to wait for advertisement / status
MAX_JSON_DECIMAL_PLACES = 6        # matches C++ Writer setting

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def display_help():
    """Print usage information (mirrors the C++ help output)."""
    print()
    print("----------------------------------------------------------------------------------------------------")
    print("USAGE: ./tye_sp_ad_return.py [ARGS]")
    print("----------------------------------------------------------------------------------------------------")
    print(f"  --ad-port .......... [OPTIONAL] Advertisement port (default is {THIS_DFLT_AD_PORT})")
    print("  --sample-rate-hz ... [REQUIRED] Retune sample rate value")
    print("  --center-freq-hz ... [REQUIRED] Retune center frequency value")
    print("  --atten-db ......... [REQUIRED] Retune attenuation value")
    print("  --ref-level ........ [REQUIRED] Retune reference level value")
    print("  --help ............. Display help")
    print()


def parse_cmdline_args():
    """Parse command line arguments, reproducing the validation logic of the C++ version."""
    parser = argparse.ArgumentParser(add_help=False, prog="tye_sp_ad_return.py")
    parser.add_argument("--ad-port", type=int, default=THIS_DFLT_AD_PORT)
    parser.add_argument("--sample-rate-hz", type=int, required=False)
    parser.add_argument("--center-freq-hz", type=int, required=False)
    parser.add_argument("--atten-db", type=int, required=False)
    parser.add_argument("--ref-level", type=float, required=False)
    parser.add_argument("--help", action="store_true")

    args, unknown = parser.parse_known_args()

    # If user asked for help or supplied unknown options, show help.
    if args.help or unknown:
        display_help()
        sys.exit(0)

    # Manual validation to keep the same error messages as the C++ code.
    missing = False
    if args.sample_rate_hz is None:
        print("\nARG => --sample-rate-hz REQUIRED")
        missing = True
    if args.center_freq_hz is None:
        print("\nARG => --center-freq-hz REQUIRED")
        missing = True
    if args.atten_db is None:
        print("\nARG => --atten-db REQUIRED")
        missing = True
    if args.ref_level is None:
        print("\nARG => --ref-level REQUIRED")
        missing = True

    if missing:
        display_help()
        sys.exit(1)

    return args


def recv_ad(ad_port):
    """
    Listen for an advertisement JSON on the given UDP port.
    Returns (dst_ipaddr, dst_port) on success, or (None, None) on failure/timeout.
    """
    print(">> OPENING AD SOCKET ", end="", flush=True)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError as e:
        print("[FAIL]")
        print(f"Socket creation error: {e}")
        return None, None

    # Set socket options
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setblocking(False)

    try:
        sock.bind(("", ad_port))
    except OSError as e:
        print("[FAIL]")
        print(f"Bind error on port {ad_port}: {e}")
        sock.close()
        return None, None

    print("[OK]")
    print(">> WAITING FOR AD ", end="", flush=True)

    deadline = time.time() + RECV_TIMEOUT_SEC
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            print("\n[TIMEOUT]")
            sock.close()
            return None, None

        rlist, _, _ = select.select([sock], [], [], remaining)
        if not rlist:
            continue

        try:
            data, addr = sock.recvfrom(1024)
        except OSError:
            continue

        try:
            msg = json.loads(data.decode())
        except json.JSONDecodeError:
            print("\n[PARSE ERROR] [FAIL]")
            continue

        # Validate message structure
        if not isinstance(msg, dict):
            print("\n[MSG IS NOT AN OBJECT] [FAIL]")
            continue

        if msg.get("msg_type") != "ad":
            # Not an advertisement; ignore
            continue

        if "retune_port" not in msg:
            print("\n[KEY \"retune_port\" NOT FOUND] [FAIL]")
            continue

        if not isinstance(msg["retune_port"], int):
            print("\n[KEY \"retune_port\" INCORRECT TYPE] [FAIL]")
            continue

        print(f"\n[OK] => {data.decode()}")
        dst_ipaddr = socket.inet_aton(addr[0])  # bytes representation
        dst_ip = struct.unpack("!I", dst_ipaddr)[0]  # uint32 network order
        dst_port = int(msg["retune_port"])
        sock.close()
        return dst_ip, dst_port


def send_retune_msg_wait_status(dst_ip, dst_port, sample_rate_hz,
                               center_freq_hz, atten_db, ref_level):
    """
    Send a retune JSON message to the destination and wait for a status reply.
    Returns True on success, False otherwise.
    """
    print(">> OPENING RETUNE SOCKET ", end="", flush=True)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError as e:
        print("[FAIL]")
        print(f"Socket creation error: {e}")
        return False

    sock.setblocking(False)
    print("[OK]")

    # Build retune JSON
    print(">> BUILDING RETUNE MSG ", end="", flush=True)
    msg = {
        "msg_type": "retune",
        "sample_rate_hz": sample_rate_hz,
        "center_freq_hz": center_freq_hz,
        "atten_db": atten_db,
        "ref_level": ref_level,
    }

    # Ensure decimal places similar to C++ Writer setting
    json_msg = json.dumps(msg, separators=(',', ':'), ensure_ascii=False)
    print("[OK]")

    # Send the message
    print(">> SENDING RETUNE MSG ", end="", flush=True)
    dst_ip_str = socket.inet_ntoa(struct.pack("!I", dst_ip))
    try:
        sent = sock.sendto(json_msg.encode(), (dst_ip_str, dst_port))
    except OSError as e:
        print("[FAIL]")
        print(f"Send error: {e}")
        sock.close()
        return False

    if sent != len(json_msg):
        print("[FAIL]")
        sock.close()
        return False

    print(f"[OK] DST [{dst_ip_str}:{dst_port}] MSG {json_msg}")

    # Wait for status reply
    print(">> WAITING FOR RETUNE STATUS ", end="", flush=True)
    deadline = time.time() + RECV_TIMEOUT_SEC
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            print("\n[TIMEOUT]")
            sock.close()
            return False

        rlist, _, _ = select.select([sock], [], [], remaining)
        if not rlist:
            continue

        try:
            data, _ = sock.recvfrom(256)
        except OSError:
            continue

        try:
            status_msg = json.loads(data.decode())
        except json.JSONDecodeError:
            print("\n[PARSE ERROR] [FAIL]")
            break

        if not isinstance(status_msg, dict):
            print("\n[MSG IS NOT AN OBJECT] [FAIL]")
            break

        if "msg_type" not in status_msg or "status" not in status_msg:
            missing_key = "msg_type" if "msg_type" not in status_msg else "status"
            print(f"\n[KEY \"{missing_key}\" NOT FOUND] [FAIL]")
            break

        msg_type = status_msg["msg_type"]
        status = status_msg["status"]

        if msg_type != "retune_status":
            print("\n[MSG TYPE INCORRECT] [FAIL]")
            break

        if status == "success":
            print("\n[SUCCESS]")
        else:
            print("\n[FAIL]")
        break

    sock.close()
    return True


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    args = parse_cmdline_args()

    # Receive advertisement
    dst_ip, dst_port = recv_ad(args.ad_port)
    if dst_ip is None:
        sys.exit(-1)
    # Send retune request and wait for status
    ok = send_retune_msg_wait_status(
        dst_ip,
        dst_port,
        args.sample_rate_hz,
        args.center_freq_hz,
        args.atten_db,
        args.ref_level,
    )
    if not ok:
        sys.exit(-1)

    print()
    sys.exit(0)


if __name__ == "__main__":
    main()
