"""Activate and configure CAN interfaces for Piper robot arms.

Each -p argument maps a USB port to a CAN interface name. Without -p, all
detected interfaces are brought up at the default bitrate keeping their
existing names.

Example:
    python scripts/hardware/can_activate.py -p 1-1:1.0=can_arm1 -p 1-2:1.0=can_arm2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

BITRATE = 1_000_000


def ip_link(*args: str) -> None:
    subprocess.run(["ip", "link", *args], check=True)


def get_can_interfaces() -> list[dict[str, Any]]:
    """Return all CAN-type network interfaces."""
    result = subprocess.run(
        ["ip", "-details", "-json", "link", "show", "type", "can"],
        capture_output=True,
        text=True,
        check=True,
    )
    links = json.loads(result.stdout)

    interfaces = []
    for link in links:
        name = link["ifname"]
        bittiming = (
            link.get("linkinfo", {}).get("info_data", {}).get("bittiming", {}) or {}
        )
        bitrate = bittiming.get("bitrate", 0)

        device_path = Path(f"/sys/class/net/{name}/device")
        bus_info = device_path.resolve().name if device_path.is_symlink() else None

        interfaces.append(
            {
                "name": name,
                "is_up": "UP" in link.get("flags", []),
                "bitrate": bitrate,
                "bus_info": bus_info,
            }
        )
    return interfaces


def configure(iface: dict[str, Any], target_name: str) -> None:
    """Set bitrate, rename, and bring up a CAN interface."""
    name = iface["name"]

    needs_reconfig = not iface["is_up"] or iface["bitrate"] != BITRATE
    if needs_reconfig:
        ip_link("set", name, "down")
        ip_link("set", name, "type", "can", "bitrate", str(BITRATE))
        ip_link("set", name, "up")

    if name != target_name:
        ip_link("set", name, "down")
        ip_link("set", name, "name", target_name)
        ip_link("set", target_name, "up")

    print(f"  {name} ({iface['bus_info']}) -> {target_name} @ {BITRATE}")


def main() -> None:
    if os.geteuid() != 0:
        os.execvp("sudo", ["sudo", sys.executable, *sys.argv])

    parser = argparse.ArgumentParser(description="Activate CAN interfaces.")
    parser.add_argument(
        "-p",
        "--port",
        dest="ports",
        action="append",
        metavar="USB=NAME",
        help="USB port mapping (repeatable), e.g. 1-1:1.0=can_arm1",
    )
    args = parser.parse_args()

    subprocess.run(["modprobe", "gs_usb"], capture_output=True, check=False)

    interfaces = get_can_interfaces()
    if not interfaces:
        sys.exit("Error: no CAN interfaces detected")

    if args.ports:
        by_bus = {i["bus_info"]: i for i in interfaces}
        for entry in args.ports:
            parts = entry.split("=")
            if len(parts) != 2:
                sys.exit(f"Error: invalid mapping '{entry}', use USB=NAME")
            usb_addr, name = parts
            if usb_addr not in by_bus:
                print(f"  Warning: no interface at USB port '{usb_addr}', skipping")
                continue
            configure(by_bus[usb_addr], name)
    else:
        for iface in interfaces:
            configure(iface, iface["name"])


if __name__ == "__main__":
    main()
