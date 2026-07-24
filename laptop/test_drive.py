#!/usr/bin/env python3
"""
Quick diagnostic: drives forward for 3 seconds, then stops.
Bypasses the ML model — useful to verify Pi connection & motor control work.

Usage:
    python test_drive.py --pi 192.168.100.30
"""

import argparse
import json
import socket
import sys
import time

# DRIVE_INVERTED: laptop "forward" = protocol "BACKWARD" (motor wired reverse)
FWD_CMD = 'BACKWARD'  # change to 'FORWARD' if your motor wiring is normal


def send(sock, drive, steer, speed):
    msg = json.dumps({'command': drive, 'steer': steer, 'speed': speed})
    sock.sendall(msg.encode('utf-8'))
    print(f"  sent: {msg}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pi', required=True)
    p.add_argument('--port', type=int, default=5555)
    p.add_argument('--seconds', type=float, default=3.0)
    p.add_argument('--speed', type=int, default=50)
    p.add_argument('--reverse', action='store_true', help='Drive in reverse instead of forward')
    args = p.parse_args()

    drive_cmd = 'FORWARD' if args.reverse else FWD_CMD
    label = 'REVERSE' if args.reverse else 'FORWARD'

    print(f"Connecting to Pi @ {args.pi}:{args.port}...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect((args.pi, args.port))
    print("Connected!")
    print()

    print(f"--- Driving {label} for {args.seconds}s at speed {args.speed} ---")
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        send(s, drive_cmd, 'STEER_STOP', args.speed)
        time.sleep(0.2)

    print()
    print("--- STOPPING ---")
    for _ in range(5):
        send(s, 'STOP', 'STEER_STOP', 0)
        time.sleep(0.1)

    s.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
