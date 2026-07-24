#!/usr/bin/env python3
"""
Laptop Remote Control for Vehicle
WASD + Arrow keys, works on any Linux terminal
"""

import socket
import json
import threading
import time
import sys
import os
import tty
import termios
import select

PI_IP = input("Enter Raspberry Pi IP (e.g. 192.168.1.100): ").strip()
PI_PORT = 5555


class RemoteControl:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connected = False
        self.running = True

        self.drive = 'STOP'
        self.steer = 'STEER_STOP'
        self.auto_mode = False
        self.lock = threading.Lock()

        self.status = None
        self.status_lock = threading.Lock()

    def connect(self):
        print(f"Connecting to {self.ip}:{self.port}...")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.ip, self.port))
            self.sock.settimeout(0.5)
            self.connected = True
            print("Connected!")
            return True
        except Exception as e:
            print(f"Failed: {e}")
            return False

    def send(self, drive, steer):
        if not self.connected:
            return
        try:
            msg = json.dumps({'command': drive, 'steer': steer})
            self.sock.sendall(msg.encode('utf-8'))
        except:
            self.connected = False

    def sender_loop(self):
        """Keep sending current drive + steer every 300ms to keep watchdog alive"""
        while self.running and self.connected:
            with self.lock:
                d, s = self.drive, self.steer
            self.send(d, s)

            # Auto-reset steering after sending once
            # This prevents steering from running forever
            # User must keep pressing A/D to keep steering
            with self.lock:
                if self.steer in ('LEFT', 'RIGHT'):
                    self.steer = 'STEER_STOP'

            time.sleep(0.3)

    def receiver_loop(self):
        buf = ""
        while self.running and self.connected:
            try:
                data = self.sock.recv(4096)
                if not data:
                    self.connected = False
                    break
                buf += data.decode('utf-8')
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    try:
                        with self.status_lock:
                            self.status = json.loads(line)
                    except json.JSONDecodeError:
                        pass
            except socket.timeout:
                continue
            except:
                self.connected = False
                break

    def display_loop(self):
        # Clear screen once at start
        os.system('clear')

        while self.running:
            # Move cursor to top-left instead of clearing (no flicker)
            sys.stdout.write('\033[H')

            with self.lock:
                d, s, auto = self.drive, self.steer, self.auto_mode

            lines = []
            lines.append("=" * 55)
            lines.append("         VEHICLE REMOTE CONTROL")
            lines.append("=" * 55)
            lines.append(f"  Mode:       {'AUTONOMOUS' if auto else 'MANUAL'}")
            lines.append(f"  Connected:  {self.connected}")
            lines.append(f"  Drive:      {d:10s}")
            lines.append(f"  Steer:      {s:10s}")

            if self.status:
                with self.status_lock:
                    st = dict(self.status)
                lines.append(f"  Speed:      {st.get('actual_speed', 0)}%")
                lines.append(f"  Alert:      {st.get('alert_level', '?'):15s}")
                auto_state = st.get('auto_state', '')
                if auto_state:
                    lines.append(f"  Auto State: {auto_state:15s}")
                else:
                    lines.append(f"              {'':15s}")
                lines.append("-" * 55)
                dist = st.get('distances', {})
                mf = st.get('min_distance_front', 0)
                mb = st.get('min_distance_back', 0)
                lines.append(f"  FL={dist.get('FL',0):5.1f}  FW={dist.get('FW',0):5.1f}  FR={dist.get('FR',0):5.1f}  [Front: {mf:.1f}]")
                lines.append(f"  BC={dist.get('BC',0):5.1f}                       [Back:  {mb:.1f}]")
                lines.append(f"  LS={dist.get('LS',0):5.1f}  RS={dist.get('RS',0):5.1f}")
            else:
                lines.append("  Waiting for Pi data...")
                lines.append("")
                lines.append("")
                lines.append("-" * 55)
                lines.append("")
                lines.append("")
                lines.append("")

            lines.append("-" * 55)
            lines.append("  W/Up=Forward  S/Down=Backward  SPACE=Stop")
            lines.append("  A/Left=Left   D/Right=Right    X=Straight")
            lines.append("  T=Autonomous  Q/ESC=Quit")
            lines.append("=" * 55)

            # Write all lines at once + clear any leftover chars
            output = '\n'.join(f'{line:55s}' for line in lines) + '\n'
            sys.stdout.write(output)
            sys.stdout.flush()

            time.sleep(0.5)

    def read_keys(self):
        old = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while self.running:
                ch = sys.stdin.read(1)

                with self.lock:
                    if ch in ('w', 'W'):
                        if not self.auto_mode:
                            self.drive = 'FORWARD'
                            self.send('FORWARD', self.steer)

                    elif ch in ('s', 'S'):
                        if not self.auto_mode:
                            self.drive = 'BACKWARD'
                            self.send('BACKWARD', self.steer)

                    elif ch in ('a',):
                        if not self.auto_mode:
                            self.steer = 'LEFT'
                            self.send(self.drive, 'LEFT')

                    elif ch in ('d',):
                        if not self.auto_mode:
                            self.steer = 'RIGHT'
                            self.send(self.drive, 'RIGHT')

                    elif ch in ('x', 'X'):
                        if not self.auto_mode:
                            self.steer = 'STEER_STOP'
                            self.send(self.drive, 'STEER_STOP')

                    elif ch == ' ':
                        self.auto_mode = False
                        self.drive = 'STOP'
                        self.steer = 'STEER_STOP'
                        self.send('STOP', 'STEER_STOP')

                    elif ch in ('t', 'T'):
                        self.auto_mode = not self.auto_mode
                        self.send('AUTONOMOUS', 'STEER_STOP')
                        if self.auto_mode:
                            self.drive = 'STOP'
                            self.steer = 'STEER_STOP'

                    elif ch in ('q', 'Q', '\x03'):
                        self.send('EMERGENCY', 'STEER_STOP')
                        self.running = False
                        break

                    elif ch == '\x1b':
                        if select.select([sys.stdin], [], [], 0.01)[0]:
                            ch2 = sys.stdin.read(1)
                            if ch2 == '[':
                                ch3 = sys.stdin.read(1)
                                if not self.auto_mode:
                                    if ch3 == 'A':
                                        self.drive = 'FORWARD'
                                        self.send('FORWARD', self.steer)
                                    elif ch3 == 'B':
                                        self.drive = 'BACKWARD'
                                        self.send('BACKWARD', self.steer)
                                    elif ch3 == 'C':
                                        self.steer = 'RIGHT'
                                        self.send(self.drive, 'RIGHT')
                                    elif ch3 == 'D':
                                        self.steer = 'LEFT'
                                        self.send(self.drive, 'LEFT')
                            else:
                                self.send('EMERGENCY', 'STEER_STOP')
                                self.running = False
                                break
                        else:
                            self.send('EMERGENCY', 'STEER_STOP')
                            self.running = False
                            break

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)

    def run(self):
        if not self.connect():
            return

        threading.Thread(target=self.receiver_loop, daemon=True).start()
        threading.Thread(target=self.sender_loop, daemon=True).start()
        threading.Thread(target=self.display_loop, daemon=True).start()

        time.sleep(0.5)
        self.read_keys()

        self.send('STOP', 'STEER_STOP')
        time.sleep(0.3)
        self.connected = False
        if self.sock:
            self.sock.close()
        # Restore terminal
        os.system('clear')
        print("Disconnected. Goodbye!")


if __name__ == "__main__":
    print("=" * 55)
    print("     VEHICLE REMOTE CONTROL")
    print("=" * 55)
    if not PI_IP:
        print("No IP provided")
    else:
        try:
            RemoteControl(PI_IP, PI_PORT).run()
        except Exception as e:
            print(f"Error: {e}")
