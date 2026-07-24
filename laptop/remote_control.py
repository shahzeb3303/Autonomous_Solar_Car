#!/usr/bin/env python3
"""
Laptop Remote Control
CLI interface with arrow keys to control vehicle remotely
"""

import socket
import json
import threading
import time
import sys
import curses
from datetime import datetime

# Configuration (should match Raspberry Pi config)
PI_IP = input("Enter Raspberry Pi IP address (e.g., 192.168.1.100): ").strip()
PI_PORT = 5555

# Commands
CMD_FORWARD = 'FORWARD'
CMD_BACKWARD = 'BACKWARD'
CMD_LEFT = 'LEFT'
CMD_RIGHT = 'RIGHT'
CMD_STEER_STOP = 'STEER_STOP'
CMD_STOP = 'STOP'
CMD_EMERGENCY = 'EMERGENCY'

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

class RemoteControl:
    """
    Remote control client for vehicle
    """

    def __init__(self, pi_ip, pi_port):
        """Initialize remote control"""
        self.pi_ip = pi_ip
        self.pi_port = pi_port
        self.sock = None
        self.connected = False
        self.running = True

        # Current state
        self.current_command = CMD_STOP
        self.current_steer = CMD_STEER_STOP
        self.status = None
        self.status_lock = threading.Lock()
        self.command_lock = threading.Lock()

        # Threads
        self.receive_thread = None
        self.send_thread = None

    def connect(self):
        """Connect to Raspberry Pi"""
        print(f"\nConnecting to Raspberry Pi at {self.pi_ip}:{self.pi_port}...")

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.pi_ip, self.pi_port))
            self.sock.settimeout(0.5)  # For receive operations
            self.connected = True
            print(f"{Colors.GREEN}✓ Connected!{Colors.RESET}\n")
            return True

        except Exception as e:
            print(f"{Colors.RED}✗ Connection failed: {e}{Colors.RESET}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from Raspberry Pi"""
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None

    def send_command(self, command):
        """Send command to Raspberry Pi"""
        if not self.connected:
            return False

        try:
            message = json.dumps({'command': command})
            self.sock.sendall(message.encode('utf-8'))
            with self.command_lock:
                self.current_command = command
            return True

        except Exception as e:
            # print(f"Error sending command: {e}")
            self.connected = False
            return False

    def continuous_send_loop(self):
        """Continuously send current command to keep watchdog timer happy"""
        while self.running and self.connected:
            try:
                with self.command_lock:
                    cmd = self.current_command
                    steer = self.current_steer

                # Send drive command
                if self.connected:
                    message = json.dumps({'command': cmd})
                    self.sock.sendall(message.encode('utf-8'))

                # Send steer command if steering
                if steer != CMD_STEER_STOP and self.connected:
                    time.sleep(0.05)
                    message = json.dumps({'command': steer})
                    self.sock.sendall(message.encode('utf-8'))

                time.sleep(0.5)  # Send every 500ms (watchdog is 2 seconds)

            except Exception as e:
                if self.running:
                    self.connected = False
                break

    def receive_status(self):
        """Receive status updates from Pi (runs in thread)"""
        buffer = ""

        while self.running and self.connected:
            try:
                data = self.sock.recv(4096)
                if not data:
                    # Connection closed
                    self.connected = False
                    break

                buffer += data.decode('utf-8')

                # Process complete JSON messages (newline delimited)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    try:
                        status = json.loads(line)
                        with self.status_lock:
                            self.status = status
                    except json.JSONDecodeError:
                        pass

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    # print(f"Error receiving status: {e}")
                    self.connected = False
                break

    def get_alert_color(self, alert_level):
        """Get color for alert level"""
        if alert_level == 'CLEAR':
            return Colors.GREEN
        elif alert_level == 'WARNING':
            return Colors.YELLOW
        elif alert_level == 'CRITICAL':
            return Colors.MAGENTA
        elif alert_level == 'EMERGENCY':
            return Colors.RED
        else:
            return Colors.RESET

    def display_in_curses(self, stdscr):
        """Display status using curses (called from keyboard_loop)"""
        try:
            stdscr.clear()
            row = 0

            # Header
            stdscr.addstr(row, 0, "╔" + "═" * 62 + "╗", curses.A_BOLD)
            row += 1
            stdscr.addstr(row, 0, "║" + " " * 16 + "VEHICLE REMOTE CONTROL" + " " * 24 + "║", curses.A_BOLD)
            row += 1
            stdscr.addstr(row, 0, "╠" + "═" * 62 + "╣", curses.A_BOLD)
            row += 1

            # Connection status
            if self.connected:
                conn_text = f"CONNECTED to {self.pi_ip}"
                stdscr.addstr(row, 0, "║  Connection: ")
                stdscr.addstr(conn_text, curses.A_BOLD | curses.COLOR_GREEN if curses.has_colors() else curses.A_BOLD)
                stdscr.addstr(" " * (47 - len(conn_text)) + "║")
            else:
                stdscr.addstr(row, 0, "║  Connection: DISCONNECTED" + " " * 37 + "║")
            row += 1
            stdscr.addstr(row, 0, "╠" + "═" * 62 + "╣", curses.A_BOLD)
            row += 1

            # Status from Pi
            if self.status:
                with self.status_lock:
                    cmd = self.status.get('current_command', 'UNKNOWN')
                    actual_speed = self.status.get('actual_speed', 0)
                    alert = self.status.get('alert_level', 'UNKNOWN')
                    distances = self.status.get('distances', {})
                    min_front = self.status.get('min_distance_front', 0)
                    min_back = self.status.get('min_distance_back', 0)

                steer = self.status.get('current_steer', 'STRAIGHT')

                # Command, steer and speed
                status_line = f"║  Drive: {cmd:8s}  Steer: {steer:10s}  Speed: {actual_speed:3d}%          ║"
                stdscr.addstr(row, 0, status_line)
                row += 1

                # Alert
                alert_line = f"║  Alert: {alert:9s}                                             ║"
                stdscr.addstr(row, 0, alert_line)
                row += 1

                stdscr.addstr(row, 0, "╠" + "═" * 62 + "╣", curses.A_BOLD)
                row += 1

                # Sensor distances
                stdscr.addstr(row, 0, "║  Sensor Distances:                                         ║")
                row += 1

                fl = distances.get('FL', 0)
                fw = distances.get('FW', 0)
                fr = distances.get('FR', 0)
                bc = distances.get('BC', 0)
                ls = distances.get('LS', 0)
                rs = distances.get('RS', 0)

                front_line = f"║    Front:  FL={fl:5.1f}cm  FW={fw:5.1f}cm  FR={fr:5.1f}cm  [MIN: {min_front:5.1f}cm] ║"
                stdscr.addstr(row, 0, front_line)
                row += 1

                back_line = f"║    Back:   BC={bc:5.1f}cm                      [MIN: {min_back:5.1f}cm] ║"
                stdscr.addstr(row, 0, back_line)
                row += 1

                sides_line = f"║    Sides:  LS={ls:5.1f}cm  RS={rs:5.1f}cm                             ║"
                stdscr.addstr(row, 0, sides_line)
                row += 1

            else:
                stdscr.addstr(row, 0, "║  Waiting for status data from Pi...                       ║")
                row += 1

            # Controls
            stdscr.addstr(row, 0, "╠" + "═" * 62 + "╣", curses.A_BOLD)
            row += 1
            stdscr.addstr(row, 0, "║  Controls:                                                 ║")
            row += 1
            stdscr.addstr(row, 0, "║    ↑ (Up Arrow)    : Move Forward                          ║")
            row += 1
            stdscr.addstr(row, 0, "║    ↓ (Down Arrow)  : Move Backward                         ║")
            row += 1
            stdscr.addstr(row, 0, "║    ← (Left Arrow)  : Steer Left (toggle)                  ║")
            row += 1
            stdscr.addstr(row, 0, "║    → (Right Arrow) : Steer Right (toggle)                  ║")
            row += 1
            stdscr.addstr(row, 0, "║    SPACE           : Stop All                              ║")
            row += 1
            stdscr.addstr(row, 0, "║    ESC             : Emergency Stop & Quit                 ║")
            row += 1
            stdscr.addstr(row, 0, "╚" + "═" * 62 + "╝", curses.A_BOLD)
            row += 2

            # Current command indicator
            cmd_display = self.current_command
            if cmd_display == CMD_FORWARD:
                cmd_display = "↑ FORWARD"
            elif cmd_display == CMD_BACKWARD:
                cmd_display = "↓ BACKWARD"
            elif cmd_display == CMD_STOP:
                cmd_display = "■ STOPPED"

            steer_display = self.current_steer
            if steer_display == CMD_LEFT:
                steer_display = "← LEFT"
            elif steer_display == CMD_RIGHT:
                steer_display = "→ RIGHT"
            else:
                steer_display = "STRAIGHT"

            stdscr.addstr(row, 0, f"Drive: {cmd_display}   Steer: {steer_display}")

            stdscr.refresh()

        except curses.error:
            # Ignore curses errors (usually from terminal size issues)
            pass

    def keyboard_loop(self, stdscr):
        """Handle keyboard input and display using curses"""
        # Configure curses
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)  # Non-blocking input
        stdscr.timeout(100)  # 100ms timeout

        last_display_update = 0
        display_interval = 0.1  # Update display 10 times per second

        while self.running and self.connected:
            current_time = time.time()

            # Update display if enough time has passed
            if current_time - last_display_update >= display_interval:
                self.display_in_curses(stdscr)
                last_display_update = current_time

            # Handle keyboard input
            try:
                key = stdscr.getch()

                if key == curses.KEY_UP:
                    with self.command_lock:
                        self.current_command = CMD_FORWARD
                elif key == curses.KEY_DOWN:
                    with self.command_lock:
                        self.current_command = CMD_BACKWARD
                elif key == curses.KEY_LEFT:
                    with self.command_lock:
                        # Toggle: press again to go straight
                        if self.current_steer == CMD_LEFT:
                            self.current_steer = CMD_STEER_STOP
                        else:
                            self.current_steer = CMD_LEFT
                elif key == curses.KEY_RIGHT:
                    with self.command_lock:
                        # Toggle: press again to go straight
                        if self.current_steer == CMD_RIGHT:
                            self.current_steer = CMD_STEER_STOP
                        else:
                            self.current_steer = CMD_RIGHT
                elif key == ord(' '):
                    with self.command_lock:
                        self.current_command = CMD_STOP
                        self.current_steer = CMD_STEER_STOP
                elif key == 27:  # ESC key
                    self.send_command(CMD_EMERGENCY)
                    time.sleep(0.2)
                    self.running = False
                    break

            except Exception as e:
                pass

            time.sleep(0.05)

    def run(self):
        """Main run loop"""
        # Connect to Pi
        if not self.connect():
            print("\nFailed to connect. Please check:")
            print("1. Raspberry Pi IP address is correct")
            print("2. Raspberry Pi is running main.py")
            print("3. Network connection is working")
            return

        # Start receive thread
        self.receive_thread = threading.Thread(target=self.receive_status, daemon=True)
        self.receive_thread.start()

        # Start continuous send thread (keeps watchdog timer happy)
        self.send_thread = threading.Thread(target=self.continuous_send_loop, daemon=True)
        self.send_thread.start()

        # Wait a moment for first status
        time.sleep(0.5)

        # Start curses interface (handles both keyboard and display)
        print("Starting remote control interface...")
        print("Use arrow keys to control. Starting in 2 seconds...\n")
        time.sleep(2)

        try:
            curses.wrapper(self.keyboard_loop)
        except KeyboardInterrupt:
            pass

        # Cleanup
        print("\nStopping vehicle...")
        self.send_command(CMD_STOP)
        time.sleep(0.2)
        self.disconnect()
        print("Disconnected.\n")


def main():
    """Main entry point"""
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║           VEHICLE REMOTE CONTROL - LAPTOP SIDE             ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")

    if not PI_IP:
        print("\nERROR: No IP address provided")
        return

    try:
        controller = RemoteControl(PI_IP, PI_PORT)
        controller.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")

    print("\nGoodbye!\n")


if __name__ == "__main__":
    main()
