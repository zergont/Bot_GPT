#!/usr/bin/env python3
import argparse
import os
import sys
import signal
import subprocess
from pathlib import Path

def get_pid_file():
    return Path("bot.pid")

def save_pid(pid):
    with get_pid_file().open('w') as f:
        f.write(str(pid))

def get_saved_pid():
    try:
        with get_pid_file().open('r') as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return None

def is_process_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def start_bot():
    """������ ����"""
    pid = get_saved_pid()
    if pid and is_process_running(pid):
        print("��� ��� �������!")
        return

    proc = subprocess.Popen([sys.executable, "Bot_GPT.py"])
    save_pid(proc.pid)
    print(f"��� ������� (PID: {proc.pid})")

def stop_bot():
    """��������� ����"""
    pid = get_saved_pid()
    if not pid:
        print("PID ���� �� ������")
        return

    if not is_process_running(pid):
        print("��� �� �������")
        get_pid_file().unlink(missing_ok=True)
        return

    try:
        os.kill(pid, signal.SIGTERM)
        print("��� ����������")
    except OSError as e:
        print(f"������ ��� ��������� ����: {e}")
    finally:
        get_pid_file().unlink(missing_ok=True)

def status_bot():
    """�������� ������� ����"""
    pid = get_saved_pid()
    if not pid:
        print("��� �� �������")
        return

    if is_process_running(pid):
        print(f"��� ������� (PID: {pid})")
    else:
        print("��� �� ������� (PID ���� �������)")
        get_pid_file().unlink(missing_ok=True)

def update_bot():
    """���������� ���� �� GitHub"""
    try:
        subprocess.run(["git", "fetch", "origin"], check=True)
        subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("���������� ������� ���������")
        # ���������� ����, ���� �� ��� �������
        pid = get_saved_pid()
        if pid and is_process_running(pid):
            stop_bot()
            start_bot()
            print("��� �����������")
    except subprocess.CalledProcessError as e:
        print(f"������ ��� ����������: {e}")

def main():
    parser = argparse.ArgumentParser(description='���������� Telegram �����')
    parser.add_argument('command', choices=['start', 'stop', 'status', 'update'],
                       help='������� ��� ���������� �����')
    args = parser.parse_args()
    commands = {
        'start': start_bot,
        'stop': stop_bot,
        'status': status_bot,
        'update': update_bot
    }
    commands[args.command]()

if __name__ == "__main__":
    main()