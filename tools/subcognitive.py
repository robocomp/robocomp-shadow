#!/usr/bin/env python3

import subprocess
import time
import os
from pathlib import Path
import Ice
import psutil
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich import box
import toml
import argparse

parser = argparse.ArgumentParser(description="RoboComp subcognitive monitor")
parser.add_argument("file_name", type=str, default="sub.toml", help="Path to TOML components file")
args = parser.parse_args()

# Load components from TOML file
def toml_loader():
    config_path = os.path.expanduser(args.file_name)
    if not os.path.exists(config_path):
        console.print(f"[red]Missing file with components at {config_path}[/red]")
        exit(1)

    components = toml.load(args.file_name)["components"]
    return components

def ping_proxy(ice_string):
    try:
        # Initialize Ice communicator if not already done
        with Ice.initialize() as communicator:
            proxy = communicator.stringToProxy(ice_string)
            # Narrow to Ice::Object (the base interface)
            obj = proxy.ice_ping()
            return True
    except Exception as e:
        return False

def green(text):
    return f"\033[92m{text}\033[0m"

def red(text):
    return f"\033[91m{text}\033[0m"

def format_uptime(seconds):
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

processes = {}

def expand_path(p):
    return os.path.expanduser(p) if p else None

# Start all components
def launch_process(command, cwd=None, name=None):
    stdout_path = os.path.expanduser(f"~/.local/logs/{name}.out") if name else os.devnull
    stderr_path = os.path.expanduser(f"~/.local/logs/{name}.err") if name else os.devnull
    os.makedirs(os.path.dirname(stdout_path), exist_ok=True)

    stdout = open(stdout_path, "w")
    stderr = open(stderr_path, "w")

    proc = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=stdout,
        stderr=stderr
    )
    return proc, psutil.Process(proc.pid)

components = toml_loader()
print(components)

for comp in components:
    cwd = expand_path(comp.get("cwd"))
    command = comp["cmd"]
    print(f"Starting {comp['name']}...")
    proc, ps_proc = launch_process(command, cwd=cwd, name=comp["name"])
    ps_proc.cpu_percent(interval=None)  # <<< Add this here
    processes[comp["name"]] = {
        "process": proc,
        "psutil_proc": ps_proc,
        "ice_name": comp.get("ice_name"),
        "start_time": time.time()
    }

#Monitor loop
console = Console()
def build_table():
    table = Table(title="🧠 RoboComp Component Monitor", box=box.SIMPLE_HEAVY)
    table.add_column("Name", style="bold cyan")
    table.add_column("Status", style="bold")
    table.add_column("Uptime", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("CPU", justify="right")

    for name, info in processes.items():
        ice_name = info["ice_name"]
        status = "[yellow]⏳ Checking...[/yellow]"

        # Ice ping
        try:
            if ice_name:
                ping_proxy(ice_name)
                status = "[green]✅ Alive[/green]"
            else:
                status = "[blue]⚠️ No ICE[/blue]"
        except Exception:
            status = "[red]❌ Down[/red]"

        # Uptime
        uptime = time.time() - info["start_time"]
        uptime_str = format_uptime(uptime)

        # Resource usage
        try:
            mem = info["psutil_proc"].memory_info().rss / (1024 ** 2)
            cpu = info["psutil_proc"].cpu_percent(interval=0.1)
        except Exception:
            mem, cpu = 0, 0

        table.add_row(
            name,
            status,
            uptime_str,
            f"{mem:6.1f} MB",
            f"{cpu:5.1f}%"
        )

    return table

try:
    with Live(build_table(), refresh_per_second=1, console=console, screen=False) as live:
        while True:
            time.sleep(1)
            live.update(build_table())
except KeyboardInterrupt:
    console.print("\n[yellow]Exiting. Terminating all processes...[/yellow]")
    for info in processes.values():
        info["process"].terminate()
    for info in processes.values():
        info["process"].wait()
