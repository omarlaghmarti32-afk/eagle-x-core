"""Host metrics via psutil."""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

import psutil

from .config import MONITOR_INTERVAL


class HostMonitor:
    def __init__(self) -> None:
        self._prev_net = psutil.net_io_counters()
        self._prev_t = time.time()

    def snapshot(self) -> dict[str, float]:
        now = time.time()
        net = psutil.net_io_counters()
        dt = max(now - self._prev_t, 1e-3)
        sent_rate = (net.bytes_sent - self._prev_net.bytes_sent) / dt
        recv_rate = (net.bytes_recv - self._prev_net.bytes_recv) / dt
        self._prev_net, self._prev_t = net, now

        disk = psutil.disk_usage("/")
        return {
            "cpu_percent": float(psutil.cpu_percent(interval=0.05)),
            "mem_percent": float(psutil.virtual_memory().percent),
            "net_sent_rate": float(max(sent_rate, 0)),
            "net_recv_rate": float(max(recv_rate, 0)),
            "process_count": float(len(psutil.pids())),
            "connection_count": float(len(psutil.net_connections(kind="inet"))),
            "disk_percent": float(disk.percent),
        }

    async def stream(self, interval: float | None = None) -> AsyncIterator[dict[str, float]]:
        wait = interval if interval is not None else MONITOR_INTERVAL
        while True:
            yield await asyncio.to_thread(self.snapshot)
            await asyncio.sleep(wait)
