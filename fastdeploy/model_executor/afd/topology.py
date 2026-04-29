from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Callable, Dict, Optional

import requests
from paddleformers.utils.log import logger

from fastdeploy.inter_communicator.fmq import FMQ


def afd_manifest_queue_name(engine_pid: int | str) -> str:
    return f"afd_manifest_w2e_{engine_pid}"


def afd_topology_request_queue_name(engine_pid: int | str) -> str:
    return f"afd_topology_req_w2e_{engine_pid}"


def afd_topology_update_queue_name(engine_pid: int | str, rank: int | str) -> str:
    return f"afd_topology_e2w_{engine_pid}_rank{rank}"


def send_afd_expert_manifest_to_engine(manifest: Dict[str, Any], engine_pid: int | str) -> None:
    """Send one rank-level AFD expert manifest from worker to engine."""
    queue = FMQ().queue(afd_manifest_queue_name(engine_pid), "producer")
    asyncio.run(queue.put(manifest))


class AFDTopologyClient:
    """Router-backed topology watcher used by ATTN engines.

    The router owns etcd interaction. Engine processes use this client and
    forward snapshots to workers over local IPC.
    """

    def __init__(
        self,
        router_url: str,
        afd_layout=None,
        on_snapshot: Optional[Callable[[Dict[str, Any]], None]] = None,
        request_timeout: float = 30.0,
        reconnect_sleep: float = 2.0,
    ) -> None:
        self.router_url = router_url.rstrip("/")
        self.afd_layout = afd_layout
        self.on_snapshot = on_snapshot
        self.request_timeout = request_timeout
        self.reconnect_sleep = reconnect_sleep
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._revision = 0

    def start(self, block_until_ready: bool = False) -> None:
        if self._thread is not None:
            return
        if block_until_ready:
            self.wait_until_ready()
        self._thread = threading.Thread(target=self._run, name="afd-topology-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def wait_until_ready(self) -> None:
        while not self._stop.is_set():
            try:
                snapshot = self._fetch_ready_snapshot()
                if snapshot is not None:
                    self._apply_snapshot(snapshot)
                    return
            except Exception as exc:
                logger.warning(f"AFD topology initial fetch failed, will retry: {exc}")
            time.sleep(self.reconnect_sleep)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self._revision > 0:
                    self._watch(self._revision)
                else:
                    snapshot = self._fetch_ready_snapshot()
                    if snapshot is not None:
                        self._apply_snapshot(snapshot)
                    else:
                        time.sleep(self.reconnect_sleep)
            except Exception as exc:
                logger.warning(f"AFD topology watcher failed, will reconnect: {exc}")
                time.sleep(self.reconnect_sleep)

    def _fetch_ready_snapshot(self) -> Optional[Dict[str, Any]]:
        resp = requests.get(
            f"{self.router_url}/afd/topology",
            params={"wait_ready": "true", "timeout_secs": int(self.request_timeout)},
            timeout=self.request_timeout + 5,
        )
        if resp.status_code == 204:
            logger.info("AFD topology is not ready yet.")
            return None
        resp.raise_for_status()
        return resp.json()

    def _watch(self, revision: int) -> None:
        params = {"revision": int(revision)}
        with requests.get(
            f"{self.router_url}/afd/topology/watch",
            params=params,
            stream=True,
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            event_data = []
            for raw_line in resp.iter_lines(decode_unicode=True):
                if self._stop.is_set():
                    return
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    if event_data:
                        self._handle_sse_data("\n".join(event_data))
                        event_data = []
                    continue
                if line.startswith("data:"):
                    event_data.append(line[len("data:") :].strip())

    def _handle_sse_data(self, data: str) -> None:
        if not data:
            return
        snapshot = json.loads(data)
        self._apply_snapshot(snapshot)

    def _apply_snapshot(self, snapshot: Dict[str, Any]) -> None:
        revision = int(snapshot.get("revision", 0))
        if revision <= self._revision:
            return
        state = snapshot.get("state", "unknown")
        if state not in ("ready", "degraded"):
            logger.warning(
                "AFD topology update is not ready; keep previous routable layout. "
                f"state={state}, revision={revision}, missing_ffn_ranks={snapshot.get('missing_ffn_ranks')}"
            )
            self._revision = revision
            return

        if self.afd_layout is not None:
            self.afd_layout.update_from_topology_snapshot(snapshot)
        if self.on_snapshot is not None:
            self.on_snapshot(snapshot)
        self._revision = revision
        logger.info(
            "AFD topology applied from router: "
            f"state={state}, revision={revision}, layers={len(snapshot.get('expert_layout_by_layer', []))}, "
            f"missing_ffn_ranks={snapshot.get('missing_ffn_ranks')}"
        )


class AFDTopologyEngineService:
    """Engine-owned bridge between router topology and local worker IPC."""

    def __init__(
        self,
        router_url: str,
        engine_pid: int | str,
        request_timeout: float = 30.0,
        reconnect_sleep: float = 2.0,
    ) -> None:
        self.engine_pid = engine_pid
        self._latest_snapshot: Optional[Dict[str, Any]] = None
        self._snapshot_lock = threading.Lock()
        self._update_queues: Dict[str, Any] = {}
        self._request_queue = FMQ().queue(afd_topology_request_queue_name(engine_pid), "consumer")
        self._router_client = AFDTopologyClient(
            router_url,
            on_snapshot=self._publish_snapshot,
            request_timeout=request_timeout,
            reconnect_sleep=reconnect_sleep,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._request_loop, name="afd-topology-request-loop", daemon=True).start()
        self._router_client.start()

    def stop(self) -> None:
        self._router_client.stop()

    def _publish_snapshot(self, snapshot: Dict[str, Any]) -> None:
        with self._snapshot_lock:
            self._latest_snapshot = snapshot
            queues = list(self._update_queues.values())
        for queue in queues:
            asyncio.run(queue.put(snapshot))

    def _current_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._snapshot_lock:
            return self._latest_snapshot

    def _request_loop(self) -> None:
        while True:
            try:
                msg = asyncio.run(self._request_queue.get())
                if msg is None:
                    continue
                rank = str((msg.payload or {}).get("rank", "0"))
                snapshot = self._current_snapshot()
                while snapshot is None:
                    time.sleep(0.2)
                    snapshot = self._current_snapshot()
                with self._snapshot_lock:
                    queue = self._update_queues.get(rank)
                    if queue is None:
                        queue = FMQ().queue(afd_topology_update_queue_name(self.engine_pid, rank), "producer")
                        self._update_queues[rank] = queue
                asyncio.run(queue.put(snapshot))
            except Exception as exc:
                logger.warning(f"AFD topology engine local service failed, will continue: {exc}")
                time.sleep(1)


class AFDTopologyWorkerClient:
    """Worker-side local topology consumer. It talks only to the engine FMQ."""

    def __init__(
        self,
        engine_pid: int | str,
        rank: int | str,
        afd_layout,
        request_timeout_ms: int = 2000,
        reconnect_sleep: float = 2.0,
    ) -> None:
        self.engine_pid = engine_pid
        self.rank = str(rank)
        self.afd_layout = afd_layout
        self.request_timeout_ms = request_timeout_ms
        self.reconnect_sleep = reconnect_sleep
        self._request_queue = FMQ().queue(afd_topology_request_queue_name(engine_pid), "producer")
        self._update_queue = FMQ().queue(afd_topology_update_queue_name(engine_pid, self.rank), "consumer")
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._revision = 0

    def start(self, block_until_ready: bool = False) -> None:
        if self._thread is not None:
            return
        if block_until_ready:
            self.wait_until_ready()
        self._thread = threading.Thread(target=self._run, name="afd-topology-local-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def wait_until_ready(self) -> None:
        while not self._stop.is_set():
            try:
                asyncio.run(self._request_queue.put({"rank": self.rank, "revision": self._revision}))
                msg = asyncio.run(self._update_queue.get(timeout=self.request_timeout_ms))
                if msg is not None and self._apply_snapshot(msg.payload, require_ready=True):
                    return
            except Exception as exc:
                logger.warning(f"AFD topology initial local fetch failed, will retry: {exc}")
            time.sleep(self.reconnect_sleep)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                msg = asyncio.run(self._update_queue.get(timeout=self.request_timeout_ms))
                if msg is None:
                    continue
                self._apply_snapshot(msg.payload, require_ready=False)
            except Exception as exc:
                logger.warning(f"AFD topology local watcher failed, will continue: {exc}")
                time.sleep(self.reconnect_sleep)

    def _apply_snapshot(self, snapshot: Dict[str, Any], require_ready: bool) -> bool:
        revision = int(snapshot.get("revision", 0))
        if revision <= self._revision:
            return False
        state = snapshot.get("state", "unknown")
        if require_ready and state != "ready":
            logger.warning(
                "AFD topology local snapshot is not ready; keep waiting. "
                f"state={state}, revision={revision}, missing_ffn_ranks={snapshot.get('missing_ffn_ranks')}"
            )
            self._revision = revision
            return False
        if state not in ("ready", "degraded"):
            logger.warning(
                "AFD topology local update is not routable; keep previous layout. "
                f"state={state}, revision={revision}, missing_ffn_ranks={snapshot.get('missing_ffn_ranks')}"
            )
            self._revision = revision
            return False

        self.afd_layout.update_from_topology_snapshot(snapshot)
        self._revision = revision
        logger.info(
            "AFD topology applied from engine: "
            f"state={state}, revision={revision}, layers={len(snapshot.get('expert_layout_by_layer', []))}, "
            f"missing_ffn_ranks={snapshot.get('missing_ffn_ranks')}"
        )
        return state == "ready"
