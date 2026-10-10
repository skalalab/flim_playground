"""The desktop launcher follows app sessions, independent of browser processes."""

import os
import threading
from types import SimpleNamespace

import psutil
import pytest
from streamlit.runtime import Runtime, RuntimeState
from streamlit.runtime.websocket_session_manager import SESSION_EVENTS_FAMILY

import launcher


@pytest.fixture
def runtime(monkeypatch):
    counters = {"connect": 1, "reconnect": 0, "disconnect": 0}

    def get_stats(family_names):
        return {SESSION_EVENTS_FAMILY: [
            SimpleNamespace(value=value, labels={"type": event})
            for event, value in counters.items()
        ]}

    instance = SimpleNamespace(
        state=RuntimeState.ONE_OR_MORE_SESSIONS_CONNECTED,
        stats_mgr=SimpleNamespace(get_stats=get_stats),
        connection_counters=counters,
    )
    monkeypatch.setattr(Runtime, "exists", classmethod(lambda cls: True))
    monkeypatch.setattr(Runtime, "instance", classmethod(lambda cls: instance))
    monkeypatch.setattr(psutil, "process_iter", lambda *args: [])
    return instance


def test_safari_connection_owned_by_webkit_keeps_app_open(runtime, monkeypatch):
    """Safari's socket owner need not have 'safari' in its process name."""
    port = 8501
    safari = SimpleNamespace(info={"pid": 100, "name": "Safari"})
    webkit = SimpleNamespace(info={"pid": 101, "name": "com.apple.WebKit.Networking"})
    connection = SimpleNamespace(
        raddr=SimpleNamespace(port=port), status="ESTABLISHED",
    )
    processes = {
        100: SimpleNamespace(net_connections=lambda **kwargs: []),
        101: SimpleNamespace(net_connections=lambda **kwargs: [connection]),
    }
    monkeypatch.setattr(psutil, "process_iter", lambda *args: [safari, webkit])
    monkeypatch.setattr(psutil, "Process", processes.__getitem__)

    connected, _ = launcher.get_browser_session_activity()

    assert connected is True, "an active Safari app tab must not trigger shutdown"


def test_browser_socket_permission_denied_keeps_active_app_open(runtime, monkeypatch):
    safari = SimpleNamespace(info={"pid": 100, "name": "Safari"})
    monkeypatch.setattr(psutil, "process_iter", lambda *args: [safari])

    def inaccessible_process(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", inaccessible_process)

    connected, _ = launcher.get_browser_session_activity()

    assert connected is True


@pytest.mark.parametrize("state, expected", [
    (RuntimeState.ONE_OR_MORE_SESSIONS_CONNECTED, True),
    (RuntimeState.NO_SESSIONS_CONNECTED, False),
    (RuntimeState.INITIAL, None),
    (RuntimeState.STOPPING, None),
    (RuntimeState.STOPPED, None),
])
def test_browser_connection_follows_streamlit_runtime(runtime, state, expected):
    runtime.state = state

    connected, _ = launcher.get_browser_session_activity()

    assert connected is expected


def test_uninitialized_runtime_is_not_a_closed_browser(runtime, monkeypatch):
    monkeypatch.setattr(Runtime, "exists", classmethod(lambda cls: False))

    connected, _ = launcher.get_browser_session_activity()

    assert connected is None


def test_connection_history_includes_new_and_reconnected_sessions(runtime):
    runtime.state = RuntimeState.NO_SESSIONS_CONNECTED
    runtime.connection_counters.update(connect=2, reconnect=3, disconnect=5)

    connected, connection_count = launcher.get_browser_session_activity()

    assert connected is False
    assert connection_count == 5


def test_unavailable_session_history_cannot_trigger_shutdown(runtime):
    runtime.stats_mgr.get_stats = lambda family_names: {}

    connected, connection_count = launcher.get_browser_session_activity()

    assert connected is None
    assert connection_count is None


@pytest.fixture
def run_monitor(monkeypatch):
    """Run the real monitor with a virtual clock and browser connection timeline."""
    def run(observations, *, stop_after=90, server_ready_at=0):
        clock = SimpleNamespace(now=0)
        shutdown_event = threading.Event()
        shutdown_times = []
        connection_checks = []

        def advance(seconds):
            clock.now += seconds
            if clock.now >= stop_after:
                shutdown_event.set()

        def wait(seconds):
            advance(seconds)
            return shutdown_event.is_set()

        def check_connection():
            connection_checks.append(clock.now)
            current = observations[0][1]
            previous = None
            connection_count = 0
            for timestamp, observation in observations:
                if timestamp <= clock.now:
                    current = observation
                    if observation is True and previous is not True:
                        connection_count += 1
                    previous = observation
            if isinstance(current, Exception):
                raise current
            return current, connection_count

        monkeypatch.setattr(launcher, "time", SimpleNamespace(
            monotonic=lambda: clock.now, sleep=advance,
        ))
        monkeypatch.setattr(shutdown_event, "wait", wait)
        monkeypatch.setattr(launcher, "check_server_running",
                            lambda port: clock.now >= server_ready_at)
        monkeypatch.setattr(launcher, "get_browser_session_activity", check_connection)
        monkeypatch.setattr(launcher, "aggressive_shutdown",
                            lambda: shutdown_times.append(clock.now))

        launcher.monitor_browser_windows(8501, shutdown_event)
        return shutdown_times, clock.now, connection_checks

    return run


def test_slow_initial_browser_open_does_not_shut_down(run_monitor):
    shutdowns, _, checks = run_monitor([(0, False), (45, True)])

    assert not shutdowns
    assert any(timestamp >= 45 for timestamp in checks)


def test_last_app_tab_closed_shuts_down_after_30_seconds(run_monitor):
    shutdowns, _, _ = run_monitor([(0, True), (5, False)])

    assert shutdowns == [35]


def test_browser_reconnections_reset_shutdown_grace_period(run_monitor):
    shutdowns, _, checks = run_monitor([
        (0, True), (5, False), (25, True), (30, False), (55, True),
    ])

    assert not shutdowns
    assert any(timestamp >= 55 for timestamp in checks)


def test_initial_connection_between_polls_still_arms_shutdown(run_monitor):
    shutdowns, _, _ = run_monitor([(0, False), (1, True), (4, False)])

    assert shutdowns == [35]


def test_reconnection_between_polls_resets_shutdown_grace_period(run_monitor):
    shutdowns, _, _ = run_monitor([
        (0, True), (5, False), (26, True), (29, False),
    ])

    assert shutdowns == [60]


def test_unknown_connection_state_cannot_trigger_shutdown(run_monitor):
    shutdowns, _, _ = run_monitor([(0, True), (5, None)])

    assert not shutdowns


def test_unknown_connection_state_resets_shutdown_grace_period(run_monitor):
    shutdowns, _, _ = run_monitor([
        (0, True), (5, False), (25, None), (30, False),
    ])

    assert shutdowns == [60]


def test_monitor_error_does_not_shut_down_app(run_monitor):
    shutdowns, _, _ = run_monitor([(0, True), (5, RuntimeError("unavailable"))])

    assert not shutdowns


def test_shutdown_leaves_other_streamlit_processes_alone(monkeypatch):
    """A dev server or another Streamlit app outlives this app's exit."""
    terminated, exits = [], []
    dev_server = SimpleNamespace(
        info={"pid": 4242, "name": "streamlit", "cmdline": ["streamlit", "run", "app.py"]},
        terminate=lambda: terminated.append(4242),
        wait=lambda timeout=None: None,
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *args: [dev_server])
    monkeypatch.setattr(os, "_exit", exits.append)

    launcher.aggressive_shutdown()

    assert exits == [0]
    assert not terminated


def test_cancelled_server_start_stops_monitor_immediately(run_monitor):
    shutdowns, elapsed, checks = run_monitor(
        [(0, False)], stop_after=1, server_ready_at=10,
    )

    assert not shutdowns
    assert elapsed == 1
    assert not checks
