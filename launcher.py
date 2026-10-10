#!/usr/bin/env python3
"""
Launcher script for Flim-Playground Streamlit application.
This script handles proper initialization and execution of the Streamlit app
when bundled with PyInstaller.
"""

import os
import secrets
import sys
import time
import webbrowser
import socket
import threading
import platform

BROWSER_POLL_INTERVAL = 5
BROWSER_IDLE_TIMEOUT = 30

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller exposes bundled resources through _MEIPASS.
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def setup_environment():
    """Setup environment variables for Streamlit"""
    # Disable Streamlit's usage statistics and telemetry
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_GATHER_USAGE_STATS'] = 'false'

    # Disable file watchers that can cause issues in bundled apps
    os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

    # Pages this launcher serves link Quit with this token; no other page or launch knows it.
    os.environ['FLIM_PLAYGROUND_QUIT_TOKEN'] = secrets.token_urlsafe(8)

    main_script = resource_path('main.py')
    if not os.path.exists(main_script):
        print(f"Error: main.py not found at {main_script}")
        sys.exit(1)

    return main_script


def get_platform_info():
    """Get platform information for cross-platform compatibility"""
    system = platform.system()
    return {
        'system': system,
        'is_macos': system == 'Darwin',
        'is_windows': system == 'Windows',
        'is_linux': system == 'Linux'
    }


def find_free_port():
    """Find a free port for the Streamlit server"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def check_server_running(port):
    """Check if server is running on the given port"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result == 0
    except Exception:
        return False


def get_browser_session_activity():
    """Return app connection state and cumulative connect/reconnect count.

    Safari owns its sockets in a separate WebKit networking process, and
    macOS may deny inspection of browser processes. Streamlit already knows
    whether an app tab is connected. Unknown activity returns (None, None).
    """
    from streamlit.runtime import Runtime, RuntimeState
    from streamlit.runtime.websocket_session_manager import SESSION_EVENTS_FAMILY

    if not Runtime.exists():
        return None, None

    runtime = Runtime.instance()
    state = runtime.state
    if state not in (RuntimeState.ONE_OR_MORE_SESSIONS_CONNECTED,
                     RuntimeState.NO_SESSIONS_CONNECTED):
        return None, None

    events = runtime.stats_mgr.get_stats([SESSION_EVENTS_FAMILY]).get(SESSION_EVENTS_FAMILY)
    if events is None:
        return None, None
    connection_count = sum(
        event.value for event in events
        if event.labels.get("type") in ("connect", "reconnect")
    )
    return state == RuntimeState.ONE_OR_MORE_SESSIONS_CONNECTED, connection_count


def aggressive_shutdown():
    """Exit the app at once, leaving every other process alone."""
    print("Shutting down...")
    # Worker processes exit by themselves once this process is gone.
    os._exit(0)


def monitor_browser_windows(port, shutdown_event):
    """Shut down after the last app session disconnects for the grace period."""
    print("Starting browser monitoring...")

    # Wait for server to start
    max_wait = 20
    waited = 0
    while waited < max_wait and not shutdown_event.is_set():
        if check_server_running(port):
            break
        if shutdown_event.wait(0.5):
            return
        waited += 0.5

    if shutdown_event.is_set():
        return

    if waited >= max_wait:
        print("Server failed to start, disabling monitoring")
        return

    print(f"Browser monitoring active - app will close {BROWSER_IDLE_TIMEOUT} seconds after all app tabs are closed")

    # A slow browser launch is not a closed tab. Begin the idle timer only
    # after a real session has connected, and allow brief reconnections.
    has_connected = False
    disconnected_since = None
    previous_connection_count = 0

    while not shutdown_event.is_set():
        try:
            sessions_connected, connection_count = get_browser_session_activity()
        except Exception as e:
            # Failed detection is not evidence that the user closed the app.
            print(f"Monitor error: {e}. Disabling automatic shutdown.")
            return

        connection_changed = False
        if sessions_connected is not None:
            # Connection counters preserve activity that starts and ends between
            # polls, including an initial tab that the user closes immediately.
            connection_changed = connection_count > previous_connection_count
            if connection_changed:
                has_connected = True
                disconnected_since = None
            previous_connection_count = connection_count

        if sessions_connected is True:
            if connection_changed or not has_connected or disconnected_since is not None:
                print(f"Browser session active on port {port}")
            has_connected = True
            disconnected_since = None
        elif sessions_connected is False and has_connected:
            now = time.monotonic()
            if disconnected_since is None:
                disconnected_since = now
                print(f"No app tabs connected to port {port}")
            if now - disconnected_since >= BROWSER_IDLE_TIMEOUT:
                print(f"All app tabs disconnected for {BROWSER_IDLE_TIMEOUT} seconds. Shutting down...")
                shutdown_event.set()
                aggressive_shutdown()
                return
        else:
            # An uninitialized or stopping runtime has no reliable session state.
            disconnected_since = None

        if shutdown_event.wait(BROWSER_POLL_INTERVAL):
            return


def run_streamlit_app(main_script):
    """Run the Streamlit application"""
    shutdown_event = threading.Event()

    try:
        port = find_free_port()

        print("Starting Flim-Playground...")
        print(f"Server will start on port {port}")
        print("The application will open in your default web browser.")
        print(f"App will auto-close {BROWSER_IDLE_TIMEOUT} seconds after all app tabs are closed.")

        # Import streamlit and set up arguments
        from streamlit.web import cli as stcli

        # Set up the arguments for streamlit with proper bundled app config
        sys.argv = [
            "streamlit",
            "run",
            main_script,
            "--server.port",
            str(port),
            "--server.address",
            "localhost",
            "--browser.gatherUsageStats",
            "false",
            "--global.developmentMode",
            "false",
            "--server.fileWatcherType",
            "none",
            "--server.headless",
            "true"
        ]

        # Open the browser after server startup, with a bounded wait.
        def open_browser():
            print("Waiting for server to start...")

            # Wait for server to actually be ready
            max_wait = 15  # Maximum 15 seconds
            waited = 0
            while waited < max_wait:
                if check_server_running(port):
                    break
                time.sleep(0.5)  # Check every half second
                waited += 0.5

            url = f"http://localhost:{port}"
            print(f"Opening browser to: {url}")
            webbrowser.open(url)

        # Start browser opening thread
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()

        # Start browser window monitoring
        monitor_thread = threading.Thread(
            target=monitor_browser_windows,
            args=(port, shutdown_event),
            daemon=True
        )
        monitor_thread.start()

        # Run streamlit
        print("Starting Streamlit server...")
        try:
            stcli.main()
        finally:
            # Cleanup on exit
            shutdown_event.set()
            print("Server stopped, initiating cleanup...")
            aggressive_shutdown()

    except KeyboardInterrupt:
        print("\nKeyboard interrupt - shutting down...")
        shutdown_event.set()
        aggressive_shutdown()
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        shutdown_event.set()
        aggressive_shutdown()


def main():
    """Main function to launch the Streamlit app"""
    platform_info = get_platform_info()

    print("="*60)
    print("Flim-Playground Launcher")
    print(f"Platform: {platform_info['system']}")
    print("="*60)

    try:
        main_script = setup_environment()

        # Run the Streamlit application
        run_streamlit_app(main_script)

    except Exception as e:
        print(f"Fatal error: {e}")
        aggressive_shutdown()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
