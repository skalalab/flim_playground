#!/usr/bin/env python3
"""
Launcher script for FLIM Playground - opens Streamlit app in browser
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading
import signal
import socket
import warnings
from pathlib import Path

# Suppress specific warnings that occur in PyInstaller bundles
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
warnings.filterwarnings("ignore", message=".*event loop.*")
warnings.filterwarnings("ignore", message=".*protobuf.*")

# Redirect stderr to suppress asyncio errors that don't affect functionality
class ErrorFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.suppressed_errors = [
            "RuntimeError: no running event loop",
            "TypeError: 'str' object cannot be interpreted as an integer",
            "Exception in callback AppSession._on_scriptrunner_event",
            "asyncio/events.py"
        ]
    
    def write(self, data):
        # Check if this is one of the errors we want to suppress
        if any(error in str(data) for error in self.suppressed_errors):
            return  # Suppress this error
        self.original_stderr.write(data)
    
    def flush(self):
        self.original_stderr.flush()

def resource_path(rel: str) -> Path:
    """Return the absolute path to a bundled resource."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    return base / rel

def find_free_port():
    """Find a free port to run Streamlit on"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def wait_for_server(port, timeout=30):
    """Wait for the Streamlit server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', port))
                if result == 0:
                    return True
        except Exception as e:
            print(f"Error checking server: {e}")
        time.sleep(0.5)
    return False

def open_browser_after_delay(port, delay=0):
    """Open browser immediately and check if server is ready"""
    if delay > 0:
        print(f"Waiting {delay} seconds for server to be ready...")
        time.sleep(delay)
    
    if wait_for_server(port, timeout=10):
        print("Server ready, opening browser...")
        url = f"http://localhost:{port}"
        try:
            webbrowser.open(url)
            print(f"FLIM Playground opened at {url}")
            print("Close this window or press Ctrl+C to stop the application.")
        except Exception as e:
            print(f"ERROR: Failed to open browser: {e}")
            print(f"Please manually open: {url}")
    else:
        print("Failed to start server within timeout period")

def run_streamlit_bundled(main_py_path, port):
    """Run Streamlit in bundled mode using bootstrap with error filtering"""
    try:
        # Activate error filtering for PyInstaller-specific issues
        original_stderr = sys.stderr
        sys.stderr = ErrorFilter(original_stderr)
        
        # Change to the bundle directory so relative imports work
        if hasattr(sys, '_MEIPASS'):
            original_cwd = os.getcwd()
            os.chdir(sys._MEIPASS)
        
        # Set environment variables for protobuf compatibility
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        os.environ['PROTOBUF_PYTHON_IMPLEMENTATION'] = 'python'
        
        # Set up asyncio event loop to avoid issues
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                asyncio.set_event_loop(asyncio.new_event_loop())
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        
        # Configure streamlit
        from streamlit import config
        config.set_option('server.port', port)
        config.set_option('server.headless', True)
        config.set_option('browser.gatherUsageStats', False)
        config.set_option('server.address', 'localhost')
        config.set_option('global.developmentMode', False)
        config.set_option('server.enableCORS', True)
        config.set_option('server.enableXsrfProtection', False)
        config.set_option('server.fileWatcherType', 'none')
        config.set_option('global.suppressDeprecationWarnings', True)
        
        print(f"Starting Streamlit server on port {port}...")
        
        # Start browser opening in background thread
        browser_thread = threading.Thread(
            target=open_browser_after_delay, 
            args=(port, 2),  # Give server more time to start
            daemon=True
        )
        browser_thread.start()
        
        # Use bootstrap approach with proper sys.argv setup
        import streamlit.web.bootstrap as bootstrap
        
        # Set up sys.argv for streamlit
        original_argv = sys.argv.copy()
        sys.argv = ['streamlit', 'run', str(main_py_path)]
        
        try:
            # Run streamlit with bootstrap (respects port config better)
            bootstrap.run(str(main_py_path), '', [], {})
        finally:
            sys.argv = original_argv
            sys.stderr = original_stderr  # Restore original stderr
        
    except Exception as e:
        # Restore stderr before printing error
        if 'original_stderr' in locals():
            sys.stderr = original_stderr
        print(f"Error running Streamlit: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=== FLIM Playground Launcher ===")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if we're in a PyInstaller bundle
    is_bundled = hasattr(sys, '_MEIPASS')
    if is_bundled:
        print(f"Running from PyInstaller bundle: {sys._MEIPASS}")
    else:
        print("Running in development mode")
    
    # Find a free port
    port = find_free_port()
    print(f"Found free port: {port}")
    
    # Get the path to main.py
    main_py_path = resource_path("main.py")
    print(f"Main.py path: {main_py_path}")
    print(f"Main.py exists: {main_py_path.exists()}")
    
    if not main_py_path.exists():
        print("ERROR: main.py not found!")
        input("Press Enter to exit...")
        return 1
    
    # Check if streamlit is available
    try:
        import streamlit
        print(f"Streamlit version: {streamlit.__version__}")
        print(f"Streamlit location: {streamlit.__file__}")
    except ImportError as e:
        print(f"ERROR: Failed to import streamlit: {e}")
        input("Press Enter to exit...")
        return 1
    
    if is_bundled:
        # In PyInstaller bundle, run Streamlit in main thread
        print("Running Streamlit in bundled mode...")
        run_streamlit_bundled(main_py_path, port)
    else:
        # In development mode, use subprocess as before
        streamlit_cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(main_py_path),
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.address", "localhost"
        ]
        
        print(f"Streamlit command: {' '.join(streamlit_cmd)}")
        print(f"Starting FLIM Playground on port {port}...")
        
        try:
            process = subprocess.Popen(
                streamlit_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            print(f"Process started with PID: {process.pid}")
        except Exception as e:
            print(f"ERROR: Failed to start process: {e}")
            input("Press Enter to exit...")
            return 1
        
        # Wait for server to be ready
        print("Waiting for server to be ready...")
        if wait_for_server(port):
            print("Server ready, opening browser...")
            url = f"http://localhost:{port}"
            try:
                webbrowser.open(url)
                print(f"FLIM Playground opened at {url}")
                print("Close this window or press Ctrl+C to stop the application.")
            except Exception as e:
                print(f"ERROR: Failed to open browser: {e}")
                print(f"Please manually open: {url}")
        else:
            print("Failed to start server within timeout period")
            print("Server output:")
            stdout, stderr = process.communicate(timeout=5)
            if stdout:
                print("STDOUT:", stdout.decode())
            if stderr:
                print("STDERR:", stderr.decode())
            process.terminate()
            input("Press Enter to exit...")
            return 1
        
        # Handle cleanup on exit
        def signal_handler(sig, frame):
            print("\nShutting down FLIM Playground...")
            process.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for subprocess
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nShutting down FLIM Playground...")
            process.terminate()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        if exit_code != 0:
            input("Press Enter to exit...")
        sys.exit(exit_code)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1) 