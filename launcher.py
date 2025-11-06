#!/usr/bin/env python3
"""
Launcher script for Flim-Playground Streamlit application.
This script handles proper initialization and execution of the Streamlit app
when bundled with PyInstaller.
"""

import os
import sys
import time
import webbrowser
import socket
import threading
import platform

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
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
    
    # Set the main script path
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


def check_browser_windows_open(port):
    """Check if browser windows are open to our app"""
    try:
        import psutil
        
        # Browser process names
        browser_names = ['chrome', 'firefox', 'edge', 'msedge', 'safari', 'opera', 'brave']
        
        # Count browsers with active connections to our port
        connected_browsers = 0
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] and any(browser in proc.info['name'].lower() for browser in browser_names):
                    # Check if this browser has connections to our port
                    process = psutil.Process(proc.info['pid'])
                    for conn in process.net_connections(kind='inet'):
                        if (hasattr(conn, 'raddr') and conn.raddr and 
                            conn.raddr.port == port and conn.status == 'ESTABLISHED'):
                            connected_browsers += 1
                            print(f"Debug - Browser connected: {proc.info['name']} (PID: {proc.info['pid']})")
                            break  # Only count each browser process once
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # Return result
        windows_open = connected_browsers > 0
        status = f"Connected browsers: {connected_browsers}"
        
        if connected_browsers == 0:
            print(f"Debug - No browsers connected to port {port}")
        
        return windows_open, status
        
    except ImportError:
        # Fallback without psutil
        server_active = check_server_running(port)
        return server_active, "psutil not available"


def aggressive_shutdown():
    """Shut down the application"""
    print("Shutting down...")
    
    try:
        import psutil
        current_pid = os.getpid()
        
        # Terminate streamlit processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'streamlit' in proc.info['name'].lower():
                    if proc.info['pid'] != current_pid:
                        proc.terminate()
                        proc.wait(timeout=3)
                elif proc.info['cmdline'] and any('streamlit' in str(cmd).lower() for cmd in proc.info['cmdline']):
                    if proc.info['pid'] != current_pid:
                        proc.terminate()
                        proc.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue
    except ImportError:
        pass
    
    os._exit(0)


def monitor_browser_windows(port, shutdown_event):
    """Monitor browser windows and shutdown when all are closed"""
    print("Starting browser monitoring...")
    
    # Wait for server to start
    max_wait = 20
    waited = 0
    while waited < max_wait and not shutdown_event.is_set():
        if check_server_running(port):
            break
        time.sleep(0.5)
        waited += 0.5
    
    if waited >= max_wait:
        print("Server failed to start, disabling monitoring")
        return
    
    print("Browser monitoring active - app will close 10 seconds after all browser windows are closed")
    
    # Monitor browser windows
    no_windows_count = 0
    max_no_windows_checks = 2     # 2 checks * 5 seconds = 10 seconds
    
    # Wait for initial browser connection
    time.sleep(5)
    
    # Optimize: Increase check interval to reduce CPU usage
    check_interval = 10  # Check every 10 seconds instead of 5
    
    while not shutdown_event.is_set():
        try:
            # Check if browser windows are open
            windows_open, status = check_browser_windows_open(port)
            
            if not windows_open:
                no_windows_count += 1
                print(f"No browser windows detected ({no_windows_count}/{max_no_windows_checks}) - {status}")
            else:
                if no_windows_count > 0:
                    print(f"Browser windows detected - {status}")
                no_windows_count = 0
            
            # Shutdown if no windows for full duration
            if no_windows_count >= max_no_windows_checks:
                print("All browser windows closed for 10+ seconds. Shutting down...")
                shutdown_event.set()
                aggressive_shutdown()
                
        except Exception as e:
            print(f"Monitor error: {e}")
            shutdown_event.set()
            aggressive_shutdown()
        
        time.sleep(check_interval)


def run_streamlit_app(main_script):
    """Run the Streamlit application"""
    shutdown_event = threading.Event()
    
    try:
        # Find a free port
        port = find_free_port()
        
        print("Starting Flim-Playground...")
        print(f"Server will start on port {port}")
        print("The application will open in your default web browser.")
        print("App will auto-close 30 seconds after ALL browser windows are closed.")
        
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
        
        # Start a thread to open the browser after a delay
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
        # Setup environment
        main_script = setup_environment()
        
        # Run the Streamlit application
        run_streamlit_app(main_script)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        aggressive_shutdown()


if __name__ == '__main__':
    main()