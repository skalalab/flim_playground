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
        
        # Count browser processes with our URL - most reliable method
        browser_processes_with_url = 0
        
        # Get platform info for better browser detection
        platform_info = get_platform_info()
        
        # Browser names for different platforms
        browser_names = [
            # Windows/Linux names
            'chrome', 'firefox', 'edge', 'msedge', 'safari', 'opera', 'brave',
            # macOS names
            'google chrome', 'microsoft edge', 'firefox', 'safari', 'opera', 'brave browser'
        ]
        
        # Add platform-specific browser names
        if platform_info['is_macos']:
            browser_names.extend([
                'chrome helper', 'safari web content', 'firefox web content',
                'microsoft edge helper', 'opera helper'
            ])
        
        url_patterns = [f'localhost:{port}', f'127.0.0.1:{port}']
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name']:
                    proc_name_lower = proc.info['name'].lower()
                    # Check if any browser name is contained in the process name
                    if any(browser in proc_name_lower for browser in browser_names):
                        if proc.info['cmdline']:
                            cmdline_str = ' '.join(proc.info['cmdline']).lower()
                            if any(pattern in cmdline_str for pattern in url_patterns):
                                browser_processes_with_url += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Optional: Count connections for status info
        connection_count = 0
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port and conn.status == 'ESTABLISHED' and conn.raddr:
                connection_count += 1
        
        # Simple logic: windows are open if we have browser processes with our URL
        windows_open = browser_processes_with_url > 0
        status = f"Browser processes: {browser_processes_with_url}, Connections: {connection_count}"
        
        return windows_open, status
        
    except ImportError:
        # Fallback without psutil
        return False, "psutil not available"


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
    
    print("Browser monitoring active - app will close 30 seconds after all browser windows are closed")
    
    # Monitor browser windows
    no_windows_count = 0
    max_no_windows_checks = 6  # 6 checks * 5 seconds = 30 seconds
    
    # Wait for initial browser connection
    time.sleep(5)
    
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
                print("All browser windows closed for 30+ seconds. Shutting down...")
                shutdown_event.set()
                aggressive_shutdown()
                
        except Exception as e:
            print(f"Monitor error: {e}")
            shutdown_event.set()
            aggressive_shutdown()
        
        time.sleep(5)


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