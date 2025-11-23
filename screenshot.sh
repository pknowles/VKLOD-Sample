#!/bin/bash

# Script to launch an app in xvfb and take a screenshot

set -e

# Helper function to print timestamped messages
log_with_timestamp() {
    echo "[$(date +%H:%M:%S)] $*"
}

# Configuration
BUILD_DIRS=("build" "build_release" "_bin" "_bin/Debug" "_bin/Release")
APP_PATH=""  # Will be auto-detected if not provided
SCREENSHOT_PNG="screenshot.png"
FB_WIDTH=800
FB_HEIGHT=600
TIMEOUT=10  # Default timeout for launching app (seconds)
SCREENSHOT_DELAY=5  # Default delay before taking screenshot (seconds)
APP_ARGS=()  # Array to store arguments to forward to the app

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --screenshot-delay)
            SCREENSHOT_DELAY="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [APP_PATH] [OPTIONS] [-- APP_ARGS...]"
            echo ""
            echo "Options:"
            echo "  --screenshot-delay SECONDS  Delay before taking screenshot (default: 5)"
            echo "  --timeout SECONDS          Timeout for app launch (default: 10)"
            echo "  --help, -h                  Show this help message"
            echo ""
            echo "Arguments after '--' will be forwarded to the application"
            echo "If APP_PATH is not provided, will search for binary in _bin/Debug/ or _bin/Release/"
            exit 0
            ;;
        --)
            shift
            APP_ARGS=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [ -z "$APP_PATH" ]; then
                APP_PATH="$1"
            else
                echo "Error: Multiple app paths specified: $APP_PATH and $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Auto-detect APP_PATH if not provided
if [ -z "$APP_PATH" ]; then
    APP_PATH="$(ls -t $(find "${BUILD_DIRS[@]}" -maxdepth 1 -type f -executable 2>/dev/null) | head -1)"
fi

# Variables to track processes for cleanup
CLEANUP_DONE=false
GDB_SCRIPT=$(mktemp)

# Cleanup function
cleanup() {
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    CLEANUP_DONE=true
    
    # Kill any child processes (jobs started by this shell)
    # This will kill xvfb-run and its children
    local job_pids=$(jobs -p 2>/dev/null || true)
    if [ -n "$job_pids" ]; then
        log_with_timestamp "Killing remaining child processes..."
        kill $job_pids || true
        wait $job_pids || true
    fi
    
    # Clean up temp files
    rm -f "$GDB_SCRIPT" || true
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Check if application exists
if [ ! -f "$APP_PATH" ]; then
    log_with_timestamp "Error: Application not found at $APP_PATH"
    log_with_timestamp "Please build the project first"
    exit 1
fi

# Check if VULKAN_SDK is set
if [ -z "$VULKAN_SDK" ]; then
    log_with_timestamp "Note: VULKAN_SDK not set"
fi

# Check if xvfb-run is available
if ! command -v xvfb-run &> /dev/null; then
    log_with_timestamp "Error: xvfb-run not found. Please install xvfb."
    exit 1
fi

# Check if xwd is available
if ! command -v xwd &> /dev/null; then
    log_with_timestamp "Error: xwd not found. Please install xorg-x11-utils."
    exit 1
fi

# Check if ImageMagick is available (prioritize magick, fall back to convert)
CONVERT_CMD=""
if command -v magick &> /dev/null; then
    CONVERT_CMD="magick"
elif command -v convert &> /dev/null; then
    CONVERT_CMD="convert"
else
    log_with_timestamp "Error: Neither magick nor convert (ImageMagick) found. Please install ImageMagick."
    exit 1
fi

# Check if xdotool is available (for setting window size/position)
USE_XDOTOOL=false
if command -v xdotool &> /dev/null; then
    USE_XDOTOOL=true
    log_with_timestamp "xdotool found - will attempt to resize window to fill framebuffer"
else
    log_with_timestamp "Note: xdotool not found - window will use default size"
    log_with_timestamp "      Install xdotool for automatic window resizing"
fi

# Check if gdb is available for crash debugging
USE_GDB=false
if command -v gdb &> /dev/null; then
    USE_GDB=true
    log_with_timestamp "gdb found - will run with gdb to capture backtrace on crash"
else
    log_with_timestamp "Note: gdb not found - crashes will not show backtrace"
    log_with_timestamp "      Install gdb for better crash debugging"
fi

# Use MIN_TIMEOUT to ensure total timeout (handles decimal values)
# Use MIN_TIMEOUT to ensure total timeout (handles decimal values)
MIN_TIMEOUT=$(awk "BEGIN { print $SCREENSHOT_DELAY + 3 }")
if awk "BEGIN { exit !($TIMEOUT < $MIN_TIMEOUT) }"; then
    TIMEOUT="$MIN_TIMEOUT"
    log_with_timestamp "Note: Increasing timeout to ${TIMEOUT} seconds due to screenshot delay"
fi

# Write GDB commands to the temp script file
cat > "$GDB_SCRIPT" << 'GDBEOF'
set confirm off
set width 0
set height 0
set verbose off
set print thread-events off
catch throw
commands 1
where
continue
end
catch catch
commands 2
where
continue
end
catch signal SIGTRAP
commands 3
silent
python
import gdb
try:
    frame = gdb.selected_frame()
    func_name = frame.name()
    # Only suppress dynamic linker and C++ exception machinery
    if func_name and (func_name.startswith('dl_') or func_name.startswith('_dl_') or 
                      func_name.startswith('__cxa_')):
        gdb.execute('continue')
    else:
        # User code SIGTRAP - show stack trace
        print("SIGTRAP backtrace:")
        gdb.execute('bt')
        gdb.execute('continue')
except:
    # If we can't determine, just continue
    gdb.execute('continue')
end
end
handle SIGSEGV stop print
handle SIGABRT stop print
run
python
import gdb
try:
    # Check if inferior still exists and has frames
    if gdb.selected_inferior().is_valid():
        frame = gdb.selected_frame()
        if frame is not None:
            print("\n=== Fault Information ===")
            # Print the faulting address and reason from signal info
            try:
                siginfo = gdb.parse_and_eval("$_siginfo")
                if siginfo:
                    si_addr = siginfo['_sifields']['_sigfault']['si_addr']
                    si_code = int(siginfo['si_code'])
                    # Decode si_code for SIGSEGV and SIGBUS
                    code_map = {
                        1: "SEGV_MAPERR (address not mapped to object)",
                        2: "SEGV_ACCERR (invalid permissions for mapped object)",
                        3: "SEGV_BNDERR (bounds checking failure)",
                        4: "SEGV_PKUERR (protection key check failure)",
                    }
                    code_str = code_map.get(si_code, f"code {si_code}")
                    print(f"Fault address: {si_addr}")
                    print(f"Fault reason:  {code_str}")
            except:
                pass
            # Print the faulting instruction
            try:
                print("\nFaulting instruction:")
                gdb.execute('x/i $pc')
            except:
                pass
            # Print all general purpose registers
            try:
                print("\nRegisters:")
                gdb.execute('info registers')
            except:
                pass
            # Print local variables from first frame with debug info
            try:
                current_frame = gdb.newest_frame()
                while current_frame:
                    try:
                        # Check if this frame has source info (debug symbols)
                        sal = current_frame.find_sal()
                        if sal and sal.symtab and sal.symtab.filename:
                            print(f"\nLocal variables in {current_frame.name()}:")
                            gdb.execute('info locals')
                            break
                    except:
                        pass
                    current_frame = current_frame.older()
            except:
                pass
            print("\n=== Backtrace ===")
            gdb.execute('bt')
except:
    pass
end
quit
GDBEOF

# Use xvfb-run to launch the application and take a screenshot
# xvfb-run sets up the virtual framebuffer and DISPLAY automatically
log_with_timestamp "Launching $APP_PATH in xvfb..."

# Build the command with arguments
APP_CMD="'$APP_PATH'"
for arg in "${APP_ARGS[@]}"; do
    # Use printf %q to properly quote each argument
    APP_CMD+=" $(printf '%q' "$arg")"
done

xvfb-run -a -s "-screen 0 ${FB_WIDTH}x${FB_HEIGHT}x24" bash -c "
    # Set up cleanup handler in subshell
    trap 'kill \$APP_PID 2>/dev/null || true; wait \$APP_PID 2>/dev/null || true' EXIT INT TERM
    
    # Show which DISPLAY xvfb-run chose
    echo \"[\$(date +%H:%M:%S)] Child has DISPLAY=\$DISPLAY\"
    
    # Launch the application
    echo \"[\$(date +%H:%M:%S)] Starting application...\"
    if [ '$USE_GDB' = 'true' ]; then
        # Run GDB in a subshell so we can manage it independently
        # Let app output go directly to stdout/stderr, unbuffered
        stdbuf -oL -eL timeout ${TIMEOUT} gdb -batch -x '$GDB_SCRIPT' --args $APP_CMD &
        APP_PID=\$!
    else
        # Let app output go directly to stdout/stderr, unbuffered
        stdbuf -oL -eL timeout ${TIMEOUT} $APP_CMD &
        APP_PID=\$!
    fi

    sleep 0.1
    if [[ ! -d /proc/\$APP_PID ]]; then
        echo 'Error: Application exited immediately, waiting for output...'
        wait \$APP_PID 2>&1 || true
        exit 1
    fi
    
    # Try to resize the window to fill the framebuffer
    if [ '$USE_XDOTOOL' = 'true' ]; then
        # Wait a bit more for the window to fully appear
        sleep 0.1
        
        # Find the window ID using --onlyvisible --class . (more reliable than searching by name)
        WINDOW_ID=\"\"
        for attempt in {1..30}; do
            WINDOW_ID=\$(xdotool search --onlyvisible --class . 2>/dev/null | head -1)
            if [ -n \"\$WINDOW_ID\" ]; then
                break
            fi
            # Wait a bit before retrying
            sleep 0.1
        done
        
        if [ -n \"\$WINDOW_ID\" ]; then
            echo \"[\$(date +%H:%M:%S)] Resizing window (ID: \$WINDOW_ID) to ${FB_WIDTH}x${FB_HEIGHT}...\"
            # Move window to 0,0 and resize to framebuffer size
            xdotool windowmove \"\$WINDOW_ID\" 0 0 2>/dev/null || true
            xdotool windowsize \"\$WINDOW_ID\" ${FB_WIDTH} ${FB_HEIGHT} 2>/dev/null || {
                echo 'Warning: Failed to resize window'
            }
            # Give time for resize to take effect
            sleep 0.5
        else
            echo 'Warning: Could not find application window to resize'
        fi
    fi
    
    # Wait for the application to render
    echo \"[\$(date +%H:%M:%S)] Waiting ${SCREENSHOT_DELAY} seconds before screenshot...\"
    
    # Wait with timeout, checking if app is still running
    start_time=\$(date +%s)
    while true; do
        # Check if app is still running
        if ! kill -0 \$APP_PID 2>/dev/null; then
            echo 'Error: Application exited before screenshot delay completed'
            exit 1
        fi
        
        # Check if enough time has elapsed
        current_time=\$(date +%s)
        elapsed=\$((current_time - start_time))
        if [ \$elapsed -ge ${SCREENSHOT_DELAY} ]; then
            break
        fi
        
        sleep 0.1
    done
    
    # Final check before taking screenshot
    if ! kill -0 \$APP_PID 2>/dev/null; then
        echo 'Error: Application exited before screenshot could be taken'
        exit 1
    fi
    
    # Take screenshot using xwd and pipe directly to PNG conversion
    # DISPLAY is automatically set by xvfb-run
    echo \"[\$(date +%H:%M:%S)] Taking screenshot...\"
    # Pipe xwd directly to ImageMagick to avoid intermediate file
    # Use the detected convert command (magick or convert)
    xwd -root | $CONVERT_CMD xwd:- '$SCREENSHOT_PNG' || {
        echo 'Error: Failed to take screenshot'
        exit 1
    }
    
    # Wait for app to finish or kill it
    kill \$APP_PID || true
    wait \$APP_PID || true
    sleep 0.5 # wait for a graceful exit
" || {
    log_with_timestamp "Error: xvfb-run command failed"
    cleanup
    exit 1
}

log_with_timestamp "Done! Screenshot saved as $SCREENSHOT_PNG"
