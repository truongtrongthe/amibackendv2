#!/bin/bash

# Default configuration
PORT=5001
LOG_LEVEL="INFO"
FORCE_KILL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -l|--log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    -f|--force-kill)
      FORCE_KILL=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  -p, --port PORT         Specify the port to run on (default: 5000)"
      echo "  -l, --log-level LEVEL   Set log level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)"
      echo "  -f, --force-kill        Force kill any process running on the specified port"
      echo "  -h, --help              Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Validate LOG_LEVEL
case $LOG_LEVEL in
  DEBUG|INFO|WARNING|ERROR|CRITICAL)
    ;;
  *)
    echo "Invalid log level: $LOG_LEVEL"
    echo "Valid options are: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    exit 1
    ;;
esac

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables for configuration
export PORT=$PORT
export DEBUG_LEVEL=$LOG_LEVEL

# Check if the port is in use
PORT_PID=$(lsof -t -i:$PORT 2>/dev/null)
if [ ! -z "$PORT_PID" ]; then
    if [ "$FORCE_KILL" = true ]; then
        echo "Found process $PORT_PID using port $PORT, killing it..."
        kill -9 $PORT_PID
        sleep 1
    else
        echo "Error: Port $PORT is already in use by process $PORT_PID"
        echo "Use --force-kill to terminate the process and start the server"
        exit 1
    fi
fi

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONSOLE_LOG="logs/console_${TIMESTAMP}.log"

# Start the server with specified configuration
echo "====================================================="
echo "Starting server with the following configuration:"
echo "  Port:      $PORT"
echo "  Log Level: $LOG_LEVEL"
echo "  Log File:  logs/amibackend.log (rotating, max 5 files of 10MB each)"
echo "  Console:   $CONSOLE_LOG"
echo "====================================================="

# Run the Python application with output to console and file
{
    echo "[$(date)] Server starting with PORT=$PORT, LOG_LEVEL=$LOG_LEVEL"
    python main.py 2>&1 
} | tee -a "$CONSOLE_LOG"

# Get the exit status of the Python command
EXIT_STATUS=${PIPESTATUS[0]}

if [ $EXIT_STATUS -ne 0 ]; then
    echo "[$(date)] Server exited with code $EXIT_STATUS" | tee -a "$CONSOLE_LOG"
    echo "Check the logs for details: $CONSOLE_LOG and logs/amibackend.log"
else
    echo "[$(date)] Server stopped normally" | tee -a "$CONSOLE_LOG"
fi

exit $EXIT_STATUS 