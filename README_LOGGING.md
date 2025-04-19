# AMI Backend Logging System

This document describes the logging system implemented in the AMI Backend application and how to use the provided utilities for managing and analyzing logs.

## Logging Configuration

The AMI Backend uses Python's built-in `logging` module with the following features:

- **Log Levels:** DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Default Level:** INFO
- **Log Format:** 
  - Console: `%(levelname)s: %(message)s`
  - File: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Log Files:**
  - Main log file: `logs/amibackend.log`
  - Rotating file handler that keeps up to 5 backup files of 10MB each
  - Console output is also logged to `logs/console_*.log` with timestamps

## Server Startup Script

A convenient script `start_server.sh` is provided to start the server with the appropriate logging configuration.

### Basic Usage

```bash
./start_server.sh
```

This will start the server with the default port (5000) and log level (INFO).

### Advanced Options

```bash
./start_server.sh [options]
```

Options:
- `-p, --port PORT`: Specify the port to run on (default: 5000)
- `-l, --log-level LEVEL`: Set log level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
- `-f, --force-kill`: Force kill any process running on the specified port
- `-h, --help`: Display help message

Examples:
```bash
# Start with debug logging
./start_server.sh --log-level DEBUG

# Start on a different port
./start_server.sh --port 8000

# Kill existing process and start on port 5000
./start_server.sh --force-kill
```

## Log Analysis Tool

The `analyze_logs.py` script helps you analyze log files to identify common issues and patterns.

### Basic Usage

```bash
./analyze_logs.py
```

This will analyze the default log file (`logs/amibackend.log`) and the most recent console log.

### Advanced Options

```bash
./analyze_logs.py [options]
```

Options:
- `-f, --log-file FILE`: Specify the log file to analyze (default: logs/amibackend.log)
- `-c, --console-log FILE`: Specify a console log file (default: most recent in logs/)
- `-t, --last-hours HOURS`: Only analyze logs from the last N hours (default: 24)
- `-e, --errors-only`: Show only errors
- `-s, --stats`: Show statistics summary

Examples:
```bash
# Show only errors from the last hour
./analyze_logs.py --last-hours 1 --errors-only

# Full statistics on a specific log file
./analyze_logs.py --log-file logs/amibackend.log.1 --stats
```

## Common Log Messages

### Server Startup

- `INFO: Global event loop initialized` - The event loop has been successfully set up
- `INFO: Starting server on port {port}` - Server is starting on the specified port
- `WARNING: Port {port} already in use, trying port {port+1}` - Port conflict detected, trying next port

### Error Patterns

- `RuntimeError: Event loop is closed` - Occurs when trying to use a closed event loop
- `RuntimeError: got Future <Future...> attached to a different loop` - Event loop conflict
- `OSError: [Errno 48] Address already in use` - Port is already in use by another process

### Async Operation

- `INFO: Thread {id} started` - A new worker thread has started
- `INFO: Thread {id} completed` - A worker thread has completed
- `ERROR: Error in thread {id}: {error}` - An error occurred in a worker thread

## Troubleshooting Common Issues

### Address Already In Use

If you see an error about the address already being in use:

1. Use the `--force-kill` option with the startup script:
   ```bash
   ./start_server.sh --force-kill
   ```

2. Or manually find and kill the process:
   ```bash
   lsof -i :5000  # Find process using port 5000
   kill -9 <PID>  # Kill the process
   ```

### Event Loop Errors

If you see event loop-related errors:

1. These are generally handled by the patched async functions in the application
2. Restarting the server usually resolves these issues
3. In persistent cases, check the system for too many concurrent connections

### Missing Module Errors

If you see errors about missing modules:

1. Ensure your virtual environment is activated
2. Install dependencies from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

## Adding More Logging to the Code

To add more logging to your code:

```python
from utilities import logger

# Log at different levels
logger.debug("Detailed information, typically of interest only when diagnosing problems")
logger.info("Confirmation that things are working as expected")
logger.warning("An indication that something unexpected happened")
logger.error("The application has failed to perform some function")
logger.critical("A very serious error that might prevent the program from continuing")
```

## Log File Management

Log files can grow large over time. The system automatically rotates log files, but you may want to periodically archive or clean up old logs:

```bash
# Archive logs older than 30 days
find logs -name "*.log.*" -mtime +30 -exec gzip {} \;

# Delete logs older than 60 days
find logs -name "*.log.*" -mtime +60 -delete
``` 