#!/usr/bin/env python3
"""
Log Analyzer for AMI Backend

This script analyzes log files to identify common issues and display statistics.
It looks for patterns in errors, counts occurrences of different error types,
and provides recommendations for fixing common problems.
"""

import os
import re
import sys
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze AMI backend log files.')
    parser.add_argument('--log-file', '-f', default='logs/amibackend.log',
                      help='Path to log file (default: logs/amibackend.log)')
    parser.add_argument('--console-log', '-c', default=None,
                      help='Path to console log file (default: most recent in logs/)')
    parser.add_argument('--last-hours', '-t', type=int, default=24,
                      help='Only analyze logs from the last N hours (default: 24)')
    parser.add_argument('--errors-only', '-e', action='store_true',
                      help='Show only errors')
    parser.add_argument('--stats', '-s', action='store_true',
                      help='Show statistics summary')
    
    return parser.parse_args()

def find_most_recent_console_log():
    """Find the most recent console log file in the logs directory."""
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        return None
    
    console_logs = [f for f in os.listdir(logs_dir) if f.startswith('console_') and f.endswith('.log')]
    if not console_logs:
        return None
    
    # Sort by modification time (most recent first)
    console_logs.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)
    return os.path.join(logs_dir, console_logs[0])

def parse_timestamp(line):
    """
    Extract timestamp from a log line. 
    Handles multiple timestamp formats commonly used in logs.
    """
    # Try various timestamp formats
    timestamp_patterns = [
        # ISO format with milliseconds: 2023-11-27T14:23:45.123
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})',
        # Standard format: 2023-11-27 14:23:45,123
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})',
        # Simple format: 2023-11-27 14:23:45
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
        # Bracket format: [2023-11-27 14:23:45]
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]',
    ]
    
    for pattern in timestamp_patterns:
        match = re.search(pattern, line)
        if match:
            ts_str = match.group(1)
            try:
                # Handle different formats
                if 'T' in ts_str:
                    # ISO format
                    return datetime.fromisoformat(ts_str)
                elif ',' in ts_str:
                    # Standard format with comma for milliseconds
                    return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S,%f')
                else:
                    # Simple format
                    return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
    
    return None  # No valid timestamp found

def extract_error_info(line):
    """Extract error type, message and other details from a log line."""
    # Extract log level
    level_match = re.search(r'(ERROR|WARNING|CRITICAL|INFO|DEBUG):', line)
    level = level_match.group(1) if level_match else 'UNKNOWN'
    
    # Extract error message patterns
    error_patterns = [
        # Python exceptions
        r'(?:ERROR|CRITICAL).*?([A-Za-z.]+Error|Exception): (.+?)(?:\n|$)',
        # General error message
        r'(?:ERROR|CRITICAL): (.+?)(?:\n|$)',
        # Warning messages
        r'WARNING: (.+?)(?:\n|$)'
    ]
    
    for pattern in error_patterns:
        match = re.search(pattern, line)
        if match:
            if len(match.groups()) > 1:
                error_type = match.group(1)
                message = match.group(2)
            else:
                error_type = "General" + level
                message = match.group(1)
            
            return {
                'level': level,
                'type': error_type,
                'message': message.strip(),
                'line': line.strip()
            }
    
    # If no specific pattern matches but it's an error/warning
    if level in ('ERROR', 'WARNING', 'CRITICAL'):
        return {
            'level': level,
            'type': f'Generic{level}',
            'message': line.strip(),
            'line': line.strip()
        }
    
    return None

def analyze_log_file(file_path, since=None, errors_only=False):
    """
    Analyze a log file and extract relevant information.
    
    Args:
        file_path: Path to the log file
        since: Only process log entries after this datetime
        errors_only: Whether to only process error entries
    
    Returns:
        A dict containing analyzed log information
    """
    if not os.path.exists(file_path):
        print(f"Error: Log file not found: {file_path}")
        return None
    
    events = []
    error_counts = Counter()
    log_levels = Counter()
    error_examples = defaultdict(list)
    
    # Parse the log file
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Skip empty lines
            if not line.strip():
                continue
            
            # Extract timestamp
            timestamp = parse_timestamp(line)
            
            # Skip old entries if since is specified
            if since and timestamp and timestamp < since:
                continue
            
            # Process line based on content
            if errors_only and not any(level in line for level in ('ERROR', 'WARNING', 'CRITICAL')):
                continue
            
            # Extract log level
            if 'DEBUG:' in line:
                log_levels['DEBUG'] += 1
            elif 'INFO:' in line:
                log_levels['INFO'] += 1
            elif 'WARNING:' in line:
                log_levels['WARNING'] += 1
            elif 'ERROR:' in line:
                log_levels['ERROR'] += 1
            elif 'CRITICAL:' in line:
                log_levels['CRITICAL'] += 1
            
            # Extract error information
            error_info = extract_error_info(line)
            if error_info:
                events.append({
                    'timestamp': timestamp,
                    'line_num': line_num,
                    'error_info': error_info
                })
                
                error_type = error_info['type']
                error_counts[error_type] += 1
                
                # Store example of this error type (up to 3 per type)
                if len(error_examples[error_type]) < 3:
                    error_examples[error_type].append({
                        'line_num': line_num,
                        'message': error_info['message'],
                        'timestamp': timestamp
                    })
    
    return {
        'file_path': file_path,
        'events': events,
        'error_counts': error_counts,
        'log_levels': log_levels,
        'error_examples': error_examples,
        'total_lines': line_num if 'line_num' in locals() else 0
    }

def get_error_recommendations(error_type):
    """Provide recommendations for fixing common error types."""
    recommendations = {
        'RuntimeError': [
            "Check for event loop conflicts in async code",
            "Ensure you're not mixing asyncio libraries or contexts",
            "Consider using the run_async_in_thread wrapper for async operations"
        ],
        'ConnectionError': [
            "Check network connectivity",
            "Verify API endpoints are accessible",
            "Check if service is running and accessible"
        ],
        'ValueError': [
            "Check input validation in the code",
            "Ensure data format matches what the code expects"
        ],
        'TimeoutError': [
            "Increase timeout settings for operations",
            "Check if external services are responding slowly",
            "Consider optimizing slow operations"
        ],
        'ModuleNotFoundError': [
            "Ensure all dependencies are installed correctly",
            "Check for typos in import statements",
            "Verify the virtual environment is activated"
        ],
        'FileNotFoundError': [
            "Check if the referenced file paths exist",
            "Verify permissions on the files",
            "Ensure paths are relative to the working directory"
        ],
        'KeyError': [
            "Check for missing dictionary keys",
            "Ensure API responses contain expected fields",
            "Add validation before accessing dictionary keys"
        ]
    }
    
    # Check for partial matches for error types not directly in our dict
    for known_error in recommendations:
        if known_error in error_type:
            return recommendations[known_error]
    
    # Default recommendations for unknown error types
    return [
        "Review the error message for specific details",
        "Check related code for logical errors",
        "Consider adding more logging around this area of code"
    ]

def print_analysis(analysis, show_stats=True, errors_only=True):
    """Print analysis results in a readable format."""
    if not analysis:
        return
    
    file_path = analysis['file_path']
    events = analysis['events']
    error_counts = analysis['error_counts']
    log_levels = analysis['log_levels']
    error_examples = analysis['error_examples']
    total_lines = analysis['total_lines']
    
    print(f"\n{'='*80}")
    print(f"Analysis of {file_path}")
    print(f"{'='*80}")
    
    if show_stats:
        print("\nLog Statistics:")
        print(f"Total lines: {total_lines}")
        print("Log levels breakdown:")
        for level, count in log_levels.items():
            percentage = (count / total_lines) * 100 if total_lines > 0 else 0
            print(f"  {level}: {count} ({percentage:.1f}%)")
    
    if error_counts:
        print("\nError Types Summary:")
        for error_type, count in error_counts.most_common():
            print(f"  {error_type}: {count}")
        
        print("\nError Examples and Recommendations:")
        for error_type, examples in error_examples.items():
            print(f"\n{error_type} ({error_counts[error_type]} occurrences):")
            for i, example in enumerate(examples, 1):
                timestamp = example['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if example['timestamp'] else 'N/A'
                print(f"  Example {i} (Line {example['line_num']}, {timestamp}):")
                print(f"    {example['message']}")
            
            # Print recommendations
            print("  Recommendations:")
            for rec in get_error_recommendations(error_type):
                print(f"    - {rec}")
    else:
        print("\nNo errors found in the log file.")

def main():
    """Main function to analyze logs."""
    args = parse_args()
    
    # Determine the log files to analyze
    log_file = args.log_file
    console_log = args.console_log or find_most_recent_console_log()
    
    # Determine the time range
    since = datetime.now() - timedelta(hours=args.last_hours) if args.last_hours > 0 else None
    
    # Print header
    print(f"\nAMI Backend Log Analyzer")
    print(f"Time range: {since.strftime('%Y-%m-%d %H:%M:%S') if since else 'All time'}")
    
    # Analyze log files
    log_analysis = analyze_log_file(log_file, since, args.errors_only)
    if log_analysis:
        print_analysis(log_analysis, args.stats, args.errors_only)
    
    if console_log:
        console_analysis = analyze_log_file(console_log, since, args.errors_only)
        if console_analysis:
            print_analysis(console_analysis, args.stats, args.errors_only)
    
    # Print summary recommendations
    print("\nGeneral Recommendations:")
    print("  1. Ensure the logs directory exists and is writable")
    print("  2. Check for 'Address already in use' errors which indicate port conflicts")
    print("  3. Verify that all required environment variables are set")
    print("  4. Consider adjusting the log level for more detailed diagnostics")
    print("  5. Use --force-kill option if the server won't start due to port conflicts")

if __name__ == "__main__":
    main() 