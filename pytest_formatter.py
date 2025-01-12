#!/usr/bin/env python3
"""
Pytest Formatter - Simplify and format pytest output for easier sharing

Quick Start:
    python pytest_formatter.py              # Basic usage
    python pytest_formatter.py -v -c        # Verbose output, copy to clipboard
    python pytest_formatter.py -f errors    # Show only errors

Options:
    -o, --output FILE    Save output to specified file (default: pytest_summary.txt)
    -v, --verbose       Include detailed output with timestamps and stats
    -c, --clipboard     Automatically copy results to clipboard
    -f, --filter TYPE   Filter output by: all|errors|failures
    -j, --json         Output in JSON format
    --history          Save results to pytest_history.json for tracking

Requirements:
    pip install pyperclip

Examples:
    # Show only errors and copy to clipboard
    python pytest_formatter.py -f errors -c

    # Generate verbose JSON output with history
    python pytest_formatter.py -v -j --history

    # Custom output file with verbose output
    python pytest_formatter.py -o custom_results.txt -v
"""

import sys
import re
import argparse
from pathlib import Path
import subprocess
from typing import List, Tuple
import pyperclip  # For clipboard functionality
from datetime import datetime
import json

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Format pytest output for easier sharing')
    parser.add_argument('-o', '--output', type=str, default='pytest_summary.txt',
                      help='Output file path (default: pytest_summary.txt)')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Include more detailed output')
    parser.add_argument('-c', '--clipboard', action='store_true',
                      help='Copy results to clipboard')
    parser.add_argument('-f', '--filter', choices=['all', 'errors', 'failures'],
                      default='all', help='Filter output type')
    parser.add_argument('-j', '--json', action='store_true',
                      help='Output in JSON format')
    parser.add_argument('--history', action='store_true',
                      help='Save results to history file')
    return parser.parse_args()

def run_pytest() -> str:
    """Run pytest with short traceback and capture output."""
    args = ['pytest', '--tb=short']
    if not is_verbose():
        args.append('-q')

    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout + result.stderr

def parse_pytest_output(output: str) -> Tuple[str, List[str], List[str], dict]:
    """Parse pytest output into summary, errors, and failures."""
    # Extract summary
    summary_match = re.search(r'=+ short test summary info =+\n(.*?)\n=+.*in \d+\.\d+s =+',
                            output,
                            re.DOTALL)
    summary = summary_match.group(1) if summary_match else "No summary found"

    # Extract errors
    errors = re.findall(r'ERROR.*?\n(.*?)\n', output)

    # Extract failures
    failures = re.findall(r'FAILED.*?\n(.*?)\n', output)

    # Extract statistics
    stats_match = re.search(r'(\d+) failed, (\d+) passed, (\d+) errors', output)
    stats = {
        'failed': int(stats_match.group(1)) if stats_match else 0,
        'passed': int(stats_match.group(2)) if stats_match else 0,
        'errors': int(stats_match.group(3)) if stats_match else 0
    } if stats_match else {}

    return summary, errors, failures, stats

def format_output(summary: str, errors: List[str], failures: List[str],
                 stats: dict, args: argparse.Namespace) -> str:
    """Format the parsed output based on arguments."""
    if args.json:
        return format_json_output(summary, errors, failures, stats)

    output_parts = []

    # Add timestamp if verbose
    if args.verbose:
        output_parts.append(f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    output_parts.append("=== Pytest Summary ===\n")

    # Add summary if showing all or if there are issues
    if args.filter == 'all' or (errors and args.filter == 'errors') or (failures and args.filter == 'failures'):
        output_parts.append(f"Summary:\n{summary}\n")

    # Add errors if showing all or errors
    if (args.filter in ['all', 'errors']) and errors:
        output_parts.append("\nErrors:")
        for error in errors:
            output_parts.append(f"- {error.strip()}")

    # Add failures if showing all or failures
    if (args.filter in ['all', 'failures']) and failures:
        output_parts.append("\nFailures:")
        for failure in failures:
            output_parts.append(f"- {failure.strip()}")

    # Add statistics if verbose
    if args.verbose and stats:
        output_parts.append("\nStatistics:")
        output_parts.append(f"- Passed: {stats.get('passed', 0)}")
        output_parts.append(f"- Failed: {stats.get('failed', 0)}")
        output_parts.append(f"- Errors: {stats.get('errors', 0)}")

    return "\n".join(output_parts)

def format_json_output(summary: str, errors: List[str], failures: List[str], stats: dict) -> str:
    """Format output as JSON."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'errors': errors,
        'failures': failures,
        'statistics': stats
    }
    return json.dumps(data, indent=2)

def save_output(formatted_output: str, output_file: Path, args: argparse.Namespace):
    """Save the formatted output to a file."""
    # Save main output
    with output_file.open('w') as f:
        f.write(formatted_output)

    # Save to history if requested
    if args.history:
        history_file = Path('pytest_history.json')
        try:
            if history_file.exists():
                with history_file.open('r') as f:
                    history = json.load(f)
            else:
                history = []

            history.append({
                'timestamp': datetime.now().isoformat(),
                'output': formatted_output
            })

            with history_file.open('w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save to history file: {e}")

    print(f"\nFormatted pytest results saved to: {output_file.absolute()}")

    # Copy to clipboard if requested
    if args.clipboard:
        try:
            pyperclip.copy(formatted_output)
            print("Results copied to clipboard!")
        except Exception as e:
            print(f"Warning: Failed to copy to clipboard: {e}")

    # Show preview
    print("\nResults preview:")
    print("=" * 40)
    print(formatted_output)
    print("=" * 40)

def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return '--verbose' in sys.argv or '-v' in sys.argv

def main():
    """Main function to run pytest and format output."""
    args = parse_arguments()

    # Run pytest and get output
    print("Running pytest...")
    pytest_output = run_pytest()

    # Parse the output
    summary, errors, failures, stats = parse_pytest_output(pytest_output)

    # Format the results
    formatted_output = format_output(summary, errors, failures, stats, args)

    # Save to file
    output_file = Path(args.output)
    save_output(formatted_output, output_file, args)

if __name__ == "__main__":
    main()
