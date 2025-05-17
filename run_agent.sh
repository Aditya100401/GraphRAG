#!/bin/bash

# Script to run the event prediction agent
# Usage: ./run_agent.sh -q "Your question" [-a "Actor name"] [-r "Recipient name"] [-d "YYYY-MM-DD"]

# Default values
QUERY=""
ACTOR=""
RECIPIENT=""
DATE=""
DEBUG=false

# Parse command line options
while getopts "q:a:r:d:hv" opt; do
    case $opt in
    q) QUERY="$OPTARG" ;;
    a) ACTOR="$OPTARG" ;;
    r) RECIPIENT="$OPTARG" ;;
    d) DATE="$OPTARG" ;;
    h) # Help option
        echo "Usage: $0 -q \"Your question\" [-a \"Actor name\"] [-r \"Recipient name\"] [-d \"YYYY-MM-DD\"] [-v]"
        echo ""
        echo "Options:"
        echo "  -q  Query (required): The question to ask the agent"
        echo "  -a  Actor (optional): The actor name"
        echo "  -r  Recipient (optional): The recipient name"
        echo "  -d  Date (optional): The date in YYYY-MM-DD format"
        echo "  -v  Verbose mode: Show debugging information"
        echo "  -h  Show this help message"
        exit 0
        ;;
    v) DEBUG=true ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

# Check if query is provided
if [ -z "$QUERY" ]; then
    echo "Error: Query is required. Use -q \"Your question\""
    echo "Use -h for help"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if the create_agent.py file exists
if [ ! -f "create_agent.py" ]; then
    echo "Error: create_agent.py not found in the current directory"
    exit 1
fi

# Print debug information if verbose mode is enabled
if $DEBUG; then
    echo "Running with parameters:"
    echo "  Query: $QUERY"
    echo "  Actor: $ACTOR"
    echo "  Recipient: $RECIPIENT"
    echo "  Date: $DATE"
    echo ""
fi

# Construct the command
CMD="python3 create_agent.py --query \"$QUERY\""

# Add optional parameters if provided
if [ ! -z "$ACTOR" ]; then
    CMD="$CMD --actor \"$ACTOR\""
fi

if [ ! -z "$RECIPIENT" ]; then
    CMD="$CMD --recipient \"$RECIPIENT\""
fi

if [ ! -z "$DATE" ]; then
    CMD="$CMD --date \"$DATE\""
fi

# Print the command if in debug mode
if $DEBUG; then
    echo "Executing command: $CMD"
    echo ""
fi

# Execute the command (using eval to handle the quotes properly)
eval $CMD

# Exit with the same status as the Python script
exit $?