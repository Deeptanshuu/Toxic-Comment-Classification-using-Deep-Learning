#!/bin/bash

SESSION_NAME="jupyter"
PASSWORD="deeptanshul"

# Generate hashed password
HASHED_PASSWORD=$(python3 -c "from jupyter_server.auth import passwd; print(passwd('$PASSWORD'))")

# Generate Jupyter config if it doesn't exist
CONFIG_FILE=~/.jupyter/jupyter_notebook_config.py
if [ ! -f "$CONFIG_FILE" ]; then
    jupyter notebook --generate-config --allow-root
fi

# Inject config (idempotent)
grep -q "NotebookApp.password" "$CONFIG_FILE" || cat <<EOL >> "$CONFIG_FILE"

# Auto-added by setup script
c.NotebookApp.password = u'$HASHED_PASSWORD'
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.allow_remote_access = True
c.NotebookApp.allow_origin = '*'
EOL

# Check for tmux session and create if not exists
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
  tmux new-session -d -s $SESSION_NAME -n notebook
  tmux send-keys -t "$SESSION_NAME:notebook" "jupyter server --NotebookApp.password='$HASHED_PASSWORD' --no-browser --port=8888 --ip=0.0.0.0" C-m
  echo "Started new tmux session: $SESSION_NAME"
else
  echo "tmux session '$SESSION_NAME' already exists."
fi

