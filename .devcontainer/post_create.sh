#!/bin/bash
echo "-----POSTCOMMAND START-----"

# Login to GitHub (can be skipped with Ctrl+C)
BROWSER=False gh auth login -h github.com -s user || true
/app/.devcontainer/set_git_conig_from_gh.sh || true

echo "-----POSTCOMMANDS END-----"

