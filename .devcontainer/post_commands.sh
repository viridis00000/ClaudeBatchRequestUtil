#!/bin/bash
echo "-----POSTCOMMAND START-----"

BROWSER=False gh auth login -h github.com -s user
/app/.devcontainer/set_git_conig_from_gh.sh

echo "-----POSTCOMMANDS END-----"

