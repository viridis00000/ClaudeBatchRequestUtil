#!/bin/bash
user=$(gh api -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" /user | jq -r .login)
email=$(gh api -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" /user/emails | jq -r ".[0].email")
echo "Setting $user <$email> as the default Git user..."
git config --global user.name "$user"
git config --global user.email "$email"
gh auth setup-git