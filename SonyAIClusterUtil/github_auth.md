# GitHub Authentication Setup for Sony Cluster

This guide explains how to set up GitHub authentication on the Sony cluster to avoid entering passwords every time you use `git pull`, `git push`, etc.

## Option 1: SSH Keys (Recommended)

SSH keys provide the most secure and convenient way to authenticate with GitHub.

### Step 1: Generate SSH Key on Cluster

SSH into the cluster and generate a new SSH key:

```bash
ssh mfml1
ssh-keygen -t ed25519 -C "your_email@example.com"
```

- Press Enter to accept the default file location (`~/.ssh/id_ed25519`)
- Optionally set a passphrase (recommended for security) or press Enter for no passphrase
- The key pair will be created: `~/.ssh/id_ed25519` (private) and `~/.ssh/id_ed25519.pub` (public)

### Step 2: Display Your Public Key

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the entire output. It should look like:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG4iCnReKZ9E3VOEk+cT07gNjNaXI8KM/df8T30uJakU your_email@example.com
```

### Step 3: Add Public Key to GitHub

1. Go to GitHub Settings: https://github.com/settings/keys
2. Click **"New SSH key"**
3. Fill in:
   - **Title**: A descriptive name (e.g., "Sony Cluster mfml1")
   - **Key**: Paste the entire public key you copied in Step 2
4. Click **"Add SSH key"**
5. You may be prompted to enter your GitHub password to confirm

### Step 4: Configure Git to Use SSH

Change your repository's remote URL from HTTPS to SSH:

```bash
cd /music-shared-disk/group/ct/yiwen/codes/FLUX_finetune
git remote set-url origin git@github.com:yiwenchen1999/FLUX_finetune.git

# Verify the change
git remote -v
# Should show: git@github.com:yiwenchen1999/FLUX_finetune.git
```

### Step 5: Test SSH Connection

```bash
ssh -T git@github.com
```

You should see:
```
Hi yiwenchen1999! You've successfully authenticated, but GitHub does not provide shell access.
```

### Step 6: Test Git Operations

```bash
git pull
# Should work without asking for password!
```

## Troubleshooting

### "Permission denied (publickey)" Error

If you get this error when running `git pull` or `ssh -T git@github.com`:

1. **Verify the public key is added to GitHub:**
   - Go to: https://github.com/settings/keys
   - Check if your key is listed
   - If not, add it following Step 3 above

2. **Ensure SSH agent is running and has the key:**
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```

3. **Check key permissions:**
   ```bash
   ls -la ~/.ssh/id_ed25519
   # Should show: -rw------- (600)
   # If not, fix with:
   chmod 600 ~/.ssh/id_ed25519
   chmod 644 ~/.ssh/id_ed25519.pub
   ```

4. **Test with verbose output:**
   ```bash
   ssh -vT git@github.com
   # This will show detailed connection information
   ```

### Multiple SSH Keys

If you have multiple SSH keys, you can configure which key to use for GitHub by creating/editing `~/.ssh/config`:

```bash
cat >> ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/config
```

## Option 2: Git Credential Helper (HTTPS)

If you prefer to continue using HTTPS, you can configure Git to store your credentials:

### Store Credentials Permanently

```bash
git config --global credential.helper store
```

**Note:** This stores your password in plain text in `~/.git-credentials`. Less secure but convenient.

### Cache Credentials Temporarily (More Secure)

```bash
git config --global credential.helper 'cache --timeout=3600'
```

This caches your credentials for 1 hour (3600 seconds). Adjust the timeout as needed.

## Option 3: Personal Access Token (HTTPS)

If using HTTPS and you have 2FA enabled, you'll need a Personal Access Token instead of a password:

1. **Create a Personal Access Token:**
   - Go to: https://github.com/settings/tokens
   - Click **"Generate new token (classic)"**
   - Select scopes (at minimum, check `repo` for full repository access)
   - Click **"Generate token"**
   - **Copy the token immediately** (you won't be able to see it again)

2. **Use the token as your password:**
   - When Git prompts for a password, paste the token instead
   - The token will be stored if you've configured credential helper (Option 2)

## Quick Reference

```bash
# Check if SSH key exists
ls -la ~/.ssh/id_ed25519*

# Display public key
cat ~/.ssh/id_ed25519.pub

# Test GitHub connection
ssh -T git@github.com

# Switch to SSH (if using HTTPS)
git remote set-url origin git@github.com:yiwenchen1999/FLUX_finetune.git

# Verify remote URL
git remote -v
```

## Security Best Practices

1. **Use SSH keys** (Option 1) - Most secure and convenient
2. **Set a passphrase** on your SSH key for extra security
3. **Use different keys** for different machines/clusters
4. **Never share your private key** (`~/.ssh/id_ed25519`)
5. **Rotate keys periodically** if compromised or lost access

