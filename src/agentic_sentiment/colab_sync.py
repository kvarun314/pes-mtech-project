"""Commits and pushes Colab-produced files (checkpoints' history, results,
charts) straight back to the repo, so a run's outputs are documented in
git, not stranded on Drive until someone remembers to copy them over.

The token is only ever placed in a subprocess argv list (visible to local
process listing on that Colab VM, same exposure as any CLI tool using a
token this way) -- it is never interpolated into a shell string, never
logged, and is stripped out of any error text before it's raised or
returned, so a failed push can't leak it into a notebook cell's output."""

import subprocess


def push_repo_changes(repo_dir: str, paths: list[str], message: str, token: str,
                       remote: str = "https://github.com/kvarun314/pes-mtech-project.git",
                       branch: str = "main") -> str:
    """git add <paths>, commit, and push <branch> to <remote> using <token>
    for HTTPS auth. Returns a short status string. Raises RuntimeError (with
    the token redacted from its message) if any step fails."""
    if not token:
        raise RuntimeError("push_repo_changes: empty token")

    authed_remote = remote.replace("https://", f"https://{token}@", 1)

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", repo_dir, *args], capture_output=True, text=True,
        )

    add = run("add", *paths)
    if add.returncode != 0:
        raise RuntimeError(f"git add failed: {_redact(add.stderr, token)}")

    commit = run("-c", "user.email=colab@pes-mtech-project", "-c", "user.name=Colab Run",
                  "commit", "-m", message, "--allow-empty")
    if commit.returncode != 0:
        raise RuntimeError(f"git commit failed: {_redact(commit.stderr, token)}")

    push = subprocess.run(
        ["git", "-C", repo_dir, "push", authed_remote, branch],
        capture_output=True, text=True,
    )
    if push.returncode != 0:
        raise RuntimeError(f"git push failed: {_redact(push.stderr, token)}")

    return f"Pushed {len(paths)} path(s) to {branch}: {message}"


def _redact(text: str, token: str) -> str:
    return text.replace(token, "<redacted>") if token else text
