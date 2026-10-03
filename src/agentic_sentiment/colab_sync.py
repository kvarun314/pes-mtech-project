"""Commits and pushes Colab-produced files (checkpoints' history, results,
charts) straight back to the repo, so a run's outputs are documented in
git, not stranded on Drive until someone remembers to copy them over.

The token is only ever placed in a subprocess argv list (visible to local
process listing on that Colab VM, same exposure as any CLI tool using a
token this way) -- it is never interpolated into a shell string, never
logged, and is stripped out of any error text before it's raised or
returned, so a failed push can't leak it into a notebook cell's output.

A long-running notebook (e.g. a multi-hour LoRA train) can sit between its
clone and its push for long enough that `main` has moved underneath it
(another notebook pushed, or a human did) -- push_repo_changes rebases
onto the remote branch first so that case doesn't just fail outright."""

import subprocess


def push_repo_changes(repo_dir: str, paths: list[str], message: str, token: str,
                       remote: str = "https://github.com/kvarun314/pes-mtech-project.git",
                       branch: str = "main") -> str:
    """git add <paths>, commit (skipped if nothing actually changed), rebase
    onto <remote>/<branch>, and push -- using <token> for HTTPS auth.
    Returns a short status string. Raises RuntimeError (with the token
    redacted from its message) if any step fails."""
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

    staged = run("diff", "--cached", "--quiet")
    if staged.returncode == 0:
        return f"Nothing to commit for {paths} -- skipped (already up to date)."

    commit = run("-c", "user.email=colab@pes-mtech-project", "-c", "user.name=Colab Run",
                  "commit", "-m", message)
    if commit.returncode != 0:
        raise RuntimeError(f"git commit failed: {_redact(commit.stderr, token)}")

    remote_has_branch = subprocess.run(
        ["git", "-C", repo_dir, "ls-remote", "--exit-code", authed_remote, branch],
        capture_output=True, text=True,
    ).returncode == 0
    if remote_has_branch:
        pull = subprocess.run(
            ["git", "-C", repo_dir, "pull", "--rebase", authed_remote, branch],
            capture_output=True, text=True,
        )
        if pull.returncode != 0:
            raise RuntimeError(f"git pull --rebase failed: {_redact(pull.stderr, token)}")

    push = subprocess.run(
        ["git", "-C", repo_dir, "push", authed_remote, branch],
        capture_output=True, text=True,
    )
    if push.returncode != 0:
        raise RuntimeError(f"git push failed: {_redact(push.stderr, token)}")

    return f"Pushed {len(paths)} path(s) to {branch}: {message}"


def _redact(text: str, token: str) -> str:
    return text.replace(token, "<redacted>") if token else text
