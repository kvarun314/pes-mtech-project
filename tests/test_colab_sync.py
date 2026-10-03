import subprocess
from unittest.mock import patch

import pytest

from agentic_sentiment.colab_sync import push_repo_changes


def _git(*args, cwd):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


def _init_repo_with_remote(tmp_path, name="work"):
    """A real local repo with a real local bare 'remote' -- push mechanics
    (add/commit/pull --rebase/push) are exercised against actual git, not
    mocked."""
    bare = tmp_path / "remote.git"
    work = tmp_path / name
    if not bare.exists():
        _git("init", "--bare", str(bare), cwd=tmp_path)
    work.mkdir()
    _git("init", cwd=work)
    (work / "README.md").write_text("initial\n")
    _git("add", "README.md", cwd=work)
    _git("-c", "user.email=t@t.com", "-c", "user.name=t", "commit", "-m", "initial", cwd=work)
    _git("branch", "-M", "main", cwd=work)
    return work, bare


def test_push_commits_and_pushes_to_remote(tmp_path):
    work, bare = _init_repo_with_remote(tmp_path)
    (work / "RESULTS.md").write_text("# Results\n")

    result = push_repo_changes(
        repo_dir=str(work), paths=["RESULTS.md"], message="add results",
        token="unused-for-file-remote", remote=f"file://{bare}", branch="main",
    )

    assert "Pushed" in result
    log = subprocess.run(["git", "-C", str(bare), "log", "--oneline"], capture_output=True, text=True)
    assert "add results" in log.stdout


def test_push_skips_commit_when_nothing_staged(tmp_path):
    work, bare = _init_repo_with_remote(tmp_path)
    (work / "RESULTS.md").write_text("# Results\n")
    push_repo_changes(repo_dir=str(work), paths=["RESULTS.md"], message="first",
                       token="x", remote=f"file://{bare}")

    # Re-run with the exact same (already-committed) content: nothing new to add.
    result = push_repo_changes(repo_dir=str(work), paths=["RESULTS.md"], message="first again",
                                token="x", remote=f"file://{bare}")
    assert "Nothing to commit" in result

    log = subprocess.run(["git", "-C", str(bare), "log", "--oneline"], capture_output=True, text=True)
    assert log.stdout.count("first") == 1  # not duplicated


def test_push_rebases_onto_changes_pushed_by_another_clone(tmp_path):
    # work1 and work2 both clone the same initial state; work1 pushes first,
    # work2 pushes a different file afterward without ever pulling --
    # push_repo_changes must rebase onto work1's push rather than failing
    # non-fast-forward.
    work1, bare = _init_repo_with_remote(tmp_path, name="work1")
    _git("clone", str(bare), str(tmp_path / "work2"), cwd=tmp_path)
    work2 = tmp_path / "work2"
    _git("branch", "-M", "main", cwd=work2)

    (work1 / "a.txt").write_text("from work1\n")
    push_repo_changes(repo_dir=str(work1), paths=["a.txt"], message="from work1",
                       token="x", remote=f"file://{bare}")

    (work2 / "b.txt").write_text("from work2\n")
    result = push_repo_changes(repo_dir=str(work2), paths=["b.txt"], message="from work2",
                                token="x", remote=f"file://{bare}")

    assert "Pushed" in result
    log = subprocess.run(["git", "-C", str(bare), "log", "--oneline"], capture_output=True, text=True)
    assert "from work1" in log.stdout
    assert "from work2" in log.stdout


def test_push_raises_on_empty_token(tmp_path):
    work, bare = _init_repo_with_remote(tmp_path)
    with pytest.raises(RuntimeError, match="empty token"):
        push_repo_changes(repo_dir=str(work), paths=["README.md"], message="x",
                           token="", remote=f"file://{bare}")


def test_push_failure_before_any_git_call_never_leaks_the_token():
    fake_token = "SUPER-SECRET-TOKEN-999"
    with pytest.raises(RuntimeError) as exc_info:
        push_repo_changes(
            repo_dir="/nonexistent/repo/path", paths=["x.txt"], message="x",
            token=fake_token, remote="https://127.0.0.1:1/no-such-repo.git",
        )
    assert fake_token not in str(exc_info.value)


def test_redact_strips_the_token_from_subprocess_stderr(tmp_path):
    # push_repo_changes's redaction logic (_redact) is exercised directly
    # against a forced failure whose stderr is made to contain the token --
    # real git strips auth info from its own error URLs, so a live failing
    # push can't be used to prove redaction actually runs.
    work, bare = _init_repo_with_remote(tmp_path)
    (work / "c.txt").write_text("x\n")
    fake_token = "SUPER-SECRET-TOKEN-999"

    leaking_result = subprocess.CompletedProcess(
        args=["git", "push"], returncode=1,
        stdout="", stderr=f"fatal: could not push to https://{fake_token}@example.com/repo.git",
    )
    with patch("subprocess.run", return_value=leaking_result):
        with pytest.raises(RuntimeError) as exc_info:
            push_repo_changes(repo_dir=str(work), paths=["c.txt"], message="x",
                               token=fake_token, remote=f"file://{bare}")

    assert fake_token not in str(exc_info.value)
    assert "<redacted>" in str(exc_info.value)
