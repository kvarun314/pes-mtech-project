import subprocess

import pytest

from agentic_sentiment.colab_sync import push_repo_changes


def _git(*args, cwd):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


def _init_repo_with_remote(tmp_path):
    """A real local repo with a real local bare 'remote' -- push mechanics
    (add/commit/push) are exercised against actual git, not mocked."""
    bare = tmp_path / "remote.git"
    work = tmp_path / "work"
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


def test_push_raises_on_empty_token(tmp_path):
    work, bare = _init_repo_with_remote(tmp_path)
    with pytest.raises(RuntimeError, match="empty token"):
        push_repo_changes(repo_dir=str(work), paths=["README.md"], message="x",
                           token="", remote=f"file://{bare}")


def test_push_failure_never_leaks_the_token():
    fake_token = "SUPER-SECRET-TOKEN-999"
    with pytest.raises(RuntimeError) as exc_info:
        push_repo_changes(
            repo_dir="/nonexistent/repo/path", paths=["x.txt"], message="x",
            token=fake_token, remote="https://127.0.0.1:1/no-such-repo.git",
        )
    assert fake_token not in str(exc_info.value)


def test_push_redacts_token_from_push_failure(tmp_path):
    work, _bare = _init_repo_with_remote(tmp_path)
    fake_token = "SUPER-SECRET-TOKEN-999"
    with pytest.raises(RuntimeError) as exc_info:
        push_repo_changes(
            repo_dir=str(work), paths=["README.md"], message="x",
            token=fake_token, remote="https://127.0.0.1:1/no-such-repo.git",
        )
    assert fake_token not in str(exc_info.value)
    assert "git push failed" in str(exc_info.value)
