"""Owner-only, crash-durable local records for clinical request history."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def _validate(metadata: os.stat_result, *, directory: bool) -> None:
    expected = 0o700 if directory else 0o600
    kind_ok = (
        stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    )
    if (
        not kind_ok
        or stat.S_IMODE(metadata.st_mode) != expected
        or metadata.st_uid != os.getuid()
    ):
        raise ValueError("clinical storage must be private and owned by this user")


def private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("clinical storage must not be a symlink")
    if not path.exists():
        if not path.parent.exists():
            private_directory(path.parent)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    _validate(path.lstat(), directory=True)


def _directory_fd(path: Path) -> int:
    private_directory(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _validate(os.fstat(descriptor), directory=True)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def read_private_bytes(path: Path) -> bytes:
    directory = _directory_fd(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as stream:
            _validate(os.fstat(stream.fileno()), directory=False)
            return stream.read()
    finally:
        os.close(directory)


def write_private_bytes(path: Path, data: bytes, *, replace: bool = False) -> None:
    with private_atomic_path(path, replace=replace) as temporary:
        temporary.write_bytes(data)


@contextmanager
def private_atomic_path(path: Path, *, replace: bool = False) -> Iterator[Path]:
    """Install a private file only after its writer completes and fsync succeeds."""
    directory = _directory_fd(path.parent)
    temporary = f".{path.name}.{uuid4().hex}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        os.close(descriptor)
        yield path.parent / temporary
        descriptor = os.open(temporary, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        try:
            _validate(os.fstat(descriptor), directory=False)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if replace:
            try:
                _validate(
                    os.stat(path.name, dir_fd=directory, follow_symlinks=False),
                    directory=False,
                )
            except FileNotFoundError:
                pass
            os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        else:
            os.link(
                temporary,
                path.name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        os.fsync(directory)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
            os.fsync(directory)
        except FileNotFoundError:
            pass
        os.close(directory)


def unlink_private(path: Path) -> None:
    directory = _directory_fd(path.parent)
    try:
        os.unlink(path.name, dir_fd=directory)
        os.fsync(directory)
    finally:
        os.close(directory)
