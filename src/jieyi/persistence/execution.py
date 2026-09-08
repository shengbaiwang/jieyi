"""Process-safe document execution ownership, released by the OS on process exit."""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path


class DocumentBusyError(ValueError):
    pass


class DocumentRunLock:
    def __init__(self, database: str, document_id: str):
        # Resolve aliases so two servers using a symlink share the same lock.
        database_path = Path(database).expanduser().resolve()
        directory = database_path.with_name(database_path.name + ".runs")
        directory.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(document_id.encode()).hexdigest()
        self._file = (directory / name).open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self._file.write(b"0")
                self._file.flush()
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._file.close()
            self._file = None
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            raise DocumentBusyError("这本书已有翻译任务正在执行，请暂停或等待完成后再启动。") from exc

    def release(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        # Never unlink the lock file: waiters may still hold its inode.

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
