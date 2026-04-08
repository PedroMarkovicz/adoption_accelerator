"""
In-memory job store for async prediction tracking.

Stores prediction results keyed by session_id with thread-safe access
and automatic TTL-based cleanup of stale entries.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

# TTL for job entries (seconds)
JOB_TTL_SECONDS = 600  # 10 minutes
CLEANUP_INTERVAL_SECONDS = 60  # Run cleanup every 60 seconds


@dataclass
class JobRecord:
    """Holds the state of a single prediction job."""

    status: Literal["pending", "phase1_ready", "complete", "error"] = "pending"
    phase1_result: dict[str, Any] | None = None
    phase2_result: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)


class JobStore:
    """Thread-safe in-memory store for prediction jobs.

    All public methods acquire the internal lock before reading or
    writing, keeping critical sections small.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._cleanup_running = False

    # -- Public API --------------------------------------------------------

    def create(self, session_id: str) -> None:
        """Register a new pending job."""
        with self._lock:
            self._jobs[session_id] = JobRecord()

    def set_phase1_ready(
        self,
        session_id: str,
        phase1: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark Phase 1 as complete and store its results.

        Creates the job entry if it does not already exist (upsert).
        """
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                job = JobRecord()
                self._jobs[session_id] = job
            job.status = "phase1_ready"
            job.phase1_result = phase1
            job.metadata = metadata

    def set_complete(
        self,
        session_id: str,
        phase1: dict[str, Any] | None,
        phase2: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark the job as fully complete with both phase results.

        Creates the job entry if it does not already exist (upsert).
        If phase1 or metadata is None, the existing stored value is preserved
        (allows Phase 2 to complete without re-supplying Phase 1 data).
        """
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                job = JobRecord()
                self._jobs[session_id] = job
            job.status = "complete"
            if phase1 is not None:
                job.phase1_result = phase1
            if phase2 is not None:
                job.phase2_result = phase2
            if metadata is not None:
                job.metadata = metadata

    def set_error(self, session_id: str, error_message: str) -> None:
        """Mark the job as failed. Creates the entry if needed (upsert)."""
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                job = JobRecord()
                self._jobs[session_id] = job
            job.status = "error"
            job.error = error_message

    def get(self, session_id: str) -> JobRecord | None:
        """Return a snapshot of the job record (or None if not found)."""
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                return None
            # Return a shallow copy so callers cannot mutate store state
            return JobRecord(
                status=job.status,
                phase1_result=job.phase1_result,
                phase2_result=job.phase2_result,
                metadata=job.metadata,
                error=job.error,
                created_at=job.created_at,
            )

    def get_all(self) -> list[tuple[str, JobRecord]]:
        """Return a list of (session_id, JobRecord snapshot) pairs."""
        with self._lock:
            return [
                (
                    sid,
                    JobRecord(
                        status=job.status,
                        phase1_result=job.phase1_result,
                        phase2_result=job.phase2_result,
                        metadata=job.metadata,
                        error=job.error,
                        created_at=job.created_at,
                    ),
                )
                for sid, job in self._jobs.items()
            ]

    def cleanup_expired(self) -> int:
        """Remove entries older than JOB_TTL_SECONDS. Returns count removed."""
        cutoff = time.time() - JOB_TTL_SECONDS
        with self._lock:
            expired = [
                sid for sid, job in self._jobs.items() if job.created_at < cutoff
            ]
            for sid in expired:
                del self._jobs[sid]
        return len(expired)

    # -- Background cleanup ------------------------------------------------

    def start_cleanup_loop(self) -> threading.Thread:
        """Start a daemon thread that periodically evicts stale entries."""
        self._cleanup_running = True
        t = threading.Thread(target=self._cleanup_worker, daemon=True)
        t.start()
        return t

    def stop_cleanup_loop(self) -> None:
        """Signal the cleanup thread to stop."""
        self._cleanup_running = False

    def _cleanup_worker(self) -> None:
        """Runs in a background thread; sleeps between cleanup passes."""
        while self._cleanup_running:
            time.sleep(CLEANUP_INTERVAL_SECONDS)
            if not self._cleanup_running:
                break
            removed = self.cleanup_expired()
            if removed > 0:
                import logging
                logging.getLogger(__name__).info(
                    "Job store cleanup: evicted %d expired entries", removed
                )


# Module-level singleton
job_store = JobStore()
