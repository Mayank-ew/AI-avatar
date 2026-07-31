"""
Phase 7 — Metadata store & storage layer.  docs/05 §0, docs/03 §0.

A pluggable `ProfileStore` behind which either backend sits, so the eventual prod swap never
touches calling code (docs/05 §0 POC note):
  - PostgresProfileStore  — the POC default (small hosted Postgres via a modal.Secret conn str).
  - ModalDictProfileStore — the Modal-native production path (one write, one read-by-key).

Access pattern is exactly one write at onboarding and one read-by-key at every generation.
`put` is an UPSERT — re-onboarding overwrites in place, never duplicates (docs/07, Phase 7).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import constants


@dataclass
class HostProfile:
    host_id: str
    voice_id: str
    base_video_path: str
    created_at: str                  # ISO-8601 string; caller supplies (see docs/05 §4 note)
    voice_character_hint: str | None = None
    # Reimagined studio portrait (reference_studio.py) that Function A (Wan2.2-S2V) animates.
    # This is the DEFAULT reference (used when no aspect-specific one is registered).
    reference_image_path: str | None = None
    # Optional per-aspect references, keyed by canonical ratio ("9:16"/"16:9"/"1:1"). Lets a host
    # register a reel-framed (waist-up, 9:16) portrait so generation doesn't crop a landscape
    # close-up. Generation picks references_by_ratio[target] if present, else reference_image_path.
    references_by_ratio: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HostProfile":
        return cls(
            host_id=d["host_id"],
            voice_id=d["voice_id"],
            base_video_path=d["base_video_path"],
            created_at=d["created_at"],
            voice_character_hint=d.get("voice_character_hint"),
            reference_image_path=d.get("reference_image_path"),
            references_by_ratio=d.get("references_by_ratio") or {},
        )


class ProfileStore:
    """Interface. get returns None on a clean miss (Phase 11 keys profile_lookup off this)."""

    def get(self, host_id: str) -> HostProfile | None:
        raise NotImplementedError

    def put(self, host_id: str, profile: HostProfile) -> None:
        raise NotImplementedError

    def delete(self, host_id: str) -> None:
        raise NotImplementedError

    def list_host_ids(self) -> list[str]:
        raise NotImplementedError


class PostgresProfileStore(ProfileStore):
    """
    POC store. Connection string comes from the `postgres-conn-string` Secret
    (POSTGRES_CONN_STRING). Table DDL is created on first connect (idempotent).
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS host_profiles (
        host_id              TEXT PRIMARY KEY,
        voice_id             TEXT NOT NULL,
        base_video_path      TEXT NOT NULL,
        voice_character_hint TEXT,
        created_at           TEXT NOT NULL,
        reference_image_path TEXT,
        references_by_ratio  TEXT
    );
    ALTER TABLE host_profiles ADD COLUMN IF NOT EXISTS reference_image_path TEXT;
    ALTER TABLE host_profiles ADD COLUMN IF NOT EXISTS references_by_ratio TEXT;
    """

    def __init__(self, conn_string: str | None = None):
        self._conn_string = conn_string or os.environ.get("POSTGRES_CONN_STRING")
        if not self._conn_string:
            raise RuntimeError(
                "POSTGRES_CONN_STRING not set (missing postgres-conn-string modal.Secret?)"
            )
        self._ensure_table()

    def _connect(self):
        import psycopg2

        return psycopg2.connect(self._conn_string)

    def _ensure_table(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(self._DDL)
            conn.commit()

    def get(self, host_id: str) -> HostProfile | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT host_id, voice_id, base_video_path, "
                "voice_character_hint, created_at, reference_image_path, references_by_ratio "
                "FROM host_profiles WHERE host_id = %s",
                (host_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return HostProfile(
            host_id=row[0],
            voice_id=row[1],
            base_video_path=row[2],
            voice_character_hint=row[3],
            created_at=row[4],
            reference_image_path=row[5],
            references_by_ratio=json.loads(row[6]) if row[6] else {},
        )

    def put(self, host_id: str, profile: HostProfile) -> None:
        # UPSERT — re-onboarding overwrites, never duplicates (Phase 7 acceptance).
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO host_profiles
                    (host_id, voice_id, base_video_path,
                     voice_character_hint, created_at, reference_image_path, references_by_ratio)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (host_id) DO UPDATE SET
                    voice_id = EXCLUDED.voice_id,
                    base_video_path = EXCLUDED.base_video_path,
                    voice_character_hint = EXCLUDED.voice_character_hint,
                    created_at = EXCLUDED.created_at,
                    reference_image_path = EXCLUDED.reference_image_path,
                    references_by_ratio = EXCLUDED.references_by_ratio
                """,
                (
                    profile.host_id,
                    profile.voice_id,
                    profile.base_video_path,
                    profile.voice_character_hint,
                    profile.created_at,
                    profile.reference_image_path,
                    json.dumps(profile.references_by_ratio or {}),
                ),
            )
            conn.commit()

    def delete(self, host_id: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM host_profiles WHERE host_id = %s", (host_id,))
            conn.commit()

    def list_host_ids(self) -> list[str]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT host_id FROM host_profiles ORDER BY created_at DESC")
            return [r[0] for r in cur.fetchall()]


class ModalDictProfileStore(ProfileStore):
    """
    Production Modal-native store. Lazily binds the `host-profiles` modal.Dict by name (avoids
    a circular import with app.py). Dict put is inherently an upsert by key.
    """

    def __init__(self):
        import modal

        self._dict = modal.Dict.from_name("host-profiles", create_if_missing=True)

    def get(self, host_id: str) -> HostProfile | None:
        raw = self._dict.get(host_id)
        if raw is None:
            return None
        return HostProfile.from_dict(raw)

    def put(self, host_id: str, profile: HostProfile) -> None:
        self._dict[host_id] = profile.to_dict()

    def delete(self, host_id: str) -> None:
        try:
            del self._dict[host_id]
        except KeyError:
            pass

    def list_host_ids(self) -> list[str]:
        try:
            return [str(k) for k in self._dict.keys()]
        except Exception:  # noqa: BLE001 — older modal.Dict may not support keys()
            return []


def get_store() -> ProfileStore:
    """Factory — picks the backend from constants.PROFILE_STORE_BACKEND."""
    backend = constants.PROFILE_STORE_BACKEND
    if backend == "postgres":
        return PostgresProfileStore()
    if backend == "modal_dict":
        return ModalDictProfileStore()
    raise ValueError(f"unknown PROFILE_STORE_BACKEND: {backend!r}")
