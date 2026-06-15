import sqlite3
import numpy as np
from pathlib import Path


class Database:
    def __init__(self, db_path: "str | Path"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Enable foreign-key enforcement so ON DELETE CASCADE works for
        # user_embeddings when a user row is deleted.
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                nim         TEXT NOT NULL UNIQUE,
                name        TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            -- Individual per-capture embeddings for multi-embedding matching.
            -- Stored alongside the averaged embedding in users.embedding so that
            -- recognition can match against the closest single capture rather than
            -- a blended average, improving cross-device / cross-lighting recall.
            CREATE TABLE IF NOT EXISTS user_embeddings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                embedding   BLOB NOT NULL,
                hand        TEXT NOT NULL DEFAULT 'unknown',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS access_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER,
                matched_name    TEXT NOT NULL,
                status          TEXT NOT NULL,
                similarity      REAL NOT NULL,
                duration_ms     INTEGER,
                description     TEXT,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS device_status (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                worker_state TEXT NOT NULL,
                camera_connected INTEGER NOT NULL,
                last_error TEXT,
                fps REAL,
                last_inference_ms REAL,
                last_recognition_at TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._ensure_user_nim_column()
        self._ensure_user_embedding_metadata_columns()
        self._ensure_access_log_metadata_columns()
        self.conn.commit()

    def _ensure_user_nim_column(self):
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(users)")}
        if "nim" not in columns:
            self.conn.execute("ALTER TABLE users ADD COLUMN nim TEXT")
            self.conn.execute(
                "UPDATE users SET nim = 'legacy-' || id WHERE nim IS NULL OR TRIM(nim) = ''"
            )
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_nim ON users(nim)")

    def _ensure_user_embedding_metadata_columns(self):
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(user_embeddings)")}
        if "hand" not in columns:
            self.conn.execute("ALTER TABLE user_embeddings ADD COLUMN hand TEXT NOT NULL DEFAULT 'unknown'")

    def _ensure_access_log_metadata_columns(self):
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(access_logs)")}
        if "duration_ms" not in columns:
            self.conn.execute("ALTER TABLE access_logs ADD COLUMN duration_ms INTEGER")
        if "description" not in columns:
            self.conn.execute("ALTER TABLE access_logs ADD COLUMN description TEXT")

    def add_user(
        self,
        name: str,
        embedding: np.ndarray,
        *,
        nim: str,
        individual_embeddings: "list[np.ndarray] | None" = None,
        embedding_hands: "list[str] | None" = None,
    ) -> int:
        clean_nim = nim.strip()
        clean_name = name.strip()
        if not clean_nim:
            raise ValueError("NIM is required")
        if not clean_name:
            raise ValueError("Name is required")

        individual_rows = []
        if individual_embeddings:
            hands = embedding_hands or ["unknown"] * len(individual_embeddings)
            if len(hands) != len(individual_embeddings):
                raise ValueError("embedding_hands must match individual_embeddings length")
            individual_rows = [(e.astype(np.float32).tobytes(), hand) for e, hand in zip(individual_embeddings, hands)]

        try:
            cursor = self.conn.execute(
                "INSERT INTO users (nim, name, embedding) VALUES (?, ?, ?)",
                (clean_nim, clean_name, embedding.astype(np.float32).tobytes()),
            )
            user_id = cursor.lastrowid
            if individual_rows:
                self.conn.executemany(
                    "INSERT INTO user_embeddings (user_id, embedding, hand) VALUES (?, ?, ?)",
                    [(user_id, blob, hand) for blob, hand in individual_rows],
                )
            self.conn.commit()
            return user_id
        except sqlite3.IntegrityError as exc:
            self.conn.rollback()
            if "users.nim" in str(exc) or "idx_users_nim" in str(exc) or "UNIQUE" in str(exc):
                raise ValueError("NIM already exists") from exc
            raise
        except Exception:
            self.conn.rollback()
            raise

    def get_all_users(self) -> list:
        rows = self.conn.execute(
            "SELECT id, nim, name, created_at FROM users ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_embeddings(self) -> list:
        """Return one entry per stored embedding.

        For users registered after multi-embedding support was added, each of
        their individual capture embeddings is returned as a separate entry
        (all sharing the same user_id/name).  For legacy users who only have
        the averaged embedding, that single embedding is returned instead.
        """
        users = self.conn.execute(
            "SELECT id, name, embedding FROM users ORDER BY id"
        ).fetchall()

        indiv_rows = self.conn.execute(
            "SELECT user_id, embedding, hand FROM user_embeddings ORDER BY user_id, id"
        ).fetchall()

        indiv_map: dict[int, list] = {}
        for row in indiv_rows:
            indiv_map.setdefault(row["user_id"], []).append({
                "embedding": np.frombuffer(row["embedding"], dtype=np.float32).copy(),
                "hand": row["hand"],
            })

        result = []
        for u in users:
            uid, name = u["id"], u["name"]
            if uid in indiv_map:
                for item in indiv_map[uid]:
                    result.append({
                        "id": uid,
                        "name": name,
                        "embedding": item["embedding"],
                        "hand": item["hand"],
                    })
            else:
                result.append({
                    "id": uid,
                    "name": name,
                    "embedding": np.frombuffer(u["embedding"], dtype=np.float32).copy(),
                    "hand": "unknown",
                })
        return result

    def delete_user(self, user_id: int) -> bool:
        # Preserve historical access logs when a user is removed. Existing log
        # rows keep their matched_name/status/similarity, but their foreign key
        # is detached so the user row can be deleted safely.
        self.conn.execute(
            "UPDATE access_logs SET user_id = NULL WHERE user_id = ?",
            (user_id,),
        )
        cursor = self.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def add_access_log(
        self,
        user_id,
        matched_name: str,
        status: str,
        similarity: float,
        duration_ms: int | None = None,
        description: str | None = None,
    ):
        self.conn.execute(
            """
            INSERT INTO access_logs (
                user_id, matched_name, status, similarity, duration_ms, description
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, matched_name, status, similarity, duration_ms, description),
        )
        self.conn.commit()

    def get_access_logs(self, limit: int = 20, offset: int = 0) -> list:
        rows = self.conn.execute(
            "SELECT id, user_id, matched_name, status, similarity, duration_ms, description, timestamp "
            "FROM access_logs ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_access_logs(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM access_logs"
        ).fetchone()[0]

    def upsert_device_status(
        self,
        *,
        worker_state: str,
        camera_connected: bool,
        last_error: str | None,
        fps: float | None,
        last_inference_ms: float | None,
        last_recognition_at: str | None = None,
    ):
        self.conn.execute(
            """
            INSERT INTO device_status (
                id, worker_state, camera_connected, last_error, fps, last_inference_ms, last_recognition_at, updated_at
            ) VALUES (1, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                worker_state = excluded.worker_state,
                camera_connected = excluded.camera_connected,
                last_error = excluded.last_error,
                fps = excluded.fps,
                last_inference_ms = excluded.last_inference_ms,
                last_recognition_at = excluded.last_recognition_at,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                worker_state,
                int(camera_connected),
                last_error,
                fps,
                last_inference_ms,
                last_recognition_at,
            ),
        )
        self.conn.commit()

    def get_device_status(self) -> dict | None:
        row = self.conn.execute(
            "SELECT worker_state, camera_connected, last_error, fps, last_inference_ms, last_recognition_at, updated_at FROM device_status WHERE id = 1"
        ).fetchone()
        return dict(row) if row else None

    def close(self):
        self.conn.close()
