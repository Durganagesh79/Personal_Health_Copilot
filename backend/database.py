import json
import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "health_copilot.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    with get_connection() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                email TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                document_type TEXT DEFAULT 'Medical Report',
                report_date TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                extracted_text TEXT DEFAULT '',
                structured_json TEXT DEFAULT '{}',
                verification_status TEXT DEFAULT 'Needs review',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                language TEXT DEFAULT 'English',
                theme TEXT DEFAULT 'light',
                FOREIGN KEY(user_id) REFERENCES users(id)
                    ON DELETE CASCADE
            );
        """)


def row_to_dict(row):
    return dict(row) if row else None


def create_user(username, password_hash, display_name, email=""):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO users (username, password_hash, display_name, email)
            VALUES (?, ?, ?, ?)
            """,
            (username, password_hash, display_name, email),
        )
        user_id = cursor.lastrowid
        connection.execute(
            "INSERT OR IGNORE INTO user_settings (user_id) VALUES (?)",
            (user_id,),
        )
        return user_id


def get_user_by_username(username):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return row_to_dict(row)


def get_user_by_id(user_id):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return row_to_dict(row)


def update_profile(user_id, display_name, email):
    with get_connection() as connection:
        connection.execute(
            "UPDATE users SET display_name = ?, email = ? WHERE id = ?",
            (display_name, email, user_id),
        )


def save_report(
    user_id,
    filename,
    document_type,
    report_date,
    summary,
    extracted_text,
    structured,
    verification_status,
):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO reports (
                user_id, filename, document_type, report_date,
                summary, extracted_text, structured_json,
                verification_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                filename,
                document_type,
                report_date,
                summary,
                extracted_text,
                json.dumps(structured, ensure_ascii=False),
                verification_status,
            ),
        )
        return cursor.lastrowid


def get_reports(user_id):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM reports
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()

    reports = []
    for row in rows:
        report = dict(row)
        try:
            report["structured"] = json.loads(
                report.pop("structured_json", "{}") or "{}"
            )
        except (TypeError, json.JSONDecodeError):
            report["structured"] = {}
            report.pop("structured_json", None)
        reports.append(report)

    return reports


def get_report(user_id, report_id):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT * FROM reports
            WHERE user_id = ? AND id = ?
            """,
            (user_id, report_id),
        ).fetchone()

    if not row:
        return None

    report = dict(row)
    try:
        report["structured"] = json.loads(
            report.pop("structured_json", "{}") or "{}"
        )
    except (TypeError, json.JSONDecodeError):
        report["structured"] = {}
        report.pop("structured_json", None)

    return report


def get_report_count(user_id):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS total FROM reports WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return row["total"]


def get_settings(user_id):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM user_settings WHERE user_id = ?",
            (user_id,),
        ).fetchone()

    return row_to_dict(row) or {
        "language": "English",
        "theme": "light",
    }


def update_settings(user_id, language, theme):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO user_settings (user_id, language, theme)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                language = excluded.language,
                theme = excluded.theme
            """,
            (user_id, language, theme),
        )