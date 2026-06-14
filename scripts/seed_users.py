import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DB_PATH
from app.database import Database
from app.palm_processor import PalmProcessor
from app.services.seed_users import seed_users_from_directory


def main():
    parser = argparse.ArgumentParser(description="Seed PalmGate users from named full-hand images.")
    parser.add_argument("seed_dir", nargs="?", default="seeds")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--replace-users", action="store_true")
    args = parser.parse_args()

    db = Database(args.db)
    palm_processor = PalmProcessor()
    try:
        summary = seed_users_from_directory(
            args.seed_dir,
            db,
            palm_processor,
            replace_users=args.replace_users,
        )
        for name in summary.created:
            print(f"CREATED {name}")
        for name in summary.skipped:
            print(f"SKIPPED {name}")
        for name, error in summary.failed.items():
            print(f"FAILED {name}: {error}")
        if summary.failed:
            raise SystemExit(1)
    finally:
        palm_processor.close()
        db.close()


if __name__ == "__main__":
    main()
