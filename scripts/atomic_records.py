"""Small JSON records with retries for temporary Windows/OneDrive file locks."""
import json
import time
import uuid
from pathlib import Path


def write_json(path, value, *, required=True):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        for attempt in range(10):
            try:
                temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
                temporary.replace(path)
                return True
            except PermissionError:
                if attempt == 9:
                    if required:
                        raise
                    return False
                time.sleep(.05 * (attempt + 1))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
