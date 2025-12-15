
import os
import sys
# Add current dir to path to find app
sys.path.insert(0, ".")

from app.database import init_db

db_file = "dados_mercado.db"

if os.path.exists(db_file):
    try:
        os.remove(db_file)
        print(f"{db_file} deleted")
    except Exception as e:
        print(f"Error deleting {db_file}: {e}")
        sys.exit(1)
else:
    print(f"{db_file} not found")

print("Initializing DB...")
init_db()
print("DB initialized")
