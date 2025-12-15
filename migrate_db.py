
import sqlite3

db_file = "dados_mercado.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE trained_models ADD COLUMN mape FLOAT")
    conn.commit()
    print("Column MAPE added successfully")
except Exception as e:
    print(f"Error adding column: {e}")
    # Maybe it already exists?
    if "duplicate column" in str(e):
        print("Column already exists")

conn.close()
