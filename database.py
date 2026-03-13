import sqlite3

conn = sqlite3.connect("smart_logistics.db")
cursor = conn.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS shipments(

shipment_id INTEGER PRIMARY KEY AUTOINCREMENT,
origin TEXT,
destination TEXT,
route TEXT,
distance REAL,
eta REAL,
risk REAL,
shipment_date TEXT

)

""")

conn.commit()
conn.close()

print("Database created successfully")