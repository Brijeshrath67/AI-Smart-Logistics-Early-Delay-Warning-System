import sqlite3
import json
from datetime import datetime

def save_shipment(origin, destination, route, distance, eta, risk):

    conn = sqlite3.connect("smart_logistics.db")
    cursor = conn.cursor()

    cursor.execute("""

    INSERT INTO shipments
    (origin, destination, route, distance, eta, risk, shipment_date)

    VALUES (?, ?, ?, ?, ?, ?, ?)

    """, (

        origin,
        destination,
        str(route),
        float(distance),
        float(eta),
        float(risk),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ))

    conn.commit()
    conn.close()

    print("Shipment saved successfully")