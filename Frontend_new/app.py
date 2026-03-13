from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
from email_validator import validate_email, EmailNotValidError
import random
import string
import math
import requests
from datetime import datetime, timedelta
import config

app = Flask(__name__)
app.secret_key = "smart_logistics_secret_2025"

# MongoDB Config
app.config["MONGO_URI"] = getattr(config, "MONGO_URI", "mongodb://localhost:27017/")
app.config["MONGO_DB"] = getattr(config, "MONGO_DB", "smart_logistics")

client = MongoClient(app.config["MONGO_URI"])
db = client[app.config["MONGO_DB"]]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def geocode_location(place_name):
    """Convert a place name to (lat, lon) using Nominatim."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place_name, "format": "json", "limit": 1}
        headers = {"User-Agent": "SmartLogisticsAI/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None, None


def compute_leg_risk(distance_km, eta_hours, sla_deadline_str, created_at_str=None):
    """
    Compute SLA risk % and delay risk % for a single leg.
    sla_deadline_str: ISO format string of SLA deadline
    Returns: (sla_risk, delay_risk, recommendation)
    """
    now = datetime.now()

    # Parse SLA deadline
    try:
        if len(sla_deadline_str) > 16:
            sla_dt = datetime.fromisoformat(sla_deadline_str)
        else:
            sla_dt = datetime.strptime(sla_deadline_str, "%Y-%m-%d %H:%M")
    except Exception:
        sla_dt = now + timedelta(hours=72)

    hours_to_sla = (sla_dt - now).total_seconds() / 3600

    # SLA Risk: how much of the SLA window will ETA consume?
    if hours_to_sla <= 0:
        sla_risk = 100.0
    else:
        ratio = eta_hours / hours_to_sla
        sla_risk = min(round(ratio * 100, 1), 100.0)

    # Delay Risk: deterministic simulation based on distance buckets
    # Base risk increases with longer hauls + simulated traffic/weather factor
    if distance_km < 200:
        base_delay = 15.0
    elif distance_km < 500:
        base_delay = 30.0
    elif distance_km < 1000:
        base_delay = 45.0
    elif distance_km < 2000:
        base_delay = 60.0
    else:
        base_delay = 75.0

    # Simulated external factors (deterministic but varied based on distance)
    weather_factor = (distance_km % 37) / 37 * 15   # 0–15%
    traffic_factor = (distance_km % 23) / 23 * 10   # 0–10%
    delay_risk = min(round(base_delay + weather_factor + traffic_factor, 1), 100.0)

    # Recommendation
    combined = (sla_risk * 0.6) + (delay_risk * 0.4)
    if combined >= 70:
        rec = "Reroute shipment via alternative path"
    elif combined >= 50:
        rec = "Assign alternative carrier"
    elif combined >= 30:
        rec = "Send pre-alert to recipient"
    else:
        rec = "On track - no action required"

    return sla_risk, delay_risk, rec


def generate_employee_id():
    """Generate a unique employee ID like EMP-4827."""
    return "EMP-" + "".join(random.choices(string.digits, k=4))


def generate_shipment_ref():
    """Generate a shipment reference like SHP-2025-AB3X."""
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"SHP-{datetime.now().year}-{suffix}"


# =========================================================
# LANDING PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# ADMIN REGISTER (direct — no OTP)
# =========================================================

@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    confirm = request.form.get("confirm_password", "").strip()

    if not username or not email or not password:
        return render_template("index.html", error="All fields are required.", show_register=True)

    if password != confirm:
        return render_template("index.html", error="Passwords do not match.", show_register=True)

    try:
        validate_email(email)
    except EmailNotValidError:
        return render_template("index.html", error="Invalid email format.", show_register=True)

    if db.admins.find_one({"email": email}):
        return render_template("index.html", error="An admin with this email already exists.", show_register=True)

    db.admins.insert_one({
        "username": username,
        "email": email,
        "password": password,
        "created_at": datetime.now()
    })

    return render_template("index.html", success="Registration successful. Please login as Admin.")


# =========================================================
# ADMIN LOGIN
# =========================================================

@app.route("/login/admin", methods=["POST"])
def login_admin():
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()

    user = db.admins.find_one({"email": email, "password": password})

    if user:
        session["admin_id"] = str(user["_id"])
        session["admin_name"] = user["username"]
        session["role"] = "admin"
        return redirect(url_for("admin_dashboard"))

    return render_template("index.html", error="Invalid admin credentials.", show_login=True, role="admin")


# =========================================================
# EMPLOYEE LOGIN
# =========================================================

@app.route("/login/employee", methods=["POST"])
def login_employee():
    employee_id = request.form.get("employee_id", "").strip()
    password = request.form.get("password", "").strip()

    emp = db.employees.find_one({"employee_id": employee_id, "password": password})

    if emp:
        session["employee_db_id"] = str(emp["_id"])
        session["employee_name"] = emp["name"]
        session["employee_id"] = emp["employee_id"]
        session["role"] = "employee"
        return redirect(url_for("employee_dashboard"))

    return render_template("index.html", error="Invalid Employee ID or password.", show_login=True, role="employee")


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route("/admin/dashboard")
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect(url_for("home"))
    admin_id = session["admin_id"]
    
    employees_cursor = db.employees.find({"admin_id": admin_id}).sort("created_at", -1)
    employees = []
    
    for emp in employees_cursor:
        employees.append([
            emp.get("employee_id", ""),
            emp.get("name", ""),
            emp.get("email", ""),
            emp.get("created_at")
        ])
        
    today = datetime.now().strftime("%Y-%m-%d")
    return render_template("admin_dashboard.html", employees=employees, today=today)


@app.route("/admin/add_employee", methods=["POST"])
def add_employee():
    if session.get("role") != "admin":
        return redirect(url_for("home"))

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    admin_id = session["admin_id"]

    if not name or not email or not password:
        return redirect(url_for("admin_dashboard"))

    while True:
        emp_id = generate_employee_id()
        if not db.employees.find_one({"employee_id": emp_id}):
            break

    db.employees.insert_one({
        "employee_id": emp_id,
        "name": name,
        "email": email,
        "password": password,
        "admin_id": admin_id,
        "created_at": datetime.now()
    })

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/remove_employee", methods=["POST"])
def remove_employee():
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    emp_id = request.form.get("employee_id", "").strip()
    
    db.employees.delete_one({"employee_id": emp_id, "admin_id": session["admin_id"]})
    db.shipments.delete_many({"employee_id": emp_id})

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/shipments")
def admin_shipments():
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    emp_filter = request.args.get("employee_id", "")
    admin_id = session["admin_id"]

    # Get employees under this admin for the filter dropdown
    employees_cursor = list(db.employees.find({"admin_id": admin_id}))
    employees = [{"employee_id": e["employee_id"], "name": e["name"]} for e in employees_cursor]
    
    emp_mapping = {e["employee_id"]: e["name"] for e in employees_cursor}

    # Build DB Query
    query = {"date_of_shipment": date_str}
    if emp_filter:
        if emp_filter in emp_mapping:
            query["employee_id"] = emp_filter
        else:
            query["employee_id"] = "NONE" # invalid, won't match
    else:
        query["employee_id"] = {"$in": list(emp_mapping.keys())}

    shipments_cursor = db.shipments.find(query).sort("created_at", -1)
    
    shipments = []
    for s in shipments_cursor:
        shipments.append({
            "shipment_ref": s.get("shipment_ref", ""),
            "employee_id": s.get("employee_id", ""),
            "employee_name": emp_mapping.get(s.get("employee_id"), "Unknown"),
            "status": s.get("status", "In Transit"),
            "sla_deadline": str(s.get("sla_deadline", "")),
            "overall_risk_score": float(s.get("overall_risk_score", 0)),
            "recommendation": s.get("recommendation", ""),
            "created_at": str(s.get("created_at", ""))
        })

    return jsonify({"shipments": shipments, "employees": employees})


# =========================================================
# EMPLOYEE DASHBOARD
# =========================================================

@app.route("/employee/dashboard")
def employee_dashboard():
    if session.get("role") != "employee":
        return redirect(url_for("home"))
    return render_template("employee_dashboard.html")


@app.route("/employee/add_shipment", methods=["POST"])
def add_shipment():
    if session.get("role") != "employee":
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json()
    sla_deadline = data.get("sla_deadline")
    legs_input = data.get("legs", [])
    employee_id = session["employee_id"]

    if not sla_deadline or len(legs_input) < 1:
        return jsonify({"error": "SLA deadline and at least one leg are required."}), 400

    shipment_ref = generate_shipment_ref()
    today = datetime.now().strftime("%Y-%m-%d")

    computed_legs = []
    total_eta = 0
    max_risk = 0

    for idx, leg in enumerate(legs_input):
        origin = leg.get("origin", "").strip()
        destination = leg.get("destination", "").strip()

        if not origin or not destination:
            continue

        olat, olon = geocode_location(origin)
        dlat, dlon = geocode_location(destination)

        if olat is None or dlat is None:
            distance_km = 500.0  # fallback
            olat, olon, dlat, dlon = 20.5, 78.9, 22.5, 88.3
        else:
            distance_km = haversine(olat, olon, dlat, dlon)

        avg_speed_kmh = 60.0
        eta_hours = round(distance_km / avg_speed_kmh, 2)
        total_eta += eta_hours

        sla_risk, delay_risk, leg_rec = compute_leg_risk(distance_km, eta_hours, sla_deadline)

        combined = (sla_risk * 0.6) + (delay_risk * 0.4)
        if combined > max_risk:
            max_risk = combined

        computed_legs.append({
            "sequence": idx + 1,
            "origin": origin,
            "destination": destination,
            "origin_lat": olat,
            "origin_lon": olon,
            "dest_lat": dlat,
            "dest_lon": dlon,
            "distance_km": round(distance_km, 2),
            "eta_hours": eta_hours,
            "sla_risk": sla_risk,
            "delay_risk": delay_risk,
            "leg_recommendation": leg_rec
        })

    if not computed_legs:
        return jsonify({"error": "No valid legs could be processed."}), 400

    overall_risk = round(max_risk, 2)
    if overall_risk >= 70:
        overall_rec = "Reroute shipment via alternative path"
    elif overall_risk >= 50:
        overall_rec = "Assign alternative carrier"
    elif overall_risk >= 30:
        overall_rec = "Send pre-alert to recipient"
    else:
        overall_rec = "On track - no action required"

    shipment_doc = {
        "shipment_ref": shipment_ref,
        "employee_id": employee_id,
        "sla_deadline": str(sla_deadline),
        "status": "In Transit",
        "overall_risk_score": overall_risk,
        "recommendation": overall_rec,
        "date_of_shipment": today,
        "created_at": datetime.now(),
        "legs": computed_legs
    }
    db.shipments.insert_one(shipment_doc)

    return jsonify({
        "success": True,
        "shipment_ref": shipment_ref,
        "overall_risk_score": overall_risk,
        "recommendation": overall_rec,
        "legs": computed_legs
    })


@app.route("/employee/shipments")
def employee_shipments():
    if session.get("role") != "employee":
        return jsonify({"error": "Unauthorized"}), 403

    employee_id = session["employee_id"]
    page = int(request.args.get("page", 1))
    per_page = 10
    offset = (page - 1) * per_page

    total = db.shipments.count_documents({"employee_id": employee_id})
    shipments_cursor = db.shipments.find({"employee_id": employee_id}).sort("created_at", -1).skip(offset).limit(per_page)

    shipments = []
    for s in shipments_cursor:
        shipments.append({
            "shipment_ref": s.get("shipment_ref", ""),
            "sla_deadline": str(s.get("sla_deadline", "")),
            "status": s.get("status", "In Transit"),
            "overall_risk_score": float(s.get("overall_risk_score", 0)),
            "recommendation": s.get("recommendation", ""),
            "date_of_shipment": str(s.get("date_of_shipment", "")),
            "created_at": str(s.get("created_at", "")),
            "id": str(s["_id"])
        })

    return jsonify({"shipments": shipments, "total": total, "page": page, "per_page": per_page})


@app.route("/employee/shipment/<string:shipment_id>")
def shipment_detail(shipment_id):
    if session.get("role") != "employee":
        return jsonify({"error": "Unauthorized"}), 403

    employee_id = session["employee_id"]
    
    try:
        s = db.shipments.find_one({"_id": ObjectId(shipment_id), "employee_id": employee_id})
    except Exception:
        s = None
        
    if not s:
        return jsonify({"error": "Not found"}), 404

    legs = []
    for leg in s.get("legs", []):
        legs.append({
            "sequence": leg.get("sequence"),
            "origin": leg.get("origin"),
            "destination": leg.get("destination"),
            "origin_lat": leg.get("origin_lat"),
            "origin_lon": leg.get("origin_lon"),
            "dest_lat": leg.get("dest_lat"),
            "dest_lon": leg.get("dest_lon"),
            "distance_km": leg.get("distance_km"),
            "eta_hours": leg.get("eta_hours"),
            "sla_risk": leg.get("sla_risk"),
            "delay_risk": leg.get("delay_risk"),
            "leg_recommendation": leg.get("leg_recommendation")
        })

    return jsonify({
        "shipment_ref": s.get("shipment_ref", ""),
        "sla_deadline": str(s.get("sla_deadline", "")),
        "status": s.get("status", "In Transit"),
        "overall_risk_score": float(s.get("overall_risk_score", 0)),
        "recommendation": s.get("recommendation", ""),
        "date_of_shipment": str(s.get("date_of_shipment", "")),
        "created_at": str(s.get("created_at", "")),
        "legs": legs
    })


if __name__ == "__main__":
    app.run(debug=True)