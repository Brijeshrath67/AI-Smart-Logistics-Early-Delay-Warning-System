from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
from email_validator import validate_email, EmailNotValidError
import random
import string
import folium
import math
import requests
from datetime import datetime, timedelta
import numpy as np
import config
import model_training

app = Flask(__name__)
app.secret_key = "smart_logistics_secret_2025"

# MongoDB Config
app.config["MONGO_URI"] = getattr(config, "MONGO_URI", "mongodb://localhost:27017/")
app.config["MONGO_DB"]  = getattr(config, "MONGO_DB", "smart_logistics")

client = MongoClient(app.config["MONGO_URI"])
db = client[app.config["MONGO_DB"]]


# =========================================================
# MODEL INITIALISATION (at startup, not on every request)
# =========================================================

print("[app] Training / loading XGBoost model …")
_model_data = model_training.train_model()

MODEL        = _model_data["model"]
LE_ORIGIN    = _model_data["le_origin"]
LE_DEST      = _model_data["le_dest"]
LE_WEATHER   = _model_data["le_weather"]
LE_TRAFFIC   = _model_data["le_traffic"]
LE_CARRIER   = _model_data["le_carrier"]
LE_PORT      = _model_data.get("le_port")   # may be None if not categorical
MODEL_META   = {
    "accuracy":            _model_data["accuracy"],
    "feature_names":       _model_data["feature_names"],
    "feature_importances": _model_data["feature_importances"],
    "confusion_matrix":    _model_data["confusion_matrix"],
    "roc_auc":             _model_data["roc_auc"],
    "n_samples":           _model_data["n_samples"],
}


# =========================================================
# API KEYS  (from model.ipynb)
# =========================================================

OPENWEATHER_KEY = "2afbf96e46d17e41c8adbe87e17e34d9"
TOMTOM_KEY      = "obbl4gsimE74QnjkG7gMN8y1RQY1KsGF"


# =========================================================
# CITY COORDINATES  (from model.ipynb)
# =========================================================

CITY_COORDS = {
    "Delhi":         (28.6139, 77.2090),
    "Mumbai":        (19.0760, 72.8777),
    "Bangalore":     (12.9716, 77.5946),
    "Chennai":       (13.0827, 80.2707),
    "Hyderabad":     (17.3850, 78.4867),
    "Pune":          (18.5204, 73.8567),
    "Ahmedabad":     (23.0225, 72.5714),
    "Kolkata":       (22.5726, 88.3639),
    "Lucknow":       (26.8467, 80.9462),
    "Jaipur":        (26.9124, 75.7873),
    "Puri":          (19.8135, 85.8312),
    "Bhubaneswar":   (20.2961, 85.8245),
    "Cuttack":       (20.4625, 85.8828),
    "Baripada":      (21.9333, 86.7500),
    "Nagpur":        (21.1458, 79.0882),
    "Angul":         (20.8399, 85.1018),
    "Rourkela":      (22.2604, 84.8536),
    "Keonjhar":      (21.6231, 85.5994),
    "Dhenkanal":     (20.6602, 85.5994),
    "Balasore":      (21.4938, 86.9311),
    "Sambalpur":     (21.4667, 83.9667),
    "Berhampur":     (19.3144, 84.7911),
    "Patna":         (25.5941, 85.1376),
    "Varanasi":      (25.3176, 82.9739),
    "Vishakhapatnam":(17.6868, 83.2185),
    "Jharsuguda":    (21.9100, 84.0000),
    "Raipur":        (21.2514, 81.6296),
    "Ranch":         (23.3441, 85.3096),
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two lat/lon points."""
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


def weather_level(city):
    """
    Return weather level integer (0=Clear, 1=Clouds, 2=Rain/Thunderstorm).
    Mirrors model.ipynb weather_level() exactly.
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_KEY}"
    try:
        r = requests.get(url, timeout=3).json()
        w = r["weather"][0]["main"]
    except Exception:
        return 0
    if w in ["Rain", "Thunderstorm"]:
        return 2
    elif w == "Clouds":
        return 1
    return 0


def traffic_penalty(lat, lon):
    """
    Return traffic level (0=free, 1=moderate, 2=heavy).
    Mirrors model.ipynb traffic_penalty() exactly.
    """
    url = (f"https://api.tomtom.com/traffic/services/4/flowSegmentData/"
           f"absolute/10/json?point={lat},{lon}&key={TOMTOM_KEY}")
    try:
        r = requests.get(url, timeout=3).json()
        flow = r["flowSegmentData"]
        ratio = flow["currentSpeed"] / flow["freeFlowSpeed"]
        if ratio < 0.5:
            return 2
        elif ratio < 0.8:
            return 1
    except Exception:
        pass
    return 0


def safe_encode(encoder, value, fallback=0):
    """Transform a label with the fitted encoder; use fallback if unseen."""
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        return fallback


def get_osrm_route(olat, olon, dlat, dlon):
    """
    Query OSRM to get the exact road coordinates between two points.
    Returns list of [lat, lon] coordinates, distance (km), and eta (hours).
    """
    url = f"http://router.project-osrm.org/route/v1/driving/{olon},{olat};{dlon},{dlat}?overview=full&geometries=geojson"
    try:
        r = requests.get(url, timeout=5).json()
        if "routes" in r and len(r["routes"]) > 0:
            route = r["routes"][0]
            # OSRM returns [lon, lat], we need [lat, lon] for folium
            coords = [[c[1], c[0]] for c in route["geometry"]["coordinates"]]
            distance = route["distance"] / 1000.0
            eta = route["duration"] / 3600.0
            return coords, distance, eta
    except Exception as e:
        print(f"OSRM Error: {e}")
        pass
    return None, None, None

def predict_leg_risk(origin_city, dest_city, distance_km, eta_hours, sla_deadline_str):
    """
    Use the trained XGBoost model to predict delay probability and compute
    SLA risk for one leg.

    Returns: (sla_risk, delay_risk, delay_prob, weather, traffic, recommendation)
      - sla_risk   : 0–100 float
      - delay_risk : 0–100 float  (model probability × 100)
      - delay_prob : raw float [0,1] from model.predict_proba
      - weather    : int 0/1/2
      - traffic    : int 0/1/2
      - recommendation : str
    """
    now = datetime.now()

    # ---- SLA Risk -------------------------------------------------------
    try:
        if len(sla_deadline_str) > 16:
            sla_dt = datetime.fromisoformat(sla_deadline_str)
        else:
            sla_dt = datetime.strptime(sla_deadline_str, "%Y-%m-%d %H:%M")
    except Exception:
        sla_dt = now + timedelta(hours=72)

    hours_to_sla = (sla_dt - now).total_seconds() / 3600
    if hours_to_sla <= 0:
        sla_risk = 100.0
    else:
        ratio = eta_hours / hours_to_sla
        sla_risk = min(round(ratio * 100, 1), 100.0)

    # ---- Live weather & traffic -----------------------------------------
    coords = CITY_COORDS.get(origin_city)
    if coords:
        lat, lon = coords
    else:
        lat, lon = geocode_location(origin_city)
        if lat is None:
            lat, lon = 20.5, 78.9

    weather = weather_level(origin_city)
    traffic = traffic_penalty(lat, lon)

    # ---- XGBoost feature vector ----------------------------------------
    # Feature order must match training columns EXACTLY:
    # 0: origin, 1: destination, 2: distance, 3: weather, 4: traffic, 
    # 5: Port_Congestion, 6: Eta_Hours, 7: Carrier_History
    enc_origin  = safe_encode(LE_ORIGIN,  origin_city)
    enc_dest    = safe_encode(LE_DEST,    dest_city)
    enc_weather = safe_encode(LE_WEATHER, str(weather))
    enc_traffic = safe_encode(LE_TRAFFIC, str(traffic))
    # Use carrier history level 2 (average) as default
    enc_carrier = safe_encode(LE_CARRIER, "2", fallback=1)
    # Default port congestion 1.0
    port_congestion = 1.0

    features = np.array([[
        enc_origin,
        enc_dest,
        float(distance_km),
        float(enc_weather),
        float(enc_traffic),
        float(port_congestion),
        float(eta_hours),
        float(enc_carrier),
    ]], dtype=np.float32)

    delay_prob = float(MODEL.predict_proba(features)[0][1])
    delay_risk = round(delay_prob * 100, 1)

    # ---- Recommendation based on combined score -------------------------
    combined = (sla_risk * 0.6) + (delay_risk * 0.4)
    if combined >= 70:
        rec = "Reroute shipment via alternative path"
    elif combined >= 50:
        rec = "Assign alternative carrier"
    elif combined >= 30:
        rec = "Send pre-alert to recipient"
    else:
        rec = "On track - no action required"

    return sla_risk, delay_risk, delay_prob, weather, traffic, rec


def generate_employee_id():
    return "EMP-" + "".join(random.choices(string.digits, k=4))


def generate_shipment_ref():
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
    email    = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    confirm  = request.form.get("confirm_password", "").strip()

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
        "email":    email,
        "password": password,
        "created_at": datetime.now()
    })

    return render_template("index.html", success="Registration successful. Please login as Admin.")


# =========================================================
# ADMIN LOGIN
# =========================================================

@app.route("/login/admin", methods=["POST"])
def login_admin():
    email    = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()

    user = db.admins.find_one({"email": email, "password": password})

    if user:
        session["admin_id"]   = str(user["_id"])
        session["admin_name"] = user["username"]
        session["role"]       = "admin"
        return redirect(url_for("admin_dashboard"))

    return render_template("index.html", error="Invalid admin credentials.", show_login=True, role="admin")


# =========================================================
# EMPLOYEE LOGIN
# =========================================================

@app.route("/login/employee", methods=["POST"])
def login_employee():
    employee_id = request.form.get("employee_id", "").strip()
    password    = request.form.get("password", "").strip()

    emp = db.employees.find_one({"employee_id": employee_id, "password": password})

    if emp:
        session["employee_db_id"] = str(emp["_id"])
        session["employee_name"]  = emp["name"]
        session["employee_id"]    = emp["employee_id"]
        session["role"]           = "employee"
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

    name     = request.form.get("name", "").strip()
    email    = request.form.get("email", "").strip()
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
        "name":        name,
        "email":       email,
        "password":    password,
        "admin_id":    admin_id,
        "created_at":  datetime.now()
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

    date_str   = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    emp_filter = request.args.get("employee_id", "")
    admin_id   = session["admin_id"]

    employees_cursor = list(db.employees.find({"admin_id": admin_id}))
    employees        = [{"employee_id": e["employee_id"], "name": e["name"]} for e in employees_cursor]
    emp_mapping      = {e["employee_id"]: e["name"] for e in employees_cursor}

    query = {"date_of_shipment": date_str}
    if emp_filter:
        if emp_filter in emp_mapping:
            query["employee_id"] = emp_filter
        else:
            query["employee_id"] = "NONE"
    else:
        query["employee_id"] = {"$in": list(emp_mapping.keys())}

    shipments_cursor = db.shipments.find(query).sort("created_at", -1)

    shipments = []
    for s in shipments_cursor:
        shipments.append({
            "shipment_ref":       s.get("shipment_ref", ""),
            "employee_id":        s.get("employee_id", ""),
            "employee_name":      emp_mapping.get(s.get("employee_id"), "Unknown"),
            "status":             s.get("status", "In Transit"),
            "sla_deadline":       str(s.get("sla_deadline", "")),
            "overall_risk_score": float(s.get("overall_risk_score", 0)),
            "recommendation":     s.get("recommendation", ""),
            "created_at":         str(s.get("created_at", ""))
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

    data         = request.get_json()
    sla_deadline = data.get("sla_deadline")
    legs_input   = data.get("legs", [])
    employee_id  = session["employee_id"]

    if not sla_deadline or len(legs_input) < 1:
        return jsonify({"error": "SLA deadline and at least one leg are required."}), 400

    shipment_ref = generate_shipment_ref()
    today        = datetime.now().strftime("%Y-%m-%d")

    computed_legs = []
    total_eta     = 0
    max_risk      = 0

    # Initialize Folium Map (centered roughly on India)
    m = folium.Map(location=[22.5, 78.9], zoom_start=5)

    for idx, leg in enumerate(legs_input):
        origin_city = leg.get("origin", "").strip()
        dest_city   = leg.get("destination", "").strip()

        if not origin_city or not dest_city:
            continue

        # Coordinates
        if origin_city in CITY_COORDS:
            olat, olon = CITY_COORDS[origin_city]
        else:
            olat, olon = geocode_location(origin_city)

        if dest_city in CITY_COORDS:
            dlat, dlon = CITY_COORDS[dest_city]
        else:
            dlat, dlon = geocode_location(dest_city)

        if olat is None or dlat is None:
            distance_km = 500.0
            olat, olon, dlat, dlon = 20.5, 78.9, 22.5, 88.3
        else:
            distance_km = haversine(olat, olon, dlat, dlon)

        avg_speed_kmh = 60.0
        eta_hours     = round(distance_km / avg_speed_kmh, 2)
        total_eta    += eta_hours

        # Use XGBoost model for delay predictions
        sla_risk, delay_risk, delay_prob, weather, traffic, leg_rec = predict_leg_risk(
            origin_city, dest_city, distance_km, eta_hours, sla_deadline
        )

        combined = (sla_risk * 0.6) + (delay_risk * 0.4)
        if combined > max_risk:
            max_risk = combined

        computed_legs.append({
            "sequence":          idx + 1,
            "origin":            origin_city,
            "destination":       dest_city,
            "origin_lat":        olat,
            "origin_lon":        olon,
            "dest_lat":          dlat,
            "dest_lon":          dlon,
            "distance_km":       round(distance_km, 2),
            "eta_hours":         eta_hours,
            "sla_risk":          sla_risk,
            "delay_risk":        delay_risk,
            "delay_prob":        round(delay_prob, 4),
            "weather":           weather,
            "traffic":           traffic,
            "leg_recommendation": leg_rec,
        })

        # --- Folium Map Plotting for this leg ---
        # Get exact OSRM route for accurate road drawing
        route_coords, _, _ = get_osrm_route(olat, olon, dlat, dlon)
        if not route_coords:
            # Fallback to straight line if OSRM fails
            route_coords = [[olat, olon], [dlat, dlon]]
        
        # Color line based on predicted risk (like notebook)
        line_color = "red" if delay_prob > 0.5 else "green"
        folium.PolyLine(route_coords, color=line_color, weight=6, opacity=0.8).add_to(m)

        # Markers
        folium.Marker([olat, olon], popup=f"Origin: {origin_city}", icon=folium.Icon(color="blue")).add_to(m)
        folium.Marker([dlat, dlon], popup=f"Destination: {dest_city}", icon=folium.Icon(color="red")).add_to(m)


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

    # Render map to HTML string
    map_html = m.get_root().render()

    shipment_doc = {
        "shipment_ref":       shipment_ref,
        "employee_id":        employee_id,
        "sla_deadline":       str(sla_deadline),
        "status":             "In Transit",
        "overall_risk_score": overall_risk,
        "recommendation":     overall_rec,
        "date_of_shipment":   today,
        "created_at":         datetime.now(),
        "legs":               computed_legs,
    }
    db.shipments.insert_one(shipment_doc)

    return jsonify({
        "success":            True,
        "shipment_ref":       shipment_ref,
        "overall_risk_score": overall_risk,
        "recommendation":     overall_rec,
        "legs":               computed_legs,
        "map_html":           map_html
    })


@app.route("/employee/shipments")
def employee_shipments():
    if session.get("role") != "employee":
        return jsonify({"error": "Unauthorized"}), 403

    employee_id = session["employee_id"]
    page        = int(request.args.get("page", 1))
    per_page    = 10
    offset      = (page - 1) * per_page

    total            = db.shipments.count_documents({"employee_id": employee_id})
    shipments_cursor = db.shipments.find({"employee_id": employee_id}).sort("created_at", -1).skip(offset).limit(per_page)

    shipments = []
    for s in shipments_cursor:
        shipments.append({
            "shipment_ref":       s.get("shipment_ref", ""),
            "sla_deadline":       str(s.get("sla_deadline", "")),
            "status":             s.get("status", "In Transit"),
            "overall_risk_score": float(s.get("overall_risk_score", 0)),
            "recommendation":     s.get("recommendation", ""),
            "date_of_shipment":   str(s.get("date_of_shipment", "")),
            "created_at":         str(s.get("created_at", "")),
            "id":                 str(s["_id"])
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
            "sequence":          leg.get("sequence"),
            "origin":            leg.get("origin"),
            "destination":       leg.get("destination"),
            "origin_lat":        leg.get("origin_lat"),
            "origin_lon":        leg.get("origin_lon"),
            "dest_lat":          leg.get("dest_lat"),
            "dest_lon":          leg.get("dest_lon"),
            "distance_km":       leg.get("distance_km"),
            "eta_hours":         leg.get("eta_hours"),
            "sla_risk":          leg.get("sla_risk"),
            "delay_risk":        leg.get("delay_risk"),
            "delay_prob":        leg.get("delay_prob"),
            "weather":           leg.get("weather"),
            "traffic":           leg.get("traffic"),
            "leg_recommendation": leg.get("leg_recommendation"),
        })

    return jsonify({
        "shipment_ref":       s.get("shipment_ref", ""),
        "sla_deadline":       str(s.get("sla_deadline", "")),
        "status":             s.get("status", "In Transit"),
        "overall_risk_score": float(s.get("overall_risk_score", 0)),
        "recommendation":     s.get("recommendation", ""),
        "date_of_shipment":   str(s.get("date_of_shipment", "")),
        "created_at":         str(s.get("created_at", "")),
        "legs":               legs,
    })


# =========================================================
# MODEL INFO  (new endpoint)
# =========================================================

@app.route("/model/info")
def model_info():
    """Return model metadata for the employee dashboard Model Info tab."""
    if session.get("role") not in ("employee", "admin"):
        return jsonify({"error": "Unauthorized"}), 403

    return jsonify({
        "algorithm":           "XGBoost Classifier",
        "accuracy":            MODEL_META["accuracy"],
        "roc_auc":             MODEL_META["roc_auc"],
        "n_samples":           MODEL_META["n_samples"],
        "feature_names":       MODEL_META["feature_names"],
        "feature_importances": MODEL_META["feature_importances"],
        "confusion_matrix":    MODEL_META["confusion_matrix"],
        "hyperparameters": {
            "n_estimators":   200,
            "learning_rate":  0.6,
            "max_depth":      12,
            "subsample":      0.9,
            "colsample_bytree": 0.9,
        },
        "target":  "delay (0 = On-Time, 1 = Delayed)",
        "dataset": "Odisha shipment dataset",
    })


if __name__ == "__main__":
    app.run(debug=True)