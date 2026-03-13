# team-02-TechZero

# AI Smart Logistics – Early Delay Warning System

## Overview
AI Smart Logistics is an intelligent shipment monitoring system that predicts **shipment delays before they occur** using machine learning and real-time external data.

The system analyzes multiple factors such as **distance, weather conditions, traffic congestion, shipment medium, and carrier history** to estimate the risk of delays and suggest better routes.

It also includes **AI anomaly detection and disruption simulation** to improve logistics reliability.

---

## Key Features

### AI Delay Prediction
Uses **XGBoost Classifier** to predict shipment delay probability.

### Real-Time Risk Factors
The system analyzes:

- Weather conditions (OpenWeather API)
- Traffic congestion (TomTom API)
- Distance & ETA
- Carrier history
- Shipment medium (Land / Air / Water)

### AI Anomaly Detection
Detects unusual situations such as:

- Extremely long routes
- High travel time
- High delay probability

### AI Disruption Simulation
Simulates possible disruptions like:

- Weather impact
- Traffic congestion

### Smart Route Selection
Compares multiple routes and selects the **lowest risk route**.

### Interactive Map Dashboard
Uses **Folium** to generate a visual map showing:

- Shipment routes
- Origin & destination
- Delay risk heatmap

---

## Dataset

Dataset contains **650 logistics shipment records** including:

- Origin
- Destination
- Distance
- Weather
- Traffic
- Carrier history
- Estimated travel time
- Shipment medium (Land / Air / Water)
- Delay label

---

## Machine Learning Model

Algorithm used:

**XGBoost Classifier**

Why XGBoost?

- High accuracy
- Handles structured logistics data well
- Works efficiently with large datasets

---

## System Architecture

User Input (Shipment Details) -> Route Generation (OSRM API) -> Real-Time Data Collection(Weather API,Traffic API) -> Feature Engineering -> Machine Learning Model (XGBoost) -> Delay Risk Prediction -> AI Anomaly Detection -> Disruption Simulation -> Route Recommendation -> Map Visualization

## Interactive Map Visualization

The system generates an interactive logistics map using Folium where users can:
 - Zoom in and zoom out to explore routes across different regions.
 - View shipment routes covering multiple geographic areas.
 - Identify origin and destination points clearly on the map.
 - Visualize the shortest or most optimal route between origin and destination based on distance, traffic, and delay risk.
 - Display AI-generated route recommendations for efficient logistics planning.
 - The map is exported as an HTML dashboard, allowing users to open it in a browser and interact with the shipment routes         dynamically.

 ---



## Database Storage
All shipment records and logistics data are stored in MongoDB for efficient data management and scalability.
The system saves the following shipment details:
 - Origin city
 - Destination city
 - Route coordinates
 - Distance
 - Estimated Time of Arrival (ETA)
 - Delay risk probability
 - Shipment timestamp

MongoDB allows the system to:
 - Store large numbers of shipment records
 - Retrieve shipment history quickly
 - Enable future analytics on logistics performance
 - Scale easily for real-world logistics systems

 ---