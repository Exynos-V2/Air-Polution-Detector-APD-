# Air Pollution Detector (APD)

**Kelompok 7**

An IoT-based air pollution monitoring system with AI predictions and real-time web dashboard.

## Anggota

**Kelompok 7:**
- Willyam Andika Putra - 2332015
- Jevintantono - 2332017  
- Hardy Setiawan - 2332020
- Ryan Fiorentino Goh - 2332022

## Features

- Real-time sensor data monitoring via MQTT
- AI-powered CO₂ level predictions using Random Forest
- Web dashboard with authentication
- Mobile app API support
- Docker deployment ready

## Tech Stack

- **Backend**: Python Flask, Flask-SocketIO
- **Database**: MySQL
- **IoT**: MQTT
- **AI/ML**: Scikit-learn, Random Forest
- **Frontend**: HTML, CSS, JavaScript, Chart.js

## Quick Start

### Using Docker (Recommended)
```bash
git clone https://github.com/your-username/Air-Polution-Detector-APD-.git
cd Air-Polution-Detector-APD-
docker-compose up --build
```

### Local Development
```bash
pip install -r requirements.txt
python app.py
```

Access the dashboard at: http://localhost:5000

## Sensor Data

Monitors: CO₂, Alcohol, CO, NH₄, Toluene, and Acetone levels

*Making air quality monitoring accessible and intelligent*