document.addEventListener('DOMContentLoaded', function() {
    // --- Dark Mode Toggle ---
    const darkToggle = document.getElementById('darkModeToggle');
    function setDarkMode(enabled) {
        document.documentElement.classList.toggle('dark-mode', enabled);
        if (darkToggle) darkToggle.textContent = enabled ? '☀️' : '🌙';
        localStorage.setItem('darkMode', enabled ? 'true' : 'false');
    }
    // Load preference - this check is now primarily done by the head script
    // We ensure consistency here and set the button icon correctly
    const initialDarkModeState = localStorage.getItem('darkMode') === 'true';
    if (document.documentElement.classList.contains('dark-mode') !== initialDarkModeState) {
         // This can happen if the class was added/removed by another script or extension
         // or if the localStorage was changed manually after page load but before this script ran.
         // We trust the localStorage as the source of truth if there's a mismatch.
        document.documentElement.classList.toggle('dark-mode', initialDarkModeState);
    }
    if (darkToggle) darkToggle.textContent = initialDarkModeState ? '☀️' : '🌙';

    if (darkToggle) {
        darkToggle.addEventListener('click', () => {
            setDarkMode(!document.documentElement.classList.contains('dark-mode'));
        });
    }

    console.log('main.js loaded');
    const statusDiv = document.getElementById('status');
    const statusText = statusDiv.querySelector('.status-text');
    console.log('Status text element:', statusText);

    const socket = io();

    // --- Chart.js Setup ---
    const ctx = document.getElementById('sensorChart').getContext('2d');
    const sensorKeys = ['CO2', 'Alcohol', 'CO', 'NH4', 'Toluen', 'Acetone'];
    const chartColors = [
        '#1976d2', '#43a047', '#fbc02d', '#8e24aa', '#e64a19', '#00838f'
    ];
    // Store last 20 data points for each sensor
    const maxPoints = 20;
    const sensorHistory = {};
    sensorKeys.forEach(key => sensorHistory[key] = []);
    let timeLabels = [];

    const sensorChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timeLabels,
            datasets: sensorKeys.map((key, i) => ({
                label: key,
                data: sensorHistory[key],
                borderColor: chartColors[i],
                backgroundColor: chartColors[i] + '33',
                fill: false,
                tension: 0.3,
                pointRadius: 2,
                borderWidth: 2,
                spanGaps: true
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { top: 24, right: 24, bottom: 24, left: 24 } },
            plugins: {
                legend: { display: true, position: 'top' },
                title: { display: true, text: 'Sensor Data Over Time', font: { size: 18 } }
            },
            scales: {
                x: { title: { display: true, text: 'Time' } },
                y: { title: { display: true, text: 'Value' }, beginAtZero: true, min: 0 }
            }
        }
    });

    // Add prediction display element
    const predictionDisplay = document.createElement('div');
    predictionDisplay.id = 'prediction';
    predictionDisplay.className = 'status-box';
    predictionDisplay.innerHTML = `
        <h3>Pollution Prediction</h3>
        <div class="prediction-value">--.- ppm</div>
        <div class="prediction-label">Estimated Pollution Level</div>
    `;
    document.querySelector('.status-container').appendChild(predictionDisplay);

    // Update status immediately when socket connects
    socket.on('connect', function () {
        updateStatus('Connected to server', true);
        checkServerStatus();
    });

    socket.on('sensor_data', function (msg) {
        // Ensure msg and msg.data are valid before proceeding
        if (!msg || !msg.data) {
            console.error('Received invalid sensor data:', msg);
            return;
        }

        console.log('Received sensor data with prediction:', msg);
        const now = new Date();
        const timeLabel = now.toLocaleTimeString();
        const sensorData = msg.data;

        // Update time labels for the chart
        timeLabels.push(timeLabel);
        if (timeLabels.length > maxPoints) {
            timeLabels.shift();
        }

        // Update sensor values on the page and data for the chart
        sensorKeys.forEach(key => {
            const value = sensorData[key];
            
            // Update the text display for the sensor
            const element = document.getElementById(key);
            if (element) {
                element.textContent = value ?? '--';
            }

            // Prepare value for the chart (use null for gaps)
            let valForChart = null;
            if (value !== undefined && value !== null && value !== '') {
                valForChart = Number(value);
            }
            
            sensorHistory[key].push(valForChart);
            if (sensorHistory[key].length > maxPoints) {
                sensorHistory[key].shift();
            }
        });

        // Update the chart with new data
        sensorChart.data.labels = timeLabels;
        sensorKeys.forEach((key, i) => {
            sensorChart.data.datasets[i].data = sensorHistory[key];
        });
        sensorChart.update();

        // Update the prediction display
        if (msg.prediction !== undefined && msg.prediction !== null) {
            updatePredictionDisplay(msg.prediction);
        }
    });

    // Update status when socket disconnects
    socket.on('disconnect', function () {
        updateStatus('Disconnected from server', false);
    });

    function updateStatus(message, isConnected) {
        const statusText = statusDiv.querySelector('.status-text');
        statusText.textContent = message;
        statusDiv.className = isConnected ? 'connected' : 'disconnected';
    }

    function updatePredictionDisplay(prediction) {
        const predictionElement = document.querySelector('#prediction .prediction-value');
        if (predictionElement) {
            predictionElement.textContent = `${prediction.toFixed(1)} ppm`;

            // Color code the prediction based on 0-25 scale
            if (prediction < 10) {
                predictionElement.style.color = '#4CAF50'; // Green
            } else if (prediction < 20) {
                predictionElement.style.color = '#FFC107'; // Yellow
            } else {
                predictionElement.style.color = '#F44336'; // Red
            }
        }
    }

    function checkServerStatus() {
        fetch('/status')
            .then(res => res.json())
            .then(data => {
                if (data.mqtt_connected) {
                    updateStatus('Connected to MQTT broker', true);
                } else {
                    updateStatus('MQTT broker disconnected', false);
                }
            })
            .catch(() => {
                updateStatus('Server status unavailable', false);
            });
    }

    // Check status every 3 seconds instead of 5
    setInterval(checkServerStatus, 3000);
    checkServerStatus();
});
