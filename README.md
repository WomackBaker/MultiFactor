# Multifactor App

## Overview
The Multifactor App is a simple Android application featuring five buttons. Currently, only voice and facial authentication are functional. The app communicates with a local server, sending essential data and receiving authentication statuses.

## Current Functionality
- **Voice Authentication:** Uses preset voice and image data sent to the authentication server when activated.
- **Facial Authentication:** Similarly uses a preset image for testing purposes.
- **SMS:** Sends phone data such as UUID, latitude, longitude, IP address, memory, current time, and RSSI to the server for testing connectivity and response.

## Server Setup
The application interacts with two main components on the server-side:
1. **Authentication Server**
   - Runs inside a Docker container.
   - Utilizes a Flask application to manage API endpoints.
   - Handles data from the Android app for authentication.
   - Uses the Deepface library for facial recognition and a CNN for voice recognition from `.wav` files.

2. **Logging Server**
   - Also encapsulated within a Docker container.
   - Receives and processes JSON requests.
   - Logs are maintained in CSV format based on the UUID provided in the request.

## Future Implementations
- Implementing actual data capture for voice and facial authentication, replacing the test presets.
- Enhancing SMS functionality for broader testing or live deployment.

## Network Configuration
- The app currently interacts with servers at the local address `10.0.2.2:30080`.
