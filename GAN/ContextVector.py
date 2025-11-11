import csv

# Map feature names to their corresponding data sources/categories
# This helps categorize different authentication factors by their origin
source_map = {
    # Desktop/System information
    "region_tz_code": "desktop_timezone",
    "os_code": "desktop_os",
    "device_type_code": "desktop_device",
    "manufacturer_code": "desktop_manufacturer",
    
    # Phone/Mobile security
    "is_rooted": "phone_security",
    
    # GPS and location data
    "gps_latitude": "phone_gps",
    "gps_longitude": "phone_gps",
    "location_conf_radius": "phone_gps_accuracy",
    "location_visit_count": "phone_location_history",
    
    # Work-related profiles and sessions
    "shift_profile_code": "work_profile",
    "session_start_epoch": "work_session",
    "session_duration_mins": "work_session",
    "time_since_last_login_mins": "work_session",
    "day_type_code": "work_calendar",
    
    # Network and security information
    "ip_address_as_int": "home_network",
    "ip_reputation_code": "home_network_security",
    "vpn_tor_usage": "desktop_network_privacy",
    
    # User behavior patterns
    "typing_speed_cpm": "keyboard_input",
    "click_pattern_code": "desktop_behavior",
    
    # Authorization and access control
    "role_code": "work_role",
    "scope_code": "work_scope",
    
    # Security metrics
    "failed_login_attempts": "work_security",
    "historic_risk_score": "work_risk",
    "system_mode_code": "desktop_system"
}

# Read the input CSV file containing authentication data with trust scores
# The file is expected to have alternating columns: feature_value, trust_score, feature_value, trust_score, etc.
with open("output_with_trust_scores.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    headers = next(reader)  # Get column headers
    rows = list(reader)     # Load all data rows

# Create output file for context vectors
# Each context vector represents a user's authentication profile
with open("context_vectors.csv", "w", newline="", encoding="utf-8") as f_out:
    writer = csv.writer(f_out)

    # Process each user's data row to create context vectors
    for user_id, row in enumerate(rows):
        vector = []
        
        # Process each feature-trust_score pair in the row
        # Data is structured as: [feature1_value, feature1_trust, feature2_value, feature2_trust, ...]
        for i in range(0, len(row), 2):
            feature_name = headers[i]  # Get feature name from header
            source = source_map.get(feature_name, "unknown")  # Map to data source category
            feature = row[i]           # Feature value
            trustscore = row[i + 1]    # Corresponding trust score
            
            # Create context vector element in format: {source, feature_value, trust_score}
            vector.append(f"{{{source}, {feature}, {trustscore}}}")
        
        # Write the complete context vector for this user
        writer.writerow([f"User {user_id} CV"] + vector)
