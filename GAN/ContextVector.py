import csv

source_map = {
    "region_tz_code": "desktop_timezone",
    "os_code": "desktop_os",
    "device_type_code": "desktop_device",
    "manufacturer_code": "desktop_manufacturer",
    "is_rooted": "phone_security",
    "gps_latitude": "phone_gps",
    "gps_longitude": "phone_gps",
    "location_conf_radius": "phone_gps_accuracy",
    "location_visit_count": "phone_location_history",
    "shift_profile_code": "work_profile",
    "session_start_epoch": "work_session",
    "session_duration_mins": "work_session",
    "time_since_last_login_mins": "work_session",
    "day_type_code": "work_calendar",
    "ip_address_as_int": "home_network",
    "ip_reputation_code": "home_network_security",
    "vpn_tor_usage": "desktop_network_privacy",
    "typing_speed_cpm": "keyboard_input",
    "click_pattern_code": "desktop_behavior",
    "role_code": "work_role",
    "scope_code": "work_scope",
    "failed_login_attempts": "work_security",
    "historic_risk_score": "work_risk",
    "system_mode_code": "desktop_system"
}

with open("output_with_trust_scores.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    headers = next(reader)
    rows = list(reader)

with open("context_vectors.csv", "w", newline="", encoding="utf-8") as f_out:
    writer = csv.writer(f_out)

    for user_id, row in enumerate(rows):
        vector = []
        for i in range(0, len(row), 2):
            feature_name = headers[i]
            source = source_map.get(feature_name, "unknown")
            feature = row[i]
            trustscore = row[i + 1]
            vector.append(f"{{{source}, {feature}, {trustscore}}}")
        writer.writerow([f"User {user_id} CV"] + vector)
