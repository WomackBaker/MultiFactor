import csv
import random
import time
import datetime

def generate_random_data(num_rows=20):
    # Possible realistic values (customize as needed)
    os_versions = ["Android 13", "Android 12", "iOS 16", "iOS 15"]
    device_types = ["Phone", "Tablet"]
    manufacturers = ["Apple", "Samsung", "Google", "Xiaomi", "Huawei"]
    roles = ["basic", "admin", "guest"]
    scope_levels = ["read-only", "standard-access", "privileged-access"]
    ip_reputations = ["High", "Medium", "Low"]
    click_pattern_categories = ["rapid", "slow", "average"]
    system_modes = ["normal", "lockdown"]
    regions = ["US-East", "US-West", "EU-Central", "Asia-Pacific"]

    rows = []

    for _ in range(num_rows):
        # Device
        device_os_version = random.choice(os_versions)
        device_type = random.choice(device_types)
        manufacturer = random.choice(manufacturers)
        is_rooted = random.choice([True, False])

        # Location
        gps_latitude = round(random.uniform(-90, 90), 6)
        gps_longitude = round(random.uniform(-180, 180), 6)
        region = random.choice(regions)
        location_confidence_radius = random.randint(10, 100)  # meters
        location_visit_count = random.randint(1, 50)

        # Time
        now = datetime.datetime.now()
        # We'll create a random timestamp within the last 30 days
        random_past_offset = random.randint(0, 60*60*24*30)
        random_timestamp = now - datetime.timedelta(seconds=random_past_offset)
        timestamp_str = random_timestamp.isoformat()

        time_since_last_login = random.randint(1, 1440)  # in minutes (1 day max)
        session_duration = random.randint(1, 120)       # in minutes (2 hours max)
        # Rough classification of time-of-day
        hour = random_timestamp.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        # Determine weekday/weekend
        day_type = "weekday" if random_timestamp.weekday() < 5 else "weekend"

        # Network
        # Create a random IPv4 address
        ip_address = ".".join(str(random.randint(0, 255)) for _ in range(4))
        ip_reputation = random.choice(ip_reputations)
        vpn_tor_usage = random.choice([True, False])

        # Behavior
        typing_speed = random.randint(20, 200)  # chars per minute
        click_patterns = random.choice(click_pattern_categories)

        # Privileges
        current_role = random.choice(roles)
        scope_of_access = random.choice(scope_levels)

        # Risk Profile
        failed_login_attempts = random.randint(0, 5)
        historic_risk_score = round(random.uniform(0, 100), 2)

        # System Sensitivity
        system_mode = random.choice(system_modes)

        row = [
            device_os_version,
            device_type,
            manufacturer,
            is_rooted,
            gps_latitude,
            gps_longitude,
            region,
            location_confidence_radius,
            location_visit_count,
            timestamp_str,
            time_since_last_login,
            session_duration,
            time_of_day,
            day_type,
            ip_address,
            ip_reputation,
            vpn_tor_usage,
            typing_speed,
            click_patterns,
            current_role,
            scope_of_access,
            failed_login_attempts,
            historic_risk_score,
            system_mode
        ]
        rows.append(row)

    return rows

def write_csv(filename="sample_data.csv", num_rows=20):
    headers = [
        "device_os_version",
        "device_type",
        "manufacturer",
        "is_rooted",
        "gps_latitude",
        "gps_longitude",
        "region",
        "location_confidence_radius",
        "location_visit_count",
        "timestamp",
        "time_since_last_login",
        "session_duration",
        "time_of_day",
        "day_type",
        "ip_address",
        "ip_reputation",
        "vpn_tor_usage",
        "typing_speed",
        "click_patterns",
        "current_role",
        "scope_of_access",
        "failed_login_attempts",
        "historic_risk_score",
        "system_mode"
    ]

    data_rows = generate_random_data(num_rows)

    with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data_rows)

if __name__ == "__main__":
    write_csv("sample_data.csv", num_rows=20)
    print("CSV file 'sample_data.csv' generated successfully!")
