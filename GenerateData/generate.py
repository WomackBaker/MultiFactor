import csv
import random
import time
import datetime

# ------------
# DICTIONARIES
# ------------
#
# Combined region/time zone codes:
#   1.1 = US-East
#   1.2 = US-Central
#   1.3 = US-West
# Each entry holds bounding boxes for lat/lon so that
# location data is consistent.
region_tz_dict = {
    1.1: {
        "name": "US-East",
        "lat_min": 25.0, "lat_max": 47.0,
        "lon_min": -84.0, "lon_max": -67.0
    },
    1.2: {
        "name": "US-Central",
        "lat_min": 30.0, "lat_max": 45.0,
        "lon_min": -101.0, "lon_max": -90.0
    },
    1.3: {
        "name": "US-West",
        "lat_min": 32.0, "lat_max": 48.0,
        "lon_min": -125.0, "lon_max": -114.0
    }
}

# Manufacturer codes:
#   1=Apple, 2=Samsung, 3=Google, 4=Xiaomi, 5=Huawei
# OS codes:
#   1=Android 12, 2=Android 13, 3=iOS 15, 4=iOS 16
manufacturer_to_os_codes = {
    1: [3, 4],  # Apple -> iOS 15 or iOS 16
    2: [1, 2],  # Samsung -> Android 12 or Android 13
    3: [1, 2],  # Google -> Android 12 or Android 13
    4: [1, 2],  # Xiaomi -> Android 12 or Android 13
    5: [1, 2],  # Huawei -> Android 12 or Android 13
}

# Device type codes:
#   0=Phone, 1=Tablet
device_type_codes = [0, 1]

# is_rooted: 0=Not rooted/jailbroken, 1=Rooted/Jailbroken

# Shift profiles (1=9–17, 2=6–14, 3=10–18) - each is exactly 8 hours.
shift_profiles = {
    1: (9, 17),  # Normal 9-5
    2: (6, 14),  # Early Bird 6-2
    3: (10, 18)  # Night Owl 10-6
}

# Roles: 1=basic, 2=admin, 3=guest
role_codes = [1, 2, 3]

# Scope: 1=read-only, 2=standard-access, 3=privileged-access
scope_codes = [1, 2, 3]

# IP reputation: 1=High, 2=Medium, 3=Low
ip_rep_codes = [1, 2, 3]

# VPN/Tor usage: 0=No, 1=Yes

# Click patterns: 1=rapid, 2=slow, 3=average
click_pattern_codes = [1, 2, 3]

# System mode: 0=normal, 1=lockdown
system_mode_codes = [0,1]


def ip_to_int(ip_str):
    """
    Convert dotted IPv4 string into a 32-bit integer.
    Example: "192.168.0.1" -> 3232235521
    """
    parts = ip_str.split('.')
    return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])


def generate_us_ipv4():
    """
    Generate a random US-like IPv4 address (rough approximation).
    Then convert to a single integer using ip_to_int().
    """
    # Some plausible first octets for US IPs
    possible_first_octets = [
        24, 66, 67, 68, 69, 71, 72, 73, 74, 99, 100,
        130, 131, 140, 142, 143, 144, 147, 152, 155,
        157, 160, 168, 170, 192, 199, 204, 205, 206,
        207, 208, 209, 216
    ]
    first_octet = random.choice(possible_first_octets)
    ip_str = f"{first_octet}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"
    return ip_to_int(ip_str)


def generate_random_data(num_rows=20):
    """
    Generate rows of data, each column strictly numeric.
    Ensures:
      - Region/time zone code is e.g. 1.1, 1.2, 1.3
      - Shift start is always between 6 AM and 6 PM
      - Each shift is 8 hours
      - Apple => iOS codes only, etc.
      - Timestamps stored as an integer (Unix epoch)
      - IP stored as integer
    """
    rows = []

    # We'll generate a random set of users
    for _ in range(num_rows):
        # Pick region/timezone as a single code, e.g. 1.1, 1.2, 1.3
        rt_code = random.choice([1.1, 1.2, 1.3])
        rtz_info = region_tz_dict[rt_code]

        # Generate lat/lon within bounding box
        lat = round(random.uniform(rtz_info["lat_min"], rtz_info["lat_max"]), 6)
        lon = round(random.uniform(rtz_info["lon_min"], rtz_info["lon_max"]), 6)

        # Manufacturer => OS
        man_code = random.choice(list(manufacturer_to_os_codes.keys()))
        os_code = random.choice(manufacturer_to_os_codes[man_code])

        # Device type
        dev_type_code = random.choice(device_type_codes)

        # Rooted/jailbroken
        is_rooted = random.choice([0, 1])

        # Confidence radius + visit count
        loc_confidence = random.randint(10, 100)
        loc_visit_count = random.randint(1, 50)

        # SHIFT / TIME
        #  pick one of the shift profiles
        shift_code = random.choice(list(shift_profiles.keys()))
        start_hr, end_hr = shift_profiles[shift_code]  # e.g. (9,17)

        # pick a random day in the last 30 days
        # We'll pick a random offset in seconds up to 30 days
        now = time.time()
        rand_past = random.randint(0, 60*60*24*30)
        # day_base = (now - rand_past) truncated to the start of that day
        # for clarity, we can just do a random epoch in the last 30 days
        session_day_epoch = int(now) - rand_past

        # Convert to a datetime to check which day of week it is
        day_dt = datetime.datetime.utcfromtimestamp(session_day_epoch)
        # day_type_code: 1=weekday, 2=weekend
        if day_dt.weekday() < 5:
            day_type_code = 0
        else:
            day_type_code = 1

        # session start is day_base + shift_start_in_hours
        # for variety, let them log in a bit after shift start
        # but never after shift end
        shift_length = end_hr - start_hr  # e.g. 8 hours
        # random offset in minutes up to shift_length
        offset_minutes = random.randint(0, shift_length*60 - 1)
        session_start = session_day_epoch + (start_hr * 3600) + (offset_minutes * 60)

        # session duration in minutes (max up to remainder of shift)
        max_session_mins = shift_length*60 - offset_minutes
        session_duration = random.randint(1, max_session_mins)

        # time_since_last_login in minutes (up to 1440 = one day)
        time_since_last_login = random.randint(1, 1440)

        # IP info
        ip_int = generate_us_ipv4()
        ip_rep = random.choice(ip_rep_codes)
        vpn_usage = random.choice([0, 1])

        # Behavior
        typing_speed = random.randint(20, 200)
        click_pattern = random.choice(click_pattern_codes)

        # Privileges
        role_code = random.choice(role_codes)
        scope_code = random.choice(scope_codes)

        # Risk
        failed_logins = random.randint(0, 5)
        historic_risk = round(random.uniform(0, 100), 2)  # we’ll keep a float, still numeric

        # System
        system_mode = random.choice(system_mode_codes)

        # Collect row
        # Everything is numeric
        row = [
            rt_code,             # e.g. 1.1, 1.2, 1.3
            os_code,             # 1..4
            dev_type_code,       # 1..2
            man_code,            # 1..5
            is_rooted,           # 0 or 1
            lat,                 # float
            lon,                 # float
            loc_confidence,      # int
            loc_visit_count,     # int
            shift_code,          # 1..3
            session_start,       # epoch (int)
            session_duration,    # int (minutes)
            time_since_last_login,  # int (minutes)
            day_type_code,       # 1=weekday, 2=weekend
            ip_int,              # 32-bit IP
            ip_rep,              # 1..3
            vpn_usage,           # 0 or 1
            typing_speed,        # int
            click_pattern,       # 1..3
            role_code,           # 1..3
            scope_code,          # 1..3
            failed_logins,       # int
            historic_risk,       # float
            system_mode          # 1..2
        ]

        rows.append(row)

    return rows


def write_csv(filename="sample_data.csv", num_rows=20):
    # Define column headers (all numeric, but you may label them for clarity)
    headers = [
        "region_tz_code",
        "os_code",
        "device_type_code",
        "manufacturer_code",
        "is_rooted",
        "gps_latitude",
        "gps_longitude",
        "location_conf_radius",
        "location_visit_count",
        "shift_profile_code",
        "session_start_epoch",
        "session_duration_mins",
        "time_since_last_login_mins",
        "day_type_code",
        "ip_address_as_int",
        "ip_reputation_code",
        "vpn_tor_usage",
        "typing_speed_cpm",
        "click_pattern_code",
        "role_code",
        "scope_code",
        "failed_login_attempts",
        "historic_risk_score",
        "system_mode_code"
    ]

    data_rows = generate_random_data(num_rows)

    with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in data_rows:
            writer.writerow(row)


if __name__ == "__main__":
    write_csv("sample_data.csv", num_rows=20)
    print("CSV file 'sample_data.csv' generated successfully!")
