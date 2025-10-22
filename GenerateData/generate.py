#!/usr/bin/env python3
"""
Multi-Factor Authentication Data Generator

This script generates synthetic user authentication data for testing multi-factor 
authentication systems. It creates realistic user profiles with geographic, 
behavioral, and device characteristics, then simulates login sessions over time.
"""

import csv, random, time, datetime, math, sys

# Geographic regions with latitude/longitude boundaries
# 1.1: Eastern US, 1.2: Central US, 1.3: Western US
REGIONS = {
    1.1: {"lat_min": 25, "lat_max": 47, "lon_min": -84,  "lon_max": -67},  # Eastern US
    1.2: {"lat_min": 30, "lat_max": 45, "lon_min": -101, "lon_max": -90},  # Central US  
    1.3: {"lat_min": 32, "lat_max": 48, "lon_min": -125, "lon_max": -114},  # Western US
}

# Device manufacturer to operating system mappings
# Key: manufacturer code, Value: list of compatible OS codes
MFR_OS = {1:[3,4], 2:[1,2], 3:[1,2], 4:[1,2], 5:[1,2]}

# Work shift time profiles (start_hour, end_hour)
SHIFT_PROFILES = {1:(9,17), 2:(6,14), 3:(10,18)}  # Standard, Early, Late shifts

# Authentication factor codes
ROLE_CODES = [1,2,3]        # User role levels (e.g., employee, manager, admin)
SCOPE_CODES = [1,2,3]       # Access scope levels (e.g., limited, standard, full)
CLICK_CODES = [1,2,3]       # User clicking patterns (e.g., fast, normal, slow)
IP_REP_CODES = [1,2,3]      # IP reputation scores (e.g., good, suspicious, malicious)
DEVICE_TYPES = [0,1]        # 0: Desktop/Laptop, 1: Mobile

def ip_to_int(ip): 
    """Convert IP address string to 32-bit integer representation"""
    a,b,c,d = (int(x) for x in ip.split('.'))
    return (a<<24)|(b<<16)|(c<<8)|d

def rand_ip(pool=None):
    """
    Generate a random IP address as integer
    
    Args:
        pool: Optional tuple (first_octet, second_octet) to constrain IP range
    
    Returns:
        int: IP address as 32-bit integer
    """
    # Common first octets for realistic IP addresses
    first_pool = [24,66,67,68,69,72,74,99,130,140,142,144,147,152,155,157,160,168,170,192,199,204,205,206,207,208,209,216]
    
    if pool: 
        a,b = pool
    else: 
        a,b = random.choice(first_pool), random.randint(0,255)
    
    return ip_to_int(f"{a}.{b}.{random.randint(0,255)}.{random.randint(0,255)}")

def bearing_offset(lat, lon, dist_km, bearing_rad):
    """
    Calculate new coordinates given a starting point, distance, and bearing
    
    Uses haversine formula to compute geographic offset from a base location.
    Useful for generating realistic work/home location pairs.
    
    Args:
        lat: Starting latitude in degrees
        lon: Starting longitude in degrees  
        dist_km: Distance to travel in kilometers
        bearing_rad: Direction to travel in radians
    
    Returns:
        tuple: (new_latitude, new_longitude) in degrees
    """
    R = 6371.0  # Earth's radius in km
    dR = dist_km / R  # Angular distance
    lat1, lon1 = math.radians(lat), math.radians(lon)
    
    # Calculate new latitude
    lat2 = math.asin(math.sin(lat1)*math.cos(dR) + math.cos(lat1)*math.sin(dR)*math.cos(bearing_rad))
    
    # Calculate new longitude
    lon2 = lon1 + math.atan2(math.sin(bearing_rad)*math.sin(dR)*math.cos(lat1), 
                             math.cos(dR)-math.sin(lat1)*math.sin(lat2))
    
    return math.degrees(lat2), math.degrees(lon2)

def make_profile():
    """
    Generate a complete user profile with realistic characteristics
    
    Creates a synthetic user with:
    - Geographic home/work locations
    - Device and OS preferences  
    - Work schedule and role
    - Behavioral patterns (typing speed, clicking style)
    - Network characteristics (ISP pool)
    
    Returns:
        dict: Complete user profile with all authentication factors
    """
    p = {}
    
    # Geographic profile - pick a region and set home location
    p["rt_code"] = random.choice(list(REGIONS))
    r = REGIONS[p["rt_code"]]
    p["home_lat"] = random.uniform(r["lat_min"]+0.5, r["lat_max"]-0.5)
    p["home_lon"] = random.uniform(r["lon_min"]+0.5, r["lon_max"]-0.5)
    
    # Generate work location 12-18km from home in random direction
    lat_w, lon_w = bearing_offset(p["home_lat"], p["home_lon"], 
                                  random.uniform(12,18), random.random()*2*math.pi)
    p["work_lat"], p["work_lon"] = lat_w, lon_w
    
    # Device characteristics
    p["man_code"] = random.choice(list(MFR_OS))  # Device manufacturer
    p["os_code"] = random.choice(MFR_OS[p["man_code"]])  # Compatible OS
    p["dev_type"] = random.choice(DEVICE_TYPES)  # Desktop vs mobile
    p["is_rooted"] = random.choices([0,1], weights=[97,3])[0]  # 3% rooted devices
    
    # Work and access profile
    p["shift_code"] = random.choice(list(SHIFT_PROFILES))  # Work schedule
    p["role_code"] = random.choice(ROLE_CODES)  # User role level
    p["scope_code"] = random.choice(SCOPE_CODES)  # Access scope
    
    # Network profile - assign ISP pool for consistent IP ranges
    p["isp_pool"] = (random.choice([24,68,99,130,204]), random.randint(0,255))
    
    # Behavioral characteristics
    # Desktop users type faster than mobile users
    p["typing_base"] = random.randint(140,190) if p["dev_type"]==0 else random.randint(35,95)
    p["click_pref"] = random.choice(CLICK_CODES)  # Clicking pattern preference
    
    return p

def generate_rows(p, n=1000):
    """
    Generate authentication session data for a user profile over time
    
    Simulates realistic login patterns over a full year including:
    - Work vs weekend schedules
    - Varying session lengths and frequencies  
    - Geographic movement (mostly work/home, occasional travel)
    - Network behavior (VPN usage, IP reputation)
    - Authentication failures and risk scoring
    
    Args:
        p: User profile dictionary from make_profile()
        n: Number of sessions to generate (default 1000)
    
    Returns:
        list: List of session records, each as a list of values
    """
    rows, visit_cnt, last_login = [], 0, None
    
    # Generate data for full year leading up to today
    today = datetime.datetime.utcnow().replace(hour=0,minute=0,second=0, microsecond=0)
    start_date = today - datetime.timedelta(days=365)
    
    for day_idx in range(366):
        if len(rows) >= n: break  # Stop when we have enough sessions
        
        day = start_date + datetime.timedelta(days=day_idx)
        wd = day.weekday()  # 0=Monday, 6=Sunday
        
        # Determine number of sessions: more on weekdays, fewer on weekends
        sessions = random.randint(3,6) if wd<5 else random.randint(0,2)
        sessions = min(sessions, n-len(rows))  # Don't exceed target count
        
        # Get user's work schedule
        sh_start, sh_end = SHIFT_PROFILES[p["shift_code"]]
        span_minutes = (sh_end - sh_start)*60
        valid_offsets = range(0, span_minutes-4)  # Leave time for minimum session
        
        # Randomly distribute sessions throughout the work day
        sess_offsets = sorted(random.sample(valid_offsets, sessions))
        
        for off in sess_offsets:
            # Calculate session timing
            login_dt = day.replace(hour=sh_start) + datetime.timedelta(minutes=off)
            login_ep = int(login_dt.timestamp())
            
            # Time gap since last login (important for risk assessment)
            gap_mins = max(1, int((login_ep - last_login)/60)) if last_login else random.randint(30,720)
            last_login = login_ep
            
            # Session duration based on remaining work time
            max_left = span_minutes - off
            sess_len = random.randint(5, min(180, max_left))
            
            # Determine location: mostly work/home, 5% chance of travel
            if random.random()<0.05:
                # Simulate travel - random location within 50-300km
                lat, lon = bearing_offset(p["home_lat"], p["home_lon"], 
                                        random.uniform(50,300), random.random()*2*math.pi)
            else:
                # Normal locations: work during first half of shift, home during second half
                lat, lon = (p["work_lat"], p["work_lon"]) if off < span_minutes/2 else (p["home_lat"], p["home_lon"])
                # Add small random noise to simulate GPS accuracy
                lat += random.uniform(-0.015,0.015)
                lon += random.uniform(-0.015,0.015)
            
            visit_cnt += 1
            
            # Network characteristics
            vpn = 1 if random.random()<0.08 else 0  # 8% VPN usage
            ip_int = rand_ip() if vpn else rand_ip(p["isp_pool"])  # Use ISP pool unless VPN
            
            # Authentication failures: more likely after long gaps or Monday mornings
            failed_log = random.randint(1,3) if (gap_mins>2880 or (wd==0 and off<60 and random.random()<0.1)) else 0
            
            # Risk score calculation based on multiple factors
            risk = round(max(0,min(100, 
                p["typing_base"]/2 +      # Base risk from typing speed
                failed_log*5 +            # Failed login penalty
                vpn*8 +                   # VPN usage penalty  
                random.gauss(0,4)         # Random variation
            )),2)
            
            # Build complete session record
            rows.append([
                p["rt_code"], p["os_code"], p["dev_type"], p["man_code"], p["is_rooted"],
                round(lat,6), round(lon,6), random.randint(8,25), visit_cnt, p["shift_code"],
                login_ep, sess_len, gap_mins, 1 if wd>=5 else 0, ip_int,
                random.choices(IP_REP_CODES, weights=[70,25,5])[0], vpn,  # IP reputation: mostly good
                max(10,int(random.gauss(p["typing_base"],10))), p["click_pref"],
                p["role_code"], p["scope_code"], failed_log, risk, 0  # system_mode_code always 0
            ])
    
    return rows

# CSV column headers for the generated dataset
# These define the structure of the output data for ML training
HEADERS = [
    "region_tz_code",                    # Geographic region identifier
    "os_code",                          # Operating system code
    "device_type_code",                 # Desktop(0) vs Mobile(1)
    "manufacturer_code",                # Device manufacturer identifier
    "is_rooted",                        # Whether device is rooted/jailbroken
    "gps_latitude",                     # GPS coordinates (latitude)
    "gps_longitude",                    # GPS coordinates (longitude)  
    "location_conf_radius",             # GPS accuracy radius in meters
    "location_visit_count",             # Number of times user visited this location
    "shift_profile_code",               # Work schedule identifier
    "session_start_epoch",              # Login timestamp (Unix epoch)
    "session_duration_mins",            # How long the session lasted
    "time_since_last_login_mins",       # Gap since previous session
    "day_type_code",                    # Weekday(0) vs Weekend(1)
    "ip_address_as_int",                # IP address as 32-bit integer
    "ip_reputation_code",               # IP reputation score (1=good, 3=bad)
    "vpn_tor_usage",                    # Whether VPN/Tor was used
    "typing_speed_cpm",                 # Characters per minute typing speed
    "click_pattern_code",               # User's clicking behavior pattern
    "role_code",                        # User's organizational role
    "scope_code",                       # Access scope level
    "failed_login_attempts",            # Number of failed attempts this session
    "historic_risk_score",              # Calculated risk score (0-100)
    "system_mode_code"                  # System operation mode (always 0)
]

def write_csv(name="sample_data.csv", users=100, rows_per_user=1000):
    """
    Generate and write synthetic authentication data to CSV file
    
    Creates realistic multi-factor authentication data by:
    1. Generating user profiles with diverse characteristics
    2. Simulating login sessions over time for each user
    3. Writing all data to CSV with proper headers
    
    Args:
        name: Output CSV filename (default: "sample_data.csv")
        users: Number of unique user profiles to generate (default: 100)
        rows_per_user: Target sessions per user (default: 1000)
    """
    with open(name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)  # Write column headers first
        
        # Generate data for each user
        for _ in range(users):
            profile = make_profile()                    # Create user profile
            data = generate_rows(profile, rows_per_user)  # Generate sessions
            writer.writerows(data)                      # Write to CSV
    
    print(f"[+] {users * rows_per_user:,} rows written → {name}")

if __name__ == "__main__":
    # Command line arguments: python generate.py [users] [rows_per_user]
    users = sys.argv[1] if len(sys.argv) > 1 else "100"
    rows_per_user = sys.argv[2] if len(sys.argv) > 2 else "1000"
    write_csv(users=int(users), rows_per_user=int(rows_per_user))