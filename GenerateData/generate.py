#!/usr/bin/env python3
import csv, random, time, datetime, math, sys

REGIONS = {
    1.1: {"lat_min": 25, "lat_max": 47, "lon_min": -84,  "lon_max": -67},
    1.2: {"lat_min": 30, "lat_max": 45, "lon_min": -101, "lon_max": -90},
    1.3: {"lat_min": 32, "lat_max": 48, "lon_min": -125, "lon_max": -114},
}

MFR_OS = {1:[3,4], 2:[1,2], 3:[1,2], 4:[1,2], 5:[1,2]}
SHIFT_PROFILES = {1:(9,17), 2:(6,14), 3:(10,18)}
ROLE_CODES = [1,2,3]
SCOPE_CODES = [1,2,3]
CLICK_CODES = [1,2,3]
IP_REP_CODES = [1,2,3]
DEVICE_TYPES = [0,1]

def ip_to_int(ip): a,b,c,d = (int(x) for x in ip.split('.')); return (a<<24)|(b<<16)|(c<<8)|d
def rand_ip(pool=None):
    first_pool = [24,66,67,68,69,72,74,99,130,140,142,144,147,152,155,157,160,168,170,192,199,204,205,206,207,208,209,216]
    if pool: a,b = pool
    else: a,b = random.choice(first_pool), random.randint(0,255)
    return ip_to_int(f"{a}.{b}.{random.randint(0,255)}.{random.randint(0,255)}")

def bearing_offset(lat, lon, dist_km, bearing_rad):
    R = 6371.0; dR = dist_km / R; lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(dR) + math.cos(lat1)*math.sin(dR)*math.cos(bearing_rad))
    lon2 = lon1 + math.atan2(math.sin(bearing_rad)*math.sin(dR)*math.cos(lat1), math.cos(dR)-math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def make_profile():
    p = {}
    p["rt_code"] = random.choice(list(REGIONS))
    r = REGIONS[p["rt_code"]]
    p["home_lat"] = random.uniform(r["lat_min"]+0.5, r["lat_max"]-0.5)
    p["home_lon"] = random.uniform(r["lon_min"]+0.5, r["lon_max"]-0.5)
    lat_w, lon_w = bearing_offset(p["home_lat"], p["home_lon"], random.uniform(12,18), random.random()*2*math.pi)
    p["work_lat"], p["work_lon"] = lat_w, lon_w
    p["man_code"] = random.choice(list(MFR_OS))
    p["os_code"] = random.choice(MFR_OS[p["man_code"]])
    p["dev_type"] = random.choice(DEVICE_TYPES)
    p["is_rooted"] = random.choices([0,1], weights=[97,3])[0]
    p["shift_code"] = random.choice(list(SHIFT_PROFILES))
    p["role_code"] = random.choice(ROLE_CODES)
    p["scope_code"] = random.choice(SCOPE_CODES)
    p["isp_pool"] = (random.choice([24,68,99,130,204]), random.randint(0,255))
    p["typing_base"] = random.randint(140,190) if p["dev_type"]==0 else random.randint(35,95)
    p["click_pref"] = random.choice(CLICK_CODES)
    return p

def generate_rows(p, n=1000):
    rows, visit_cnt, last_login = [], 0, None
    today = datetime.datetime.utcnow().replace(hour=0,minute=0,second=0, microsecond=0)
    start_date = today - datetime.timedelta(days=365)
    for day_idx in range(366):
        if len(rows) >= n: break
        day = start_date + datetime.timedelta(days=day_idx)
        wd = day.weekday()
        sessions = random.randint(3,6) if wd<5 else random.randint(0,2)
        sessions = min(sessions, n-len(rows))
        sh_start, sh_end = SHIFT_PROFILES[p["shift_code"]]
        span_minutes = (sh_end - sh_start)*60
        valid_offsets = range(0, span_minutes-4)
        sess_offsets = sorted(random.sample(valid_offsets, sessions))
        for off in sess_offsets:
            login_dt = day.replace(hour=sh_start) + datetime.timedelta(minutes=off)
            login_ep = int(login_dt.timestamp())
            gap_mins = max(1, int((login_ep - last_login)/60)) if last_login else random.randint(30,720)
            last_login = login_ep
            max_left = span_minutes - off
            sess_len = random.randint(5, min(180, max_left))
            if random.random()<0.05:
                lat, lon = bearing_offset(p["home_lat"], p["home_lon"], random.uniform(50,300), random.random()*2*math.pi)
            else:
                lat, lon = (p["work_lat"], p["work_lon"]) if off < span_minutes/2 else (p["home_lat"], p["home_lon"])
                lat += random.uniform(-0.015,0.015); lon += random.uniform(-0.015,0.015)
            visit_cnt += 1
            vpn = 1 if random.random()<0.08 else 0
            ip_int = rand_ip() if vpn else rand_ip(p["isp_pool"])
            failed_log = random.randint(1,3) if (gap_mins>2880 or (wd==0 and off<60 and random.random()<0.1)) else 0
            risk = round(max(0,min(100, p["typing_base"]/2 + failed_log*5 + vpn*8 + random.gauss(0,4))),2)
            rows.append([
                p["rt_code"], p["os_code"], p["dev_type"], p["man_code"], p["is_rooted"],
                round(lat,6), round(lon,6), random.randint(8,25), visit_cnt, p["shift_code"],
                login_ep, sess_len, gap_mins, 1 if wd>=5 else 0, ip_int,
                random.choices(IP_REP_CODES, weights=[70,25,5])[0], vpn,
                max(10,int(random.gauss(p["typing_base"],10))), p["click_pref"],
                p["role_code"], p["scope_code"], failed_log, risk, 0
            ])
    return rows

HEADERS = [
    "region_tz_code","os_code","device_type_code","manufacturer_code","is_rooted",
    "gps_latitude","gps_longitude","location_conf_radius","location_visit_count",
    "shift_profile_code","session_start_epoch","session_duration_mins",
    "time_since_last_login_mins","day_type_code","ip_address_as_int",
    "ip_reputation_code","vpn_tor_usage","typing_speed_cpm","click_pattern_code",
    "role_code","scope_code","failed_login_attempts","historic_risk_score",
    "system_mode_code"
]

def write_csv(name="sample_data.csv", users=100, rows_per_user=1000):
    with open(name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        for _ in range(users):
            profile = make_profile()
            data = generate_rows(profile, rows_per_user)
            writer.writerows(data)
    print(f"[+] {users * rows_per_user:,} rows written → {name}")

if __name__ == "__main__":
    users = sys.argv[1] if len(sys.argv) > 1 else "100"
    rows_per_user = sys.argv[2] if len(sys.argv) > 2 else "1000"
    write_csv(users=int(users), rows_per_user=int(rows_per_user))