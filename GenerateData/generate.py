#!/usr/bin/env python3
import csv, time, datetime, math, sys, hashlib

# ----------------------------
# Original constants preserved
# ----------------------------
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

# ----------------------------
# Deterministic helpers (no RNG)
# ----------------------------
def _h(*xs):
    s = "|".join(str(x) for x in xs)
    return hashlib.sha256(s.encode()).hexdigest()

def dfloat01(*key):
    return int(_h(*key)[:12], 16) / float(0xFFFFFFFFFFFF)  # [0,1)

def dint(lo, hi, *key):
    # inclusive bounds
    if hi <= lo:
        return lo
    r = dfloat01(*key)
    return lo + int(r * (hi - lo + 1 - 1e-9))

def dchoice(seq, *key):
    return seq[dint(0, len(seq)-1, *key)]

def dgauss(mu, sigma, *key):
    # Box-Muller with two deterministic uniforms
    u1 = max(1e-9, dfloat01(*key, "a"))
    u2 = max(1e-9, dfloat01(*key, "b"))
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2*math.pi*u2)
    return mu + sigma * z

# ----------------------------
# IP and geo utilities
# ----------------------------
def ip_to_int(ip):
    a,b,c,d = (int(x) for x in ip.split('.'))
    return (a<<24)|(b<<16)|(c<<8)|d

def rand_ip(pool, *key):
    # Deterministic "random" IP respecting an optional (A,B) pool for ISP
    first_pool = [24,66,67,68,69,72,74,99,130,140,142,144,147,152,155,157,160,168,170,192,199,204,205,206,207,208,209,216]
    if pool:
        a, b = pool
    else:
        a = dchoice(first_pool, *key, "A")
        b = dint(0,255, *key, "B")
    c = dint(0,255, *key, "C")
    d = dint(0,255, *key, "D")
    return ip_to_int(f"{a}.{b}.{c}.{d}")

def bearing_offset(lat, lon, dist_km, bearing_rad):
    R = 6371.0
    dR = dist_km / R
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(dR) + math.cos(lat1)*math.sin(dR)*math.cos(bearing_rad))
    lon2 = lon1 + math.atan2(math.sin(bearing_rad)*math.sin(dR)*math.cos(lat1), math.cos(dR)-math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

# ----------------------------
# Profiles (deterministic)
# ----------------------------
def make_profile(uid):
    p = {}
    p["rt_code"]   = dchoice(list(REGIONS), "rt", uid)
    r              = REGIONS[p["rt_code"]]
    p["home_lat"]  = r["lat_min"] + 0.5 + dfloat01("hLA", uid) * (r["lat_max"] - r["lat_min"] - 1.0)
    p["home_lon"]  = r["lon_min"] + 0.5 + dfloat01("hLO", uid) * (r["lon_max"] - r["lon_min"] - 1.0)
    lat_w, lon_w   = bearing_offset(p["home_lat"], p["home_lon"],
                                    12 + dfloat01("dist", uid)*6,
                                    dfloat01("bear", uid)*2*math.pi)
    p["work_lat"], p["work_lon"] = lat_w, lon_w
    p["man_code"]  = dchoice(list(MFR_OS), "man", uid)
    p["os_code"]   = dchoice(MFR_OS[p["man_code"]], "os", uid)
    p["dev_type"]  = dchoice(DEVICE_TYPES, "dev", uid)
    p["is_rooted"] = 1 if dfloat01("root", uid) > 0.97 else 0
    p["shift_code"]= dchoice(list(SHIFT_PROFILES), "shift", uid)
    p["role_code"] = dchoice(ROLE_CODES, "role", uid)
    p["scope_code"]= dchoice(SCOPE_CODES, "scope", uid)
    p["isp_pool"]  = (dchoice([24,68,99,130,204], "ispA", uid), dint(0,255, "ispB", uid))
    p["typing_base"]= dint(140,190,"typeA",uid) if p["dev_type"]==0 else dint(35,95,"typeB",uid)
    p["click_pref"]= dchoice(CLICK_CODES, "click", uid)

    # Minimal new field: attacker flag (~1 in 7 deterministically)
    p["is_attacker"] = 1 if (uid % 7 == 0) else 0
    return p

# ----------------------------
# Trust model helpers
# ----------------------------
def trust_from_features(base, ip_changed, overseas, is_attacker, stability_bonus, variation):
    # Everyone "starts low"
    trust = base

    # Attackers ideally stay low (0.10..0.30)
    if is_attacker:
        trust = 0.10 + variation*0.20  # variation in [0,1) -> 0.10..0.30
    else:
        # Normals can build trust with stability
        trust += stability_bonus  # 0.08 or 0.16
        trust += 0.20             # domestic baseline bonus

        # Ensure normals end mostly high (>=0.70 floor)
        if trust < 0.70:
            trust = 0.70
        # Good/stable normals drift higher
        if ip_changed == 0:
            trust = max(trust, 0.85)

    # Overseas cap (automatic low)
    if overseas:
        trust = min(trust, 0.20)

    # Bound
    return max(0.0, min(1.0, round(trust, 4)))

def generate_rows(p, uid, n=1000):
    rows, visit_cnt, last_login = [], 0, None
    prev_ip = None

    today = datetime.datetime.utcnow().replace(hour=0,minute=0,second=0, microsecond=0)
    start_date = today - datetime.timedelta(days=365)

    for day_idx in range(366):
        if len(rows) >= n: break
        day = start_date + datetime.timedelta(days=day_idx)
        wd = day.weekday()

        # Deterministic sessions per day
        sess_week = 3 + dint(0,3,"sesW",uid,day_idx) if wd<5 else dint(0,2,"sesE",uid,day_idx)
        sessions = min(sess_week, n-len(rows))

        sh_start, sh_end = SHIFT_PROFILES[p["shift_code"]]
        span_minutes = (sh_end - sh_start)*60
        if span_minutes < 10: span_minutes = 10  # safety

        # Deterministic distinct offsets
        # Create a sorted set of session offsets without randomness
        base_step = max(5, span_minutes // max(1, sessions+1))
        sess_offsets = [(i+1)*base_step for i in range(sessions)]
        sess_offsets = [min(x, span_minutes-5) for x in sess_offsets]

        for sess_idx, off in enumerate(sess_offsets):
            if len(rows) >= n: break

            login_dt = day.replace(hour=sh_start) + datetime.timedelta(minutes=off)
            login_ep = int(login_dt.timestamp())

            gap_mins = ( (login_ep - last_login)//60 if last_login else (30 + 15*((uid+day_idx+sess_idx)%20)) )
            if gap_mins <= 0: gap_mins = 1
            last_login = login_ep

            max_left = span_minutes - off
            sess_len = max(5, min(180, 10 + ((uid+day_idx+sess_idx) % max(10, max_left))))

            # Deterministic location choice (brief travel 5% of weekdays)
            travel_flag = 1 if (dfloat01("trav", uid, day_idx, sess_idx) < 0.05 and wd<5) else 0
            if travel_flag:
                dist = 50 + dfloat01("td", uid, day_idx, sess_idx)*250
                bear = dfloat01("tb", uid, day_idx, sess_idx)*2*math.pi
                lat, lon = bearing_offset(p["home_lat"], p["home_lon"], dist, bear)
            else:
                if off < span_minutes/2:
                    lat, lon = p["work_lat"], p["work_lon"]
                else:
                    lat, lon = p["home_lat"], p["home_lon"]
                # tiny deterministic jitter
                lat += (dfloat01("j1", uid, day_idx, sess_idx)-0.5)*0.03
                lon += (dfloat01("j2", uid, day_idx, sess_idx)-0.5)*0.03

            visit_cnt += 1

            # Deterministic VPN flag (~8%)
            vpn = 1 if dfloat01("vpn", uid, day_idx, sess_idx) < 0.08 else 0

            # IP choice: if vpn -> any; else -> ISP pool
            ip_int = rand_ip(None if vpn else p["isp_pool"], "ip", uid, day_idx, sess_idx)

            # Derived ip_changed (vs previous session)
            ip_changed = 1 if (prev_ip is not None and ip_int != prev_ip) else 0
            prev_ip = ip_int

            # Deterministic failed login conditions
            monday_morning = (wd == 0 and off < 60 and dfloat01("mMo",uid,day_idx) < 0.1)
            stale_gap = (gap_mins > 2880)
            failed_log = 1 if (monday_morning or stale_gap) else 0

            # Historic risk (kept for compatibility, deterministic)
            risk = dgauss(p["typing_base"]/2 + failed_log*5 + vpn*8, 4, "risk", uid, day_idx, sess_idx)
            risk = round(max(0, min(100, risk)), 2)

            # Overseas heuristic (deterministic, ~10% domestic → overseas via VPN/proxy flag)
            overseas = 1 if (vpn and dfloat01("ovr", uid, day_idx, sess_idx) < 0.125) else 0

            # Stability bonus for trust (smaller if IP changed)
            stability_bonus = 0.16 if ip_changed == 0 else 0.08

            # Variation in [0,1)
            variation = dfloat01("var", uid, day_idx, sess_idx)

            # Base trust starts low
            base_trust = 0.20

            # Compute trust (deterministic)
            trust = trust_from_features(
                base=base_trust,
                ip_changed=ip_changed,
                overseas=overseas,
                is_attacker=p["is_attacker"],
                stability_bonus=stability_bonus,
                variation=variation
            )

            # ip reputation and typing, click remain deterministic
            ip_rep = 1
            w = dfloat01("rep", uid, day_idx, sess_idx)
            if w > 0.95: ip_rep = 3
            elif w > 0.75: ip_rep = 2

            typing_speed = max(10, int(round(dgauss(p["typing_base"], 10, "type", uid, day_idx, sess_idx))))
            click_pref  = p["click_pref"]

            rows.append([
                p["rt_code"], p["os_code"], p["dev_type"], p["man_code"], p["is_rooted"],
                round(lat,6), round(lon,6), dint(8,25,"conf",uid,day_idx,sess_idx), visit_cnt, p["shift_code"],
                login_ep, sess_len, gap_mins, 1 if wd>=5 else 0, ip_int,
                ip_rep, vpn, typing_speed, click_pref,
                p["role_code"], p["scope_code"], failed_log, risk, 0,  # system_mode_code kept = 0
                p["is_attacker"], trust  # << minimal extension at end
            ])
    return rows

# ----------------------------
# Headers (original + minimal additions)
# ----------------------------
HEADERS = [
    "region_tz_code","os_code","device_type_code","manufacturer_code","is_rooted",
    "gps_latitude","gps_longitude","location_conf_radius","location_visit_count",
    "shift_profile_code","session_start_epoch","session_duration_mins",
    "time_since_last_login_mins","day_type_code","ip_address_as_int",
    "ip_reputation_code","vpn_tor_usage","typing_speed_cpm","click_pattern_code",
    "role_code","scope_code","failed_login_attempts","historic_risk_score",
    "system_mode_code",
    "is_attacker",          # << added (minimal)
    "trust_score"           # << added (minimal)
]

# ----------------------------
# Overlap injection (~5% in [0.65, 0.70])
# ----------------------------
def inject_trust_overlap(rows):
    total = len(rows)
    if total == 0:
        return rows
    target = max(1, int(round(total * 0.05)))  # ~5%

    # Build domestic indices (overseas cap would override anyway)
    # We'll alternate between normals and attackers to guarantee small overlap.
    norm_idx = [i for i,r in enumerate(rows) if r[-2] == 0 and r[16] == 0]  # is_attacker==0 and vpn_tor_usage==0 approx domestic
    atk_idx  = [i for i,r in enumerate(rows) if r[-2] == 1 and r[16] == 0]

    band_vals = []
    if target == 1:
        band_vals = [0.675]
    else:
        step = (0.70 - 0.65) / (target - 1)
        band_vals = [round(0.65 + k*step, 4) for k in range(target)]

    take_norm = True
    ni = ai = k = 0
    while k < target and (ni < len(norm_idx) or ai < len(atk_idx)):
        if take_norm and ni < len(norm_idx):
            idx = norm_idx[ni]; ni += 1
        elif not take_norm and ai < len(atk_idx):
            idx = atk_idx[ai]; ai += 1
        elif ni < len(norm_idx):
            idx = norm_idx[ni]; ni += 1
        elif ai < len(atk_idx):
            idx = atk_idx[ai]; ai += 1
        else:
            break

        # Set trust into overlap band
        rows[idx][-1] = band_vals[k]  # trust_score at end
        take_norm = not take_norm
        k += 1

    return rows

# ----------------------------
# CSV writer (unchanged behavior)
# ----------------------------
def write_csv(name="sample_data.csv", users=100, rows_per_user=1000):
    all_rows = []
    with open(name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        for uid in range(users):
            profile = make_profile(uid)
            data = generate_rows(profile, uid, rows_per_user)
            all_rows.extend(data)

        # Inject ~5% overlap in place (after building everything)
        all_rows = inject_trust_overlap(all_rows)

        writer.writerows(all_rows)
    print(f"[+] {users * rows_per_user:,} rows written → {name}")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    users = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rows_per_user = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    write_csv(name="sample_data.csv", users=users, rows_per_user=rows_per_user)