import csv

# Fields to generate _low/_high columns for (detected from first row or hardcoded)
NUMERIC_FIELDS = [
    "latitude", "longitude", "availableMemory", "rssi",
    "Processors", "Battery", "accel", "gyro", "magnet",
    "screenWidth", "screenLength", "screenDensity"
]

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False

def generate_range_fields(input_path="base.csv", output_path="base2.csv"):
    with open(input_path, newline="") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    output_rows = []
    for row in rows:
        new_row = dict(row)  # Copy existing values
        for field in NUMERIC_FIELDS:
            value = row.get(field)
            if value and is_number(value):
                num = float(value)
                delta = num * 0.05
                new_row[f"{field}_low"] = round(num - delta, 6)
                new_row[f"{field}_high"] = round(num + delta, 6)
        output_rows.append(new_row)

    # Define output fieldnames
    fieldnames = list(rows[0].keys())
    for field in NUMERIC_FIELDS:
        fieldnames.append(f"{field}_low")
        fieldnames.append(f"{field}_high")

    with open(output_path, mode="w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"✔️ Done. Saved with ranges to: {output_path}")

if __name__ == "__main__":
    generate_range_fields("base.csv", "base2.csv")
