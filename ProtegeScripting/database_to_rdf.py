import sqlite3
from rdflib import Graph, Literal, Namespace, RDF, URIRef

print("Starting script execution...")

try:
    # Step 1: Connect to SQLite
    print("Connecting to SQLite database...")
    conn = sqlite3.connect("fake_data.db")
    print("Database connected.")

    cursor = conn.cursor()

    # Step 2: Create table
    print("Creating table...")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        position TEXT,
        salary INTEGER
    )
    """)
    print("Table created.")

    # Step 3: Delete existing data (to avoid duplicates)
    print("Clearing existing data...")
    cursor.execute("DELETE FROM employees")  # Removes all old records
    conn.commit()
    print("Old data cleared.")

    # Step 4: Insert new fake data
    print("Inserting fake data...")
    fake_data = [
        (1, "Alice", "Developer", 70000),
        (2, "Bob", "Manager", 90000),
        (3, "Charlie", "Analyst", 60000),
        (4, "Madeleine", "Information Technology", 75000)
    ]
    cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?)", fake_data)
    conn.commit()
    print("Data inserted.")

    # Step 4: Fetch data from database
    print("Fetching data...")
    cursor.execute("SELECT * FROM employees")
    rows = cursor.fetchall()
    print(f"Fetched rows: {rows}")

    # Step 5: Generate RDF
    print("Generating RDF graph...")
    EX = Namespace("http://example.org/employees/")
    g = Graph()

    for row in rows:
        employee_uri = URIRef(EX[f"employee_{row[0]}"])
        g.add((employee_uri, RDF.type, EX.Employee))
        g.add((employee_uri, EX.name, Literal(row[1])))
        g.add((employee_uri, EX.position, Literal(row[2])))
        g.add((employee_uri, EX.salary, Literal(row[3])))

    print("Saving RDF file...")
    rdf_file = "employees.rdf"
    g.serialize(destination=rdf_file, format="xml")
    print(f"RDF data saved to {rdf_file}")

except Exception as e:
    print(f"An error occurred: {e}")

print("Script execution finished.")
