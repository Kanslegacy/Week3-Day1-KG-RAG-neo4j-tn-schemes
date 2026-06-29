"""
Quick sanity check: confirms your .env credentials can actually
reach your Neo4j AuraDB instance before we build real logic on top of it.
"""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

print("URI:", os.getenv("NEO4J_URI"))
print("USERNAME:", os.getenv("NEO4J_USERNAME"))
print("PASSWORD:", repr(os.getenv("NEO4J_PASSWORD")))

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

try:
    driver.verify_connectivity()
    print("✅ Connected to Neo4j AuraDB successfully!")
except Exception as e:
    print("❌ Connection failed:", e)
finally:
    driver.close()