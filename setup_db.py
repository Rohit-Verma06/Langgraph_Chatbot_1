import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

load_dotenv()

print("Connecting to database...")
DB_URL = os.getenv("SUPABASE_DB_URL")
pool = ConnectionPool(conninfo=DB_URL, kwargs={"autocommit": True})

print("Setting up Checkpoint tables...")
checkpointer = PostgresSaver(pool)
checkpointer.setup()

print("Setting up Store tables...")
store = PostgresStore(conn=pool)
store.setup()

print("✅ Database setup complete! You can now start Streamlit.")