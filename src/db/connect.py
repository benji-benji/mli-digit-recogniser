import psycopg2

def get_connection():
    return psycopg2.connect(
        dbname="digitdb",
        user="digituser",
        password="digitpass",
        host="db",  # later this might change to a Docker service name
        port="5432"
    )
