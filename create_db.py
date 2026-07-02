import sqlite3

def create_database():
    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    # Create the users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        membership_tier TEXT NOT NULL
    )
    """)

    # Remove existing data to avoid duplicate entries
    cursor.execute("DELETE FROM users")

    # Sample users
    users = [
        (101, "Riya Sharma", "Gold"),
        (102, "Aman Verma", "Silver"),
        (103, "Neha Iyer", "Platinum")
    ]

    # Insert sample data
    cursor.executemany(
        "INSERT INTO users (user_id, name, membership_tier) VALUES (?, ?, ?)",
        users
    )

    # Save changes and close connection
    conn.commit()
    conn.close()

    print("✅ Database created successfully!")
    print("Database file: users.db")
    print("Sample users inserted.")

if __name__ == "__main__":
    create_database()