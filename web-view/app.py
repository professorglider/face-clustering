from flask import Flask, render_template
import psycopg2

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    host="postgres",
    database="faces_db",
    user="postgres",
    password="postgres"
)

# Create a cursor
cur = conn.cursor()

# Define the Flask app
app = Flask(__name__)

# Define the root route
@app.route('/')
def index():
    # Select all faces from the database
    cur.execute("SELECT * FROM faces")

    # Fetch all faces
    rows = cur.fetchall()

    # Render the template with the faces data
    return render_template('index.html', faces=rows)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
