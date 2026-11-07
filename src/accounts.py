"""SQLite-backed student accounts and saved semester-one score profiles."""
import os, sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from src.config import BASE_DIR

DB = os.path.join(BASE_DIR, 'instance', 'accounts.db')

def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c

def init_db():
    with connect() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT, matric_no TEXT UNIQUE NOT NULL,
          full_name TEXT NOT NULL, password_hash TEXT NOT NULL,
          role TEXT NOT NULL DEFAULT 'student', created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS activity_logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, event TEXT NOT NULL,
          details TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(user_id) REFERENCES users(id))''')
        c.execute('''CREATE TABLE IF NOT EXISTS prediction_logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, prediction_type TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'success', duration_ms REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(user_id) REFERENCES users(id))''')
        c.execute('''CREATE TABLE IF NOT EXISTS semester_scores (
          id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
          scores_json TEXT NOT NULL, average REAL NOT NULL,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY(user_id) REFERENCES users(id))''')
        admin_user=os.getenv('ADMIN_USERNAME','admin').strip().lower()
        admin_pass=os.getenv('ADMIN_PASSWORD','')
        if admin_pass and not c.execute('SELECT 1 FROM users WHERE matric_no=?',(admin_user,)).fetchone():
            c.execute('INSERT INTO users(matric_no,full_name,password_hash,role) VALUES(?,?,?,?)',
                      (admin_user,'System Administrator',generate_password_hash(admin_pass),'admin'))

def create_user(matric, name, password):
    try:
        with connect() as c:
            c.execute('INSERT INTO users(matric_no,full_name,password_hash) VALUES(?,?,?)',
                      (matric.strip().upper(),name.strip(),generate_password_hash(password)))
        return True
    except sqlite3.IntegrityError: return False

def authenticate(matric,password):
    with connect() as c: u=c.execute('SELECT * FROM users WHERE matric_no=?',(matric.strip().lower(),)).fetchone()
    if not u:
        with connect() as c: u=c.execute('SELECT * FROM users WHERE matric_no=?',(matric.strip().upper(),)).fetchone()
    return dict(u) if u and check_password_hash(u['password_hash'],password) else None

def save_scores(user_id,scores,average):
    import json
    with connect() as c:
        c.execute('INSERT INTO semester_scores(user_id,scores_json,average) VALUES(?,?,?)',
                  (user_id,json.dumps(scores),average))

def get_scores(user_id):
    import json
    with connect() as c: rows=c.execute('SELECT * FROM semester_scores WHERE user_id=? ORDER BY id DESC',(user_id,)).fetchall()
    return [{**dict(r),'scores':json.loads(r['scores_json'])} for r in rows]

def admin_stats():
    with connect() as c:
        students=c.execute("SELECT COUNT(*) n FROM users WHERE role='student'").fetchone()['n']
        submissions=c.execute('SELECT COUNT(*) n FROM semester_scores').fetchone()['n']
        avg=c.execute('SELECT AVG(average) a FROM semester_scores').fetchone()['a']
    return {'students':students,'submissions':submissions,'average':round(avg,1) if avg is not None else None}


def get_students():
    """Return student account summaries for the admin dashboard."""
    with connect() as c:
        rows=c.execute(
            """SELECT u.id,u.matric_no,u.full_name,u.created_at,
                      COUNT(s.id) AS submissions, MAX(s.average) AS best_average,
                      (SELECT average FROM semester_scores WHERE user_id=u.id ORDER BY id DESC LIMIT 1) AS latest_average
               FROM users u LEFT JOIN semester_scores s ON s.user_id=u.id
               WHERE u.role='student' GROUP BY u.id ORDER BY u.full_name COLLATE NOCASE"""
        ).fetchall()
    return [dict(r) for r in rows]


def log_activity(user_id, event, details=None):
    """Record a privacy-conscious application event for admin analytics."""
    import json
    with connect() as c:
        c.execute('INSERT INTO activity_logs(user_id,event,details) VALUES(?,?,?)',
                  (user_id, event, json.dumps(details or {})))

def log_prediction(user_id, prediction_type='course_analysis', status='success', duration_ms=None):
    with connect() as c:
        c.execute('INSERT INTO prediction_logs(user_id,prediction_type,status,duration_ms) VALUES(?,?,?,?)',
                  (user_id, prediction_type, status, duration_ms))

def admin_usage():
    """Aggregated usage metrics and recent events; no passwords or score payloads."""
    with connect() as c:
        active_today = c.execute("SELECT COUNT(DISTINCT user_id) n FROM activity_logs WHERE date(created_at)=date('now','localtime') AND user_id IS NOT NULL").fetchone()['n']
        logins_today = c.execute("SELECT COUNT(*) n FROM activity_logs WHERE event='login_success' AND date(created_at)=date('now','localtime')").fetchone()['n']
        predictions = c.execute('SELECT COUNT(*) n FROM prediction_logs').fetchone()['n']
        predictions_today = c.execute("SELECT COUNT(*) n FROM prediction_logs WHERE date(created_at)=date('now','localtime')").fetchone()['n']
        failed_predictions = c.execute("SELECT COUNT(*) n FROM prediction_logs WHERE status!='success'").fetchone()['n']
        daily = c.execute("""SELECT date(created_at) day, COUNT(*) count FROM prediction_logs
                            WHERE created_at >= datetime('now','-6 days') GROUP BY date(created_at) ORDER BY day""").fetchall()
        recent = c.execute("""SELECT a.event,a.details,a.created_at,COALESCE(u.full_name,'System') full_name,
                             COALESCE(u.matric_no,'—') matric_no FROM activity_logs a
                             LEFT JOIN users u ON u.id=a.user_id ORDER BY a.id DESC LIMIT 15""").fetchall()
        top = c.execute("""SELECT u.full_name,u.matric_no,COUNT(p.id) predictions
                          FROM users u LEFT JOIN prediction_logs p ON p.user_id=u.id
                          WHERE u.role='student' GROUP BY u.id ORDER BY predictions DESC LIMIT 5""").fetchall()
    return {'active_today':active_today,'logins_today':logins_today,'predictions':predictions,
            'predictions_today':predictions_today,'failed_predictions':failed_predictions,
            'daily':[dict(x) for x in daily],'recent':[dict(x) for x in recent],
            'top_students':[dict(x) for x in top]}
