from enum import Enum
import sqlite3
import hashlib
import os

class UserRole(Enum):
    ADMIN = "admin"          # Full access to all features
    ANALYST = "analyst"      # Can view dashboards and analyze data
    OPERATOR = "operator"    # Can view dashboards only
    AUDITOR = "auditor"      # Can view logs and reports

class Permission(Enum):
    VIEW_DASHBOARD = "view_dashboard"
    MANAGE_USERS = "manage_users"
    TRAIN_MODEL = "train_model"
    VIEW_REPORTS = "view_reports"
    MODIFY_SETTINGS = "modify_settings"
    BATCH_ANALYSIS = "batch_analysis"
    REAL_TIME_MONITORING = "real_time_monitoring"

# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [p for p in Permission],
    UserRole.ANALYST: [
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_REPORTS,
        Permission.BATCH_ANALYSIS,
        Permission.REAL_TIME_MONITORING
    ],
    UserRole.OPERATOR: [
        Permission.VIEW_DASHBOARD,
        Permission.REAL_TIME_MONITORING
    ],
    UserRole.AUDITOR: [
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_REPORTS
    ]
}

class PasswordError(Exception):
    pass

def check_password_strength(password):
    """Check if password meets security requirements"""
    if len(password) < 8:
        raise PasswordError("Password must be at least 8 characters long")
    if not any(c.isupper() for c in password):
        raise PasswordError("Password must contain at least one uppercase letter")
    if not any(c.islower() for c in password):
        raise PasswordError("Password must contain at least one lowercase letter")
    if not any(c.isdigit() for c in password):
        raise PasswordError("Password must contain at least one number")
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        raise PasswordError("Password must contain at least one special character")

def change_password(username, old_password, new_password):
    """Change user password with validation"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Verify old password
    c.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    
    if not result or not verify_password(result[0], old_password):
        conn.close()
        return False, "Current password is incorrect"
    
    try:
        # Check password strength
        check_password_strength(new_password)
        
        # Update password
        new_password_hash = hash_password(new_password)
        c.execute(
            'UPDATE users SET password_hash = ? WHERE username = ?',
            (new_password_hash, username)
        )
        conn.commit()
        return True, "Password updated successfully"
    except PasswordError as e:
        return False, str(e)
    finally:
        conn.close()

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            failed_attempts INTEGER DEFAULT 0,
            last_failed_attempt TIMESTAMP,
            account_locked BOOLEAN DEFAULT 0,
            last_password_change TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create login attempts table
    c.execute('''
        CREATE TABLE IF NOT EXISTS login_attempts (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN NOT NULL,
            ip_address TEXT
        )
    ''')
    conn.commit()
    conn.close()

def record_login_attempt(username, success, ip_address):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Record the attempt
    c.execute(
        'INSERT INTO login_attempts (username, success, ip_address) VALUES (?, ?, ?)',
        (username, success, ip_address)
    )
    
    if not success:
        # Update failed attempts counter
        c.execute('''
            UPDATE users 
            SET failed_attempts = failed_attempts + 1,
                last_failed_attempt = CURRENT_TIMESTAMP
            WHERE username = ?
        ''', (username,))
        
        # Check if account should be locked
        c.execute(
            'SELECT failed_attempts FROM users WHERE username = ?',
            (username,)
        )
        attempts = c.fetchone()[0]
        
        if attempts >= 5:  # Lock after 5 failed attempts
            c.execute(
                'UPDATE users SET account_locked = 1 WHERE username = ?',
                (username,)
            )
    else:
        # Reset failed attempts on successful login
        c.execute(
            'UPDATE users SET failed_attempts = 0, account_locked = 0 WHERE username = ?',
            (username,)
        )
    
    conn.commit()
    conn.close()

def hash_password(password):
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )
    return salt + key

def verify_password(stored_password, provided_password):
    salt = stored_password[:32]
    stored_key = stored_password[32:]
    key = hashlib.pbkdf2_hmac(
        'sha256',
        provided_password.encode('utf-8'),
        salt,
        100000
    )
    return key == stored_key

def create_user(username, password, role):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        password_hash = hash_password(password)
        c.execute(
            'INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
            (username, password_hash, role.value)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password_hash, role FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    
    if result and verify_password(result[0], password):
        return result[1]  # Return user role
    return None

def has_permission(role, permission):
    user_role = UserRole(role)
    return permission in ROLE_PERMISSIONS[user_role] 