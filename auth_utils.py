import hashlib
import pandas as pd
import os

USER_DB_PATH = 'data/users.csv'

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

def create_user_db():
    if not os.path.exists(USER_DB_PATH):
        df = pd.DataFrame(columns=['username', 'password_hash'])
        df.to_csv(USER_DB_PATH, index=False)

def add_user(username, password):
    create_user_db()
    df = pd.read_csv(USER_DB_PATH)
    if username in df['username'].values:
        return False # User already exists
    
    new_user = pd.DataFrame({
        'username': [username],
        'password_hash': [make_hashes(password)]
    })
    
    # Append
    # We load, append, save
    df = pd. concat([df, new_user], ignore_index=True)
    df.to_csv(USER_DB_PATH, index=False)
    return True

def login_user(username, password):
    create_user_db()
    df = pd.read_csv(USER_DB_PATH)
    
    if username not in df['username'].values:
        return False
    
    # Get hash
    stored_hash = df[df['username'] == username]['password_hash'].values[0]
    
    return check_hashes(password, stored_hash)
