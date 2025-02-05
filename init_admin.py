from auth.user_management import init_db, create_user, UserRole

def initialize_admin():
    init_db()
    # Create default admin user
    admin_created = create_user(
        username="admin",
        password="changeme123",  # Change this in production
        role=UserRole.ADMIN
    )
    if admin_created:
        print("Admin user created successfully")
        print("Username: admin")
        print("Password: changeme123")
        print("Please change the password after first login")
    else:
        print("Admin user already exists")

if __name__ == "__main__":
    initialize_admin() 