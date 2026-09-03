from app.db.mongodb import client, db, collection


def main():
    print("Testing MongoDB connection...")

    result = client.admin.command("ping")

    print("MongoDB:", result)
    print("Database:", db.name)
    print("Collection:", collection.name)


if __name__ == "__main__":
    main()