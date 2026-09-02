import os
import mysql.connector
import pytest


@pytest.fixture(scope="session")
def conn():
    connection = mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST", "localhost"),
        port=int(os.environ.get("MYSQL_PORT", 3306)),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ["MYSQL_ROOT_PASSWORD"],
        database=os.environ.get("MYSQL_DATABASE", "Insidersignal"),
    )
    yield connection
    connection.close()
