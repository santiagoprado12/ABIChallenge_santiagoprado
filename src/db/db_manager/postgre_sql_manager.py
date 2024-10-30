"""Module for managing a PostgreSQL database connection."""

import logging
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv

from src.db.db_manager.abstract import InterfaceDatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostgreSQLManager(InterfaceDatabaseManager):
    """PostgreSQL-specific implementation of the DatabaseManager.

    This class handles connecting to a PostgreSQL database, executing queries,
    fetching results, and converting query results to a pandas DataFrame.
    """

    def __init__(self):
        """Load the necessary PostgreSQL connection credentials from the .env file.

        Attributes:
            host (str): The host address of the PostgreSQL server.
            dbname (str): The name of the PostgreSQL database to connect to.
            user (str): The PostgreSQL username.
            password (str): The PostgreSQL user's password.
            port (str): The PostgreSQL server port, defaulting to 5432.
        """
        # Load environment variables from the .env file
        load_dotenv()

        # Get the PostgreSQL credentials from environment variables
        self.host = os.getenv("POSTGRES_HOST")
        self.dbname = os.getenv("POSTGRES_DB")
        self.user = os.getenv("POSTGRES_USER")
        self.password = os.getenv("POSTGRES_PASSWORD")
        self.port = os.getenv("POSTGRES_PORT", "5432")  # Default port is 5432

        self.connection = None
        self.cursor = None

    def connect(self):
        """Establish a connection to the PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                port=self.port,
            )
            self.cursor = self.connection.cursor()
            logger.info("Connected to PostgreSQL database.")
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise

    def execute_query(self, query, params=None):
        """Execute a query on the PostgreSQL database.

        Parameters:
            query (str): The SQL query to be executed.
            params (tuple, optional): Optional parameters to include in the query.

        Raises:
            Exception: If there is an error during query execution.
        """
        try:
            self.cursor.execute(query, params)
            self.connection.commit()
            logger.info("Query executed successfully.")
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    def fetch_results(self, query, params=None):
        """Execute a query and fetch the results.

        Parameters:
            query (str): The SQL query to be executed.
            params (tuple, optional): Optional parameters to include in the query.

        Returns:
            list: A list of tuples representing the rows returned by the query.

        Raises:
            Exception: If there is an error during query execution or fetching results.
        """
        try:
            self.execute_query(query, params)
            results = self.cursor.fetchall()
            logger.info(f"Fetched {len(results)} rows.")
            return results
        except Exception as e:
            logger.error(f"Error fetching results: {e}")
            raise

    def fetch_to_dataframe(self, query, params=None):
        """Execute a query and fetch the results as a pandas DataFrame.

        Parameters:
            query (str): The SQL query to be executed.
            params (tuple, optional): Optional parameters to include in the query.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the query results.

        Raises:
            Exception: If there is an error during query execution or fetching results.
        """
        try:
            self.execute_query(query, params)
            columns = [desc[0] for desc in self.cursor.description]  # Get column names
            rows = self.cursor.fetchall()
            df = pd.DataFrame(rows, columns=columns)
            logger.info(f"Fetched {len(df)} rows into a DataFrame.")
            return df
        except Exception as e:
            logger.error(f"Error fetching data as DataFrame: {e}")
            raise

    def close(self):
        """Close the cursor and the PostgreSQL connection."""
        if self.cursor:
            self.cursor.close()
            logger.info("Cursor closed.")
        if self.connection:
            self.connection.close()
            logger.info("PostgreSQL connection closed.")

    def upload_dataframe_to_postgres(self, df: pd.DataFrame, table_name: str):
        """Uploads a Pandas DataFrame to a PostgreSQL table.

        Parameters:
            df (pd.DataFrame): The DataFrame containing data to upload.
            table_name (str): The name of the target table in PostgreSQL.
        """
        # Create a connection to PostgreSQL
        try:

            self.connect()

            cursor = self.connection.cursor()

            # Create an insert query template
            columns = ", ".join(df.columns)
            values_template = ", ".join(["%s"] * len(df.columns))
            insert_query = (
                f"INSERT INTO {table_name} ({columns}) VALUES ({values_template})"
            )

            # Convert the DataFrame to a list of tuples
            data_tuples = [tuple(row) for row in df.to_numpy()]

            # Execute the insert query for all rows in the DataFrame
            cursor.executemany(insert_query, data_tuples)

            # Commit the transaction
            self.connection.commit()
            print(f"Data uploaded successfully to {table_name}")

        except Exception as error:
            print(f"Error occurred: {error}")

        finally:
            # Close the cursor and connection
            if cursor:
                cursor.close()
            if self.connection:
                self.connection.close()


# Example usage:
if __name__ == "__main__":
    p_manager = PostgreSQLManager()

    try:
        p_manager.connect()

        # Example query to fetch data into pandas DataFrame
        df = p_manager.fetch_to_dataframe("SELECT * FROM titanic;")
        print(df)

    finally:
        p_manager.close()
