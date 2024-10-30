"""Integration test for the PostgreSQLManager class."""
import unittest
from unittest.mock import MagicMock, patch

import psycopg2

from src.db.db_manager.postgre_sql_manager import PostgreSQLManager


class TestPostgreSQLManager(unittest.TestCase):
    """Integration test for PostgreSQLManager."""

    @patch('src.db.db_manager.postgre_sql_manager.psycopg2.connect')
    def test_connect(self, mock_connect):
        """Test connecting to the PostgreSQL database."""
        db_manager = PostgreSQLManager()
        db_manager.connect()
        mock_connect.assert_called_once()

    @patch('src.db.db_manager.postgre_sql_manager.psycopg2.connect')
    def test_execute_query(self, mock_connect):
        """Test executing a query."""
        mock_conn = mock_connect.return_value
        mock_cursor = mock_conn.cursor.return_value

        db_manager = PostgreSQLManager()
        db_manager.connect()
        db_manager.execute_query("SELECT * FROM coin_data;")
        
        mock_cursor.execute.assert_called_once_with("SELECT * FROM coin_data;", None)
        mock_conn.commit.assert_called_once()

    @patch('src.db.db_manager.postgre_sql_manager.psycopg2.connect')
    def test_fetch_results(self, mock_connect):
        """Test fetching results from a query."""
        mock_conn = mock_connect.return_value
        mock_cursor = mock_conn.cursor.return_value

        mock_cursor.fetchall.return_value = [('bitcoin', 50000)]
        db_manager = PostgreSQLManager()
        db_manager.connect()
        results = db_manager.fetch_results("SELECT * FROM coin_data;")
        
        self.assertEqual(results, [('bitcoin', 50000)])

if __name__ == '__main__':
    unittest.main()
