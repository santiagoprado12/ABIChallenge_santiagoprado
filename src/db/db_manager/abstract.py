"""Module with the interface to perform operations in any database."""

from abc import ABC, abstractmethod


class InterfaceDatabaseManager(ABC):
    """Abstract class to manage database connections and operations."""

    @abstractmethod
    def connect(self):
        """Connect to the database."""
        pass

    @abstractmethod
    def execute_query(self, query, params=None):
        """Execute a query against the database."""
        pass

    @abstractmethod
    def fetch_results(self, query, params=None):
        """Fetch results from a query."""
        pass

    @abstractmethod
    def close(self):
        """Close the database connection."""
        pass
