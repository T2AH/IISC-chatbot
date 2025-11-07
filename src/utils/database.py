import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from typing import List, Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger('database')

class DatabaseManager:
    def __init__(self, connection_params):
        self.conn_params = connection_params
        self.connection = None
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(**self.conn_params)
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from database")
    
    def execute_query(self, query, params=None):
        """Execute a query"""
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params)
            self.connection.commit()
            return cursor.fetchall()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Query execution error: {e}")
            raise
        finally:
            cursor.close()
