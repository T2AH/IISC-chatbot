"""
Chat Persistence Database Client
Manages chat threads and message history using SQLite
"""

import sqlite3
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from loguru import logger
import threading


class ChatDB:
    """SQLite database for persistent chat thread management"""
    
    def __init__(self, db_path: str = "./data/chat_threads.db"):
        """
        Initialize chat database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe connection handling
        self._local = threading.local()
        
        # Initialize database schema
        self._init_database()
        logger.info(f"ChatDB initialized at {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Chat threads table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_threads (
                thread_id TEXT PRIMARY KEY,
                user_id TEXT DEFAULT 'default',
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                message_count INTEGER DEFAULT 0
            )
        """)
        
        # Chat messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                context_used INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (thread_id) REFERENCES chat_threads(thread_id) ON DELETE CASCADE
            )
        """)
        
        # Message contexts table (for debugging/analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message_contexts (
                context_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                chunk_url TEXT,
                chunk_text TEXT,
                relevance_score REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (message_id) REFERENCES chat_messages(message_id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_threads_updated 
            ON chat_threads(updated_at DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_thread 
            ON chat_messages(thread_id, created_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_contexts_message 
            ON message_contexts(message_id)
        """)
        
        conn.commit()
        logger.success("Database schema initialized")
    
    def create_thread(self, initial_query: str, user_id: str = "default") -> str:
        """
        Create a new chat thread
        
        Args:
            initial_query: First user query to generate title
            user_id: Optional user identifier
        
        Returns:
            thread_id: New thread identifier
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        thread_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # Generate title from initial query (first 50 chars)
        title = initial_query[:50] + ("..." if len(initial_query) > 50 else "")
        
        cursor.execute("""
            INSERT INTO chat_threads (thread_id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (thread_id, user_id, title, now, now))
        
        conn.commit()
        logger.info(f"Created new thread: {thread_id}")
        
        return thread_id
    
    def add_message(
        self, 
        thread_id: str, 
        role: str, 
        content: str,
        context_used: int = 0,
        contexts: List[Dict] = None
    ) -> str:
        """
        Add a message to a thread
        
        Args:
            thread_id: Thread identifier
            role: 'user' or 'assistant'
            content: Message content
            context_used: Number of context chunks used
            contexts: Optional list of context chunks for debugging
        
        Returns:
            message_id: New message identifier
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # Insert message
        cursor.execute("""
            INSERT INTO chat_messages (message_id, thread_id, role, content, context_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (message_id, thread_id, role, content, context_used, now))
        
        # Store context chunks if provided
        if contexts:
            for ctx in contexts:
                context_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO message_contexts (context_id, message_id, chunk_url, chunk_text, relevance_score, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    context_id,
                    message_id,
                    ctx.get('url', ''),
                    ctx.get('text', '')[:1000],  # Limit to 1000 chars
                    ctx.get('score', 0.0),
                    now
                ))
        
        # Update thread metadata
        cursor.execute("""
            UPDATE chat_threads 
            SET updated_at = ?, message_count = message_count + 1
            WHERE thread_id = ?
        """, (now, thread_id))
        
        conn.commit()
        
        return message_id
    
    def get_thread_messages(self, thread_id: str, limit: int = 100) -> List[Dict]:
        """
        Get all messages in a thread
        
        Args:
            thread_id: Thread identifier
            limit: Maximum number of messages to retrieve
        
        Returns:
            List of message dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT message_id, role, content, context_used, created_at
            FROM chat_messages
            WHERE thread_id = ?
            ORDER BY created_at ASC
            LIMIT ?
        """, (thread_id, limit))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                'message_id': row['message_id'],
                'role': row['role'],
                'content': row['content'],
                'context_used': row['context_used'],
                'created_at': row['created_at']
            })
        
        return messages
    
    def get_all_threads(
        self, 
        user_id: str = "default", 
        status: str = "active",
        limit: int = 50
    ) -> List[Dict]:
        """
        Get all chat threads for a user
        
        Args:
            user_id: User identifier
            status: Thread status filter ('active' or 'archived')
            limit: Maximum number of threads
        
        Returns:
            List of thread dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT thread_id, title, created_at, updated_at, message_count
            FROM chat_threads
            WHERE user_id = ? AND status = ?
            ORDER BY updated_at DESC
            LIMIT ?
        """, (user_id, status, limit))
        
        threads = []
        for row in cursor.fetchall():
            threads.append({
                'thread_id': row['thread_id'],
                'title': row['title'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'message_count': row['message_count']
            })
        
        return threads
    
    def get_thread_info(self, thread_id: str) -> Optional[Dict]:
        """
        Get information about a specific thread
        
        Args:
            thread_id: Thread identifier
        
        Returns:
            Thread dictionary or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT thread_id, user_id, title, created_at, updated_at, status, message_count
            FROM chat_threads
            WHERE thread_id = ?
        """, (thread_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                'thread_id': row['thread_id'],
                'user_id': row['user_id'],
                'title': row['title'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'status': row['status'],
                'message_count': row['message_count']
            }
        return None
    
    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a chat thread (and all its messages)
        
        Args:
            thread_id: Thread identifier
        
        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM chat_threads WHERE thread_id = ?", (thread_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        
        if deleted:
            logger.info(f"Deleted thread: {thread_id}")
        
        return deleted
    
    def archive_thread(self, thread_id: str) -> bool:
        """
        Archive a thread (soft delete)
        
        Args:
            thread_id: Thread identifier
        
        Returns:
            True if archived, False if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE chat_threads 
            SET status = 'archived'
            WHERE thread_id = ?
        """, (thread_id,))
        
        archived = cursor.rowcount > 0
        conn.commit()
        
        return archived
    
    def cleanup_old_threads(self, days: int = 30) -> int:
        """
        Delete threads older than specified days
        
        Args:
            days: Age threshold in days
        
        Returns:
            Number of threads deleted
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            DELETE FROM chat_threads
            WHERE updated_at < ? AND status = 'archived'
        """, (cutoff_date,))
        
        deleted = cursor.rowcount
        conn.commit()
        
        logger.info(f"Cleaned up {deleted} old threads")
        return deleted
    
    def search_threads(self, query: str, user_id: str = "default") -> List[Dict]:
        """
        Search threads by title or content
        
        Args:
            query: Search query
            user_id: User identifier
        
        Returns:
            List of matching threads
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        search_pattern = f"%{query}%"
        
        cursor.execute("""
            SELECT DISTINCT t.thread_id, t.title, t.created_at, t.updated_at, t.message_count
            FROM chat_threads t
            LEFT JOIN chat_messages m ON t.thread_id = m.thread_id
            WHERE t.user_id = ? AND t.status = 'active'
            AND (t.title LIKE ? OR m.content LIKE ?)
            ORDER BY t.updated_at DESC
            LIMIT 20
        """, (user_id, search_pattern, search_pattern))
        
        threads = []
        for row in cursor.fetchall():
            threads.append({
                'thread_id': row['thread_id'],
                'title': row['title'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'message_count': row['message_count']
            })
        
        return threads
    
    def get_conversation_history(self, thread_id: str, limit: int = 20) -> List[Dict]:
        """
        Get conversation history in LangGraph format
        
        Args:
            thread_id: Thread identifier
            limit: Maximum number of messages
        
        Returns:
            List of {role, content} dictionaries
        """
        messages = self.get_thread_messages(thread_id, limit)
        
        # Convert to LangGraph format
        history = []
        for msg in messages:
            history.append({
                'role': msg['role'],
                'content': msg['content']
            })
        
        return history
    
    def close(self):
        """Close database connection"""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            delattr(self._local, 'conn')
