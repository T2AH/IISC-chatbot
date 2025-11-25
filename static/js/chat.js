/**
 * IISc Research Assistant - Chat & Thread Management
 */

const API_URL = 'http://localhost:8080';

// State management
let currentThreadId = null;
let currentSessionId = null;
let threads = [];
let isLoading = false;

// DOM Elements
const elements = {
    threadList: null,
    chatMessages: null,
    messageInput: null,
    sendButton: null,
    thinkingIndicator: null,
    welcomeMessage: null,
    newChatBtn: null,
    searchThreads: null,
    chatTitle: null,
    chatSubtitle: null,
    deleteBtn: null,
    archiveBtn: null,
    deleteModal: null,
    confirmDelete: null,
    cancelDelete: null,
    contextInfo: null,
    mobileMenuBtn: null,
    sidebar: null,
    toggleSidebar: null
};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    setupEventListeners();
    loadThreads();
    autoResizeTextarea();
});

// Initialize DOM element references
function initializeElements() {
    elements.threadList = document.getElementById('threadList');
    elements.chatMessages = document.getElementById('chatMessages');
    elements.messageInput = document.getElementById('messageInput');
    elements.sendButton = document.getElementById('sendButton');
    elements.thinkingIndicator = document.getElementById('thinkingIndicator');
    elements.welcomeMessage = document.getElementById('welcomeMessage');
    elements.newChatBtn = document.getElementById('newChatBtn');
    elements.searchThreads = document.getElementById('searchThreads');
    elements.chatTitle = document.getElementById('chatTitle');
    elements.chatSubtitle = document.getElementById('chatSubtitle');
    elements.deleteBtn = document.getElementById('deleteBtn');
    elements.archiveBtn = document.getElementById('archiveBtn');
    elements.deleteModal = document.getElementById('deleteModal');
    elements.confirmDelete = document.getElementById('confirmDelete');
    elements.cancelDelete = document.getElementById('cancelDelete');
    elements.contextInfo = document.getElementById('contextInfo');
    elements.mobileMenuBtn = document.getElementById('mobileMenuBtn');
    elements.sidebar = document.getElementById('sidebar');
    elements.toggleSidebar = document.getElementById('toggleSidebar');
}

// Setup event listeners
function setupEventListeners() {
    // Send message
    elements.sendButton.addEventListener('click', handleSendMessage);
    elements.messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });
    
    // New chat
    elements.newChatBtn.addEventListener('click', startNewChat);
    
    // Search threads
    elements.searchThreads.addEventListener('input', handleSearchThreads);
    
    // Example queries
    document.querySelectorAll('.example-query').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.getAttribute('data-query');
            elements.messageInput.value = query;
            handleSendMessage();
        });
    });
    
    // Delete/Archive buttons
    elements.deleteBtn.addEventListener('click', showDeleteModal);
    elements.archiveBtn.addEventListener('click', handleArchiveThread);
    elements.confirmDelete.addEventListener('click', handleDeleteThread);
    elements.cancelDelete.addEventListener('click', hideDeleteModal);
    
    // Mobile menu
    elements.mobileMenuBtn.addEventListener('click', toggleSidebar);
    elements.toggleSidebar.addEventListener('click', toggleSidebar);
    
    // Click outside modal to close
    elements.deleteModal.addEventListener('click', (e) => {
        if (e.target === elements.deleteModal) {
            hideDeleteModal();
        }
    });
}

// Load threads from API
async function loadThreads() {
    try {
        const response = await fetch(`${API_URL}/api/threads`);
        if (!response.ok) throw new Error('Failed to load threads');
        
        threads = await response.json();
        renderThreadList(threads);
    } catch (error) {
        console.error('Error loading threads:', error);
        elements.threadList.innerHTML = `
            <div class="no-threads">
                <p>Failed to load threads</p>
                <p style="font-size: 12px;">${error.message}</p>
            </div>
        `;
    }
}

// Render thread list
function renderThreadList(threadList) {
    if (!threadList || threadList.length === 0) {
        elements.threadList.innerHTML = `
            <div class="no-threads">
                <svg width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="currentColor">
                    <rect x="8" y="8" width="32" height="32" rx="4" stroke-width="2"/>
                    <path d="M16 20h16M16 28h10" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <p>No conversations yet</p>
                <p style="font-size: 12px;">Start a new chat to begin</p>
            </div>
        `;
        return;
    }
    
    elements.threadList.innerHTML = threadList.map(thread => `
        <div class="thread-item ${thread.thread_id === currentThreadId ? 'active' : ''}" 
             data-thread-id="${thread.thread_id}"
             onclick="loadThread('${thread.thread_id}')">
            <h4>${escapeHtml(thread.title)}</h4>
            <p>${thread.message_count} messages • ${formatTime(thread.updated_at)}</p>
        </div>
    `).join('');
}

// Load specific thread
async function loadThread(threadId) {
    if (isLoading) return;
    
    try {
        isLoading = true;
        currentThreadId = threadId;
        
        // Update UI
        updateThreadListActive(threadId);
        elements.welcomeMessage.style.display = 'none';
        elements.deleteBtn.style.display = 'block';
        elements.archiveBtn.style.display = 'block';
        
        // Fetch thread messages
        const response = await fetch(`${API_URL}/api/threads/${threadId}`);
        if (!response.ok) throw new Error('Failed to load thread');
        
        const data = await response.json();
        
        // Update header
        elements.chatTitle.textContent = data.title;
        elements.chatSubtitle.textContent = `${data.message_count} messages`;
        
        // Render messages
        elements.chatMessages.innerHTML = '';
        data.messages.forEach(msg => {
            addMessageToUI(msg.content, msg.role === 'user', msg.context_used);
        });
        
        scrollToBottom();
        
        // Close mobile sidebar
        if (window.innerWidth <= 768) {
            elements.sidebar.classList.remove('open');
        }
    } catch (error) {
        console.error('Error loading thread:', error);
        showError('Failed to load conversation');
    } finally {
        isLoading = false;
    }
}

// Start new chat
function startNewChat() {
    currentThreadId = null;
    currentSessionId = null;
    
    elements.chatMessages.innerHTML = '';
    elements.welcomeMessage.style.display = 'block';
    elements.chatTitle.textContent = 'IISc Research Assistant';
    elements.chatSubtitle.textContent = 'Ask me anything about IISc research, faculty, labs & more';
    elements.deleteBtn.style.display = 'none';
    elements.archiveBtn.style.display = 'none';
    elements.messageInput.value = '';
    elements.messageInput.focus();
    
    // Remove active state from threads
    document.querySelectorAll('.thread-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Close mobile sidebar
    if (window.innerWidth <= 768) {
        elements.sidebar.classList.remove('open');
    }
}

// Handle send message
async function handleSendMessage() {
    const query = elements.messageInput.value.trim();
    if (!query || isLoading) return;
    
    try {
        isLoading = true;
        elements.messageInput.value = '';
        elements.messageInput.disabled = true;
        elements.sendButton.disabled = true;
        
        // Hide welcome message
        if (elements.welcomeMessage) {
            elements.welcomeMessage.style.display = 'none';
        }
        
        // Add user message
        addMessageToUI(query, true);
        
        // Show thinking indicator
        elements.thinkingIndicator.style.display = 'block';
        scrollToBottom();
        
        let response;
        
        if (currentThreadId) {
            // Continue existing thread
            response = await fetch(`${API_URL}/api/threads/${currentThreadId}/messages`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
        } else {
            // Create new thread
            response = await fetch(`${API_URL}/api/threads?initial_query=${encodeURIComponent(query)}`, {
                method: 'POST'
            });
        }
        
        if (!response.ok) throw new Error('Failed to send message');
        
        const data = await response.json();
        
        // Update thread ID if new
        if (!currentThreadId) {
            currentThreadId = data.thread_id;
            elements.deleteBtn.style.display = 'block';
            elements.archiveBtn.style.display = 'block';
            // Reload thread list
            await loadThreads();
        }
        
        // Hide thinking indicator
        elements.thinkingIndicator.style.display = 'none';
        
        // Add assistant response
        addMessageToUI(data.answer, false, data.context_used);
        
        // Update context info
        if (data.context_used) {
            elements.contextInfo.textContent = `Answer based on ${data.context_used} sources`;
        }
        
        // Update thread in list
        await loadThreads();
        updateThreadListActive(currentThreadId);
        
    } catch (error) {
        console.error('Error sending message:', error);
        elements.thinkingIndicator.style.display = 'none';
        showError('Failed to send message. Please try again.');
    } finally {
        isLoading = false;
        elements.messageInput.disabled = false;
        elements.sendButton.disabled = false;
        elements.messageInput.focus();
    }
}

// Add message to UI
function addMessageToUI(content, isUser, contextUsed = 0) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isUser) {
        contentDiv.textContent = content;
    } else {
        // Format assistant message (convert markdown-like formatting)
        let formatted = escapeHtml(content);
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/\n\n/g, '</p><p>');
        formatted = formatted.replace(/\n/g, '<br>');
        contentDiv.innerHTML = `<p>${formatted}</p>`;
        
        // Add meta info
        if (contextUsed > 0) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            metaDiv.textContent = `📚 Based on ${contextUsed} sources`;
            contentDiv.appendChild(metaDiv);
        }
    }
    
    messageDiv.appendChild(contentDiv);
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message assistant';
    errorDiv.innerHTML = `
        <div class="message-content" style="background: #fee; border-color: #fcc;">
            <p style="color: #c33;">❌ ${escapeHtml(message)}</p>
        </div>
    `;
    elements.chatMessages.appendChild(errorDiv);
    scrollToBottom();
}

// Delete thread
async function handleDeleteThread() {
    if (!currentThreadId) return;
    
    try {
        const response = await fetch(`${API_URL}/api/threads/${currentThreadId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Failed to delete thread');
        
        hideDeleteModal();
        startNewChat();
        await loadThreads();
    } catch (error) {
        console.error('Error deleting thread:', error);
        showError('Failed to delete conversation');
    }
}

// Archive thread
async function handleArchiveThread() {
    if (!currentThreadId) return;
    
    try {
        const response = await fetch(`${API_URL}/api/threads/${currentThreadId}/archive`, {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error('Failed to archive thread');
        
        startNewChat();
        await loadThreads();
    } catch (error) {
        console.error('Error archiving thread:', error);
        showError('Failed to archive conversation');
    }
}

// Search threads
function handleSearchThreads(e) {
    const query = e.target.value.toLowerCase();
    
    if (!query) {
        renderThreadList(threads);
        return;
    }
    
    const filtered = threads.filter(thread => 
        thread.title.toLowerCase().includes(query)
    );
    
    renderThreadList(filtered);
}

// Update active thread in list
function updateThreadListActive(threadId) {
    document.querySelectorAll('.thread-item').forEach(item => {
        if (item.getAttribute('data-thread-id') === threadId) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
}

// Modal functions
function showDeleteModal() {
    elements.deleteModal.style.display = 'flex';
}

function hideDeleteModal() {
    elements.deleteModal.style.display = 'none';
}

// Toggle sidebar (mobile)
function toggleSidebar() {
    elements.sidebar.classList.toggle('open');
}

// Auto-resize textarea
function autoResizeTextarea() {
    elements.messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 150) + 'px';
    });
}

// Scroll to bottom of messages
function scrollToBottom() {
    setTimeout(() => {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }, 100);
}

// Format timestamp
function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Close mobile sidebar when clicking outside
document.addEventListener('click', (e) => {
    if (window.innerWidth <= 768) {
        if (!elements.sidebar.contains(e.target) && 
            !elements.mobileMenuBtn.contains(e.target) &&
            elements.sidebar.classList.contains('open')) {
            elements.sidebar.classList.remove('open');
        }
    }
});

// Handle window resize
window.addEventListener('resize', () => {
    if (window.innerWidth > 768) {
        elements.sidebar.classList.remove('open');
    }
});
