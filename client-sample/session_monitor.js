/**
 * Session monitoring utility for WebSocket connections
 * Ensures sessions stay active with server-side tracking
 */
class SessionMonitor {
  constructor(socket, options = {}) {
    this.socket = socket;
    this.thread_id = options.thread_id || null;
    this.pingInterval = options.pingInterval || 30000; // Default: ping every 30 seconds
    this.reconnectAttempts = options.reconnectAttempts || 5;
    this.pingTimer = null;
    this.monitorActive = false;
    this.debugEnabled = options.debug || false;
    
    // Bind event handlers
    this.onConnect = this.onConnect.bind(this);
    this.onDisconnect = this.onDisconnect.bind(this);
    this.onSessionRegistered = this.onSessionRegistered.bind(this);
    this.onSystemPing = this.onSystemPing.bind(this);
    this.ping = this.ping.bind(this);
    
    // Setup event listeners
    this.socket.on('connect', this.onConnect);
    this.socket.on('disconnect', this.onDisconnect);
    this.socket.on('session_registered', this.onSessionRegistered);
    this.socket.on('system_ping', this.onSystemPing);
    
    this.log('SessionMonitor initialized');
  }
  
  /**
   * Start monitoring the session
   * @param {string} thread_id - Optional thread ID to register with
   */
  start(thread_id = null) {
    if (thread_id) {
      this.thread_id = thread_id;
    }
    
    if (!this.thread_id) {
      console.error('Cannot start session monitor without a thread_id');
      return;
    }
    
    this.monitorActive = true;
    
    // If already connected, register session
    if (this.socket.connected) {
      this.registerSession();
    }
    
    this.log('Session monitoring started for thread:', this.thread_id);
    return this;
  }
  
  /**
   * Stop monitoring the session
   */
  stop() {
    this.monitorActive = false;
    this.clearPingTimer();
    this.log('Session monitoring stopped');
    return this;
  }
  
  /**
   * Handle socket connect event
   */
  onConnect() {
    this.log('Socket connected');
    
    if (this.monitorActive && this.thread_id) {
      this.registerSession();
    }
  }
  
  /**
   * Handle socket disconnect event
   */
  onDisconnect(reason) {
    this.log('Socket disconnected:', reason);
    this.clearPingTimer();
  }
  
  /**
   * Handle session registration confirmation
   */
  onSessionRegistered(data) {
    this.log('Session registered:', data);
    
    // Start sending periodic pings to keep the session active
    this.startPinging();
  }
  
  /**
   * Handle system-initiated ping to keep session alive
   */
  onSystemPing(data) {
    this.log('Received system ping:', data);
    
    // Immediately respond to keep the session alive
    this.socket.emit('system_ping_response', { 
      thread_id: this.thread_id,
      timestamp: new Date().toISOString(),
      received_at: data.timestamp
    });
    
    // Also send a regular ping to reset all timers
    this.ping();
  }
  
  /**
   * Register the session with the server
   */
  registerSession() {
    this.log('Registering session with thread_id:', this.thread_id);
    this.socket.emit('register_session', {
      thread_id: this.thread_id
    });
  }
  
  /**
   * Start the ping timer
   */
  startPinging() {
    this.clearPingTimer();
    this.pingTimer = setInterval(this.ping, this.pingInterval);
    this.log('Started ping timer with interval:', this.pingInterval);
  }
  
  /**
   * Send a ping to the server
   */
  ping() {
    if (!this.socket.connected) {
      this.log('Skipping ping - socket not connected');
      return;
    }
    
    this.log('Sending ping');
    this.socket.emit('ping', { 
      thread_id: this.thread_id,
      timestamp: new Date().toISOString()
    }, (response) => {
      if (response && response.pong) {
        this.log('Received pong:', response);
      }
    });
  }
  
  /**
   * Clear the ping timer
   */
  clearPingTimer() {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }
  
  /**
   * Log a message if debugging is enabled
   */
  log(...args) {
    if (this.debugEnabled) {
      console.log('[SessionMonitor]', ...args);
    }
  }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { SessionMonitor };
} 