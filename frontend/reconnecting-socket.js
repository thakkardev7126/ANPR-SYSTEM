/* Shared reconnect lifecycle for scanner and dashboard sockets. */
class ANPRSocket extends EventTarget {
  constructor(url, { onMessage = () => {}, onState = () => {}, WebSocketClass = WebSocket } = {}) {
    super();
    this.url = url;
    this.onMessage = onMessage;
    this.onState = onState;
    this.WebSocketClass = WebSocketClass;
    this.attempt = 0;
    this.stopped = false;
    this.socket = null;
    this.retry = null;
    this.online = () => { if (this.readyState !== 1) this.reconnect(); };
    this.offline = () => { this.socket?.close(); this.onState("offline"); };
    globalThis.addEventListener?.("online", this.online);
    globalThis.addEventListener?.("offline", this.offline);
    this.connect();
  }
  get readyState() { return this.socket?.readyState ?? 3; }
  connect() {
    if (this.stopped || this.readyState <= 1) return;
    clearTimeout(this.retry);
    if (globalThis.navigator?.onLine === false) { this.onState("offline"); return; }
    this.onState(this.attempt ? "reconnecting" : "connecting");
    const socket = this.socket = new this.WebSocketClass(this.url);
    let lastMessage = Date.now();
    const timeout = setTimeout(() => socket.close(), 10000);
    const heartbeat = setInterval(() => {
      if (socket.readyState !== 1) return;
      if (Date.now() - lastMessage > 45000) socket.close();
      else socket.send(JSON.stringify({ action: "ping" }));
    }, 15000);
    socket.onopen = () => {
      clearTimeout(timeout);
      if (this.stopped || this.socket !== socket) { socket.close(); return; }
      this.attempt = 0;
      this.onState("connected");
      this.dispatchEvent(new Event("open"));
    };
    socket.onmessage = event => {
      if (this.socket !== socket || this.stopped) return;
      lastMessage = Date.now();
      let data;
      try { data = JSON.parse(event.data); } catch { return; }
      if (data.type !== "pong") this.onMessage(data);
    };
    socket.onerror = () => socket.close();
    socket.onclose = () => {
      clearTimeout(timeout);
      clearInterval(heartbeat);
      if (this.socket !== socket || this.stopped) return;
      this.onState("reconnecting");
      const delay = Math.min(15000, 1000 * 2 ** this.attempt++) * (.8 + Math.random() * .4);
      this.retry = setTimeout(() => this.connect(), delay);
    };
  }
  send(payload) {
    if (this.readyState !== 1) return false;
    this.socket.send(typeof payload === "string" ? payload : JSON.stringify(payload));
    return true;
  }
  reconnect() {
    clearTimeout(this.retry);
    const old = this.socket;
    this.socket = null;
    old?.close();
    this.connect();
  }
  close() {
    this.stopped = true;
    clearTimeout(this.retry);
    this.socket?.close();
    globalThis.removeEventListener?.("online", this.online);
    globalThis.removeEventListener?.("offline", this.offline);
  }
}
globalThis.ANPRSocket = ANPRSocket;
if (typeof module !== "undefined") module.exports = ANPRSocket;

