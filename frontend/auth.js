/* Cookie-first authentication helper for the ANPR demo frontend. */
(() => {
  const base = location.port === "5500"
    ? `${location.protocol}//${location.hostname}:8000`
    : location.origin;
  let currentUser = null;
  const listeners = new Set();

  function notify() {
    for (const listener of listeners) listener(currentUser);
    window.dispatchEvent(new CustomEvent("anpr-auth-change", { detail: { user: currentUser } }));
  }

  function setUser(user) {
    currentUser = user || null;
    notify();
    return currentUser;
  }

  function clearLegacyTokens() {
    sessionStorage.removeItem("anpr_token");
    localStorage.removeItem("anpr_token");
  }

  function url(path) {
    return path.startsWith("http") ? path : `${base}${path}`;
  }

  async function request(path, options = {}, timeoutMs = 8000) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(new DOMException("Request timed out", "TimeoutError")), timeoutMs);
    try {
      const response = await fetch(url(path), {
        ...options,
        credentials: "include",
        headers: options.headers || {},
        signal: controller.signal,
      });
      if (response.status === 401) {
        setUser(null);
        window.dispatchEvent(new CustomEvent("anpr-auth-required"));
      }
      return response;
    } finally {
      clearTimeout(timeout);
    }
  }

  async function loadSession() {
    clearLegacyTokens();
    const response = await request("/api/auth/me", {}, 6000);
    if (!response.ok) {
      setUser(null);
      return null;
    }
    const data = await response.json();
    return setUser(data.user);
  }

  async function login(username, password) {
    clearLegacyTokens();
    const response = await request("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    }, 8000);
    if (!response.ok) throw new Error("Login failed");
    const data = await response.json();
    return setUser(data.user);
  }

  async function logout() {
    try {
      await request("/api/auth/logout", { method: "POST" }, 6000);
    } finally {
      clearLegacyTokens();
      setUser(null);
    }
  }

  function onChange(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  window.ANPRAuth = {
    base,
    request,
    loadSession,
    login,
    logout,
    onChange,
    get currentUser() { return currentUser; },
    isAuthenticated() { return Boolean(currentUser); },
  };
})();
