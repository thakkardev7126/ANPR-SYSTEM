/* Durable hotlist inbox shared by every operational page. */
(() => {
  const base = window.ANPRAuth?.base || (location.port === "5500" ? `${location.protocol}//${location.hostname}:8000` : location.origin);
  const host = document.createElement("aside");
  host.id = "hotlistPopups";
  host.setAttribute("aria-label", "Hotlist alerts");
  document.body.append(host);
  const pending = new Map();
  const revisions = new Map();
  let socket, timer, syncing = false, stopped = true;
  const camera = () => document.body.dataset.hotlistScope === "camera"
    ? document.getElementById("cameraSelect")?.value : null;
  const text = (tag, value, parent) => {
    const node = document.createElement(tag);
    node.textContent = value;
    parent.append(node);
    return node;
  };
  async function request(path, options = {}) {
    const res = window.ANPRAuth
      ? await window.ANPRAuth.request(path, options, 6000)
      : await fetch(base + path, {...options, credentials:"include", signal:AbortSignal.timeout(6000)});
    if (res.status === 401) stop();
    if (!res.ok) throw new Error(`Hotlist request failed (${res.status})`);
    return res.json();
  }
  function receive(alert, renderNow = true, fromSnapshot = false) {
    if (!alert || !Number.isInteger(alert.id)) return;
    const knownRevision = revisions.get(alert.id) || 0;
    if (knownRevision > alert.revision || (knownRevision === alert.revision && (!fromSnapshot || pending.has(alert.id)))) return;
    revisions.set(alert.id, alert.revision);
    if (alert.acknowledged_at || alert.match_status === "retracted") pending.delete(alert.id);
    else pending.set(alert.id, alert);
    while (pending.size > 200) pending.delete(Math.min(...pending.keys()));
    while (revisions.size > 1000) revisions.delete(revisions.keys().next().value);
    if (renderNow) render();
    window.dispatchEvent(new CustomEvent("hotlist-update", {detail:alert}));
  }
  async function acknowledge(alert) {
    const updated = await request(`/api/hotlist-alerts/${alert.id}/acknowledge`, {method:"POST"});
    receive(updated);
    return updated;
  }
  function render() {
    const scope = camera();
    const items = [...pending.values()].filter(a => document.body.dataset.hotlistScope === "camera"
      ? scope && a.camera_id === scope : true).sort((a,b) => b.id-a.id).slice(0,3);
    const fingerprint = items.map(a=>`${a.id}:${a.revision}`).join(",");
    if (host.dataset.fingerprint === fingerprint) return;
    host.dataset.fingerprint = fingerprint;
    host.replaceChildren();
    for (const alert of items) {
      const panel = document.createElement("section");
      panel.className = "hotlist-popup";
      panel.dataset.status = alert.match_status;
      panel.dataset.alertId = alert.id;
      panel.setAttribute("role", "alert");
      text("div", alert.match_status === "review" ? "Possible hotlist match - needs verification" : "Hotlist plate match - verify vehicle", panel);
      text("strong", alert.plate_text, panel);
      text("p", `${alert.category} listing | ${alert.camera_label} | ${new Date(alert.seen_at).toLocaleString()}`, panel);
      text("p", alert.reason, panel);
      if (alert.reference) text("p", `Reference: ${alert.reference}`, panel);
      const actions = document.createElement("div");
      actions.className = "hotlist-actions";
      const button = text("button", "Acknowledge", actions);
      button.type = "button";
      const link = text("a", "Alert history", actions);
      link.href = "hotlist.html#alerts";
      button.onclick = async () => {
        button.disabled = true;
        try { await acknowledge(alert); }
        catch (error) { button.disabled = false; button.textContent = "Retry acknowledgement"; button.title = error.message; }
      };
      panel.append(actions);
      host.append(panel);
    }
  }
  async function sync() {
    if (syncing || stopped) return;
    syncing = true;
    try {
      const scope = camera();
      const suffix = scope ? `&camera_id=${encodeURIComponent(scope)}` : "";
      const requested = new Map([...pending].map(([id, alert]) => [id, alert.revision]));
      const data = await request(`/api/hotlist-alerts?unacknowledged=true&limit=200${suffix}`);
      // Reconcile acknowledgements made on other devices, even without a socket.
      const ids = new Set(data.items.map(a => a.id));
      for (const [id, alert] of pending) {
        if ((!scope || alert.camera_id === scope) && requested.get(id) === alert.revision && !ids.has(id)) pending.delete(id);
      }
      data.items.forEach(a => receive(a, false, true));
      render();
    } catch { /* The durable inbox remains available on the next retry. */ }
    finally { syncing = false; }
  }
  function start() {
    if (socket || !window.ANPRAuth?.isAuthenticated()) return;
    stopped = false;
    socket = new ANPRSocket(base.replace(/^http/,"ws") + `/ws/events`, {
      onMessage: message => { if (message.type === "hotlist_alert") receive(message.alert); },
      onState: state => {
        if (state === "connected") sync();
        if (state === "unauthorized" || state === "disconnected") socket = null;
      },
      shouldReconnect: () => window.ANPRAuth?.isAuthenticated() === true,
    });
    timer = setInterval(sync, 10000);
    sync();
  }
  function stop() {
    stopped = true;
    clearInterval(timer);
    timer = null;
    socket?.close();
    socket = null;
    pending.clear();
    render();
  }
  window.HotlistAlerts = {receive, acknowledge, sync};
  document.getElementById("cameraSelect")?.addEventListener("change", () => { render(); sync(); });
  window.addEventListener("pagehide", stop);
  window.addEventListener("pageshow", event => { if (event.persisted) start(); });
  window.addEventListener("anpr-auth-change", event => { event.detail.user ? start() : stop(); });
  if (window.ANPRAuth?.isAuthenticated()) start();
})();
