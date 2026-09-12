(() => {
  const base = location.port === "5500" ? `${location.protocol}//${location.hostname}:8000` : location.origin;
  const $ = id => document.getElementById(id);
  let editing = null, entryOffset = 0, alertOffset = 0, entryGeneration = 0, alertGeneration = 0;
  const size = 25;
  const node = (tag, value, parent, className = "") => {
    const element = document.createElement(tag);
    element.textContent = value;
    element.className = className;
    parent.append(element);
    return element;
  };
  async function api(path, options = {}) {
    const response = await fetch(base + path, {...options, signal:AbortSignal.timeout(8000)});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Check the plate number and required fields.");
    return data;
  }
  const date = value => value ? new Date(value).toLocaleString() : "No expiry";
  function pager(prefix, offset, total) {
    $(prefix + "Prev").disabled = offset === 0;
    $(prefix + "Next").disabled = offset + size >= total;
    $(prefix + "Count").textContent = total ? `${offset+1}-${Math.min(total,offset+size)} of ${total}` : "0 entries";
  }
  function empty(parent, message, columns) { node("td", message, node("tr","",parent)).colSpan = columns; }
  function edit(entry = null) {
    editing = entry;
    $("entryForm").reset();
    $("entryTitle").textContent = entry ? "Edit hotlist entry" : "Add hotlist plate";
    $("entryPlate").value = entry?.plate_text || "";
    $("entryPlate").readOnly = Boolean(entry);
    $("entryCategory").value = entry?.category || "stolen";
    $("entryReason").value = entry?.reason || "";
    $("entryReference").value = entry?.reference || "";
    $("entryActive").checked = entry?.active ?? true;
    if (entry?.expires_at) {
      const value = new Date(entry.expires_at);
      $("entryExpiry").value = new Date(value.getTime()-value.getTimezoneOffset()*60000).toISOString().slice(0,16);
    }
    $("formError").textContent = "";
    $("entryDialog").showModal();
  }
  async function entries() {
    const generation = ++entryGeneration;
    try {
      const data = await api(`/api/hotlist?limit=${size}&offset=${entryOffset}&q=${encodeURIComponent($("entrySearch").value)}`);
      if (generation !== entryGeneration) return;
      $("entryError").textContent = "";
      $("entryRows").replaceChildren();
      for (const entry of data.items) {
        const row = node("tr","",$("entryRows"));
        node("td",entry.plate_text,row,"plate");
        node("td",entry.category,row);
        const reason = node("td",entry.reason,row);
        if (entry.reference) node("div",entry.reference,reason,"muted");
        node("td",!entry.active ? "Inactive" : entry.expired ? "Expired" : "Active",row,`state ${!entry.active || entry.expired ? "inactive" : ""}`);
        node("td",date(entry.expires_at),row);
        const action = node("td","",row);
        const button = node("button","Edit",action,"edit-button");
        button.setAttribute("aria-label",`Edit ${entry.plate_text}`);
        button.onclick = () => edit(entry);
      }
      if (!data.items.length) empty($("entryRows"),"No matching hotlist entries.",6);
      pager("entries",entryOffset,data.total);
    } catch (error) { $("entryError").textContent = `${error.message} Try Search again.`; }
  }
  async function alerts() {
    const generation = ++alertGeneration;
    try {
      const data = await api(`/api/hotlist-alerts?limit=${size}&offset=${alertOffset}&unacknowledged=${$("unacknowledgedOnly").checked}`);
      if (generation !== alertGeneration) return;
      $("alertError").textContent = "";
      $("alertRows").replaceChildren();
      for (const alert of data.items) {
        const row = node("tr","",$("alertRows"));
        const plate = node("td",alert.plate_text,row,"plate");
        node("div",alert.category,plate,"muted");
        const camera = node("td",alert.camera_label,row);
        node("div",date(alert.seen_at),camera,"muted");
        const reason = node("td",alert.reason,row);
        if (alert.reference) node("div",alert.reference,reason,"muted");
        const label = alert.match_status === "matched" ? "Exact plate match" : alert.match_status === "review" ? "Needs verification" : "Retracted after correction";
        const status = node("td",label,row,`state ${alert.match_status}`);
        node("div",`${Math.round(alert.confidence*100)}% OCR confidence`,status,"muted");
        const action = node("td","",row);
        if (alert.acknowledged_at) node("span",`Acknowledged ${date(alert.acknowledged_at)}`,action,"muted");
        else if (alert.match_status !== "retracted") {
          const button = node("button","Acknowledge",action);
          button.onclick = async () => {
            button.disabled = true;
            try { await HotlistAlerts.acknowledge(alert); alerts(); }
            catch (error) { button.disabled = false; $("alertError").textContent = error.message; }
          };
        }
      }
      if (!data.items.length) empty($("alertRows"),"No matching alerts.",5);
      pager("alerts",alertOffset,data.total);
    } catch (error) { $("alertError").textContent = `${error.message} Try Refresh again.`; }
  }
  $("entryForm").onsubmit = async event => {
    event.preventDefault();
    $("saveEntry").disabled = true;
    try {
      await api(editing ? `/api/hotlist/${editing.id}` : "/api/hotlist", {
        method:editing ? "PUT" : "POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({plate_text:$("entryPlate").value,category:$("entryCategory").value,
          reason:$("entryReason").value,reference:$("entryReference").value,
          active:$("entryActive").checked,expires_at:$("entryExpiry").value ? new Date($("entryExpiry").value).toISOString() : null})
      });
      $("entryDialog").close();
      entryOffset = 0;
      entries();
    } catch (error) { $("formError").textContent = error.message; }
    finally { $("saveEntry").disabled = false; }
  };
  $("addEntry").onclick = () => edit();
  $("cancelEntry").onclick = () => $("entryDialog").close();
  $("searchEntries").onclick = () => { entryOffset=0; entries(); };
  $("entrySearch").onkeydown = event => { if (event.key === "Enter") { entryOffset=0; entries(); } };
  $("refreshAlerts").onclick = alerts;
  $("unacknowledgedOnly").onchange = () => { alertOffset=0; alerts(); };
  $("entriesPrev").onclick = () => { entryOffset=Math.max(0,entryOffset-size); entries(); };
  $("entriesNext").onclick = () => { entryOffset+=size; entries(); };
  $("alertsPrev").onclick = () => { alertOffset=Math.max(0,alertOffset-size); alerts(); };
  $("alertsNext").onclick = () => { alertOffset+=size; alerts(); };
  let refreshTimer;
  window.addEventListener("hotlist-update", () => { clearTimeout(refreshTimer); refreshTimer=setTimeout(alerts,250); });
  entries();
  alerts();
})();
