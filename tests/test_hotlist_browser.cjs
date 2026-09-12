const {chromium} = require("playwright");
const assert = require("node:assert/strict");
const {spawn} = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const net = require("node:net");
const {once} = require("node:events");
const root = path.resolve(__dirname,"..");
const work = fs.mkdtempSync(path.join(os.tmpdir(),"anpr-hotlist-browser-"));
const out = process.env.ANPR_SCREENSHOT_DIR || work;
const pixel = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l1sAAAAASUVORK5CYII=","base64");

(async () => {
  const portProbe = net.createServer();
  portProbe.listen(0,"127.0.0.1");
  await once(portProbe,"listening");
  const port = portProbe.address().port;
  await new Promise(resolve=>portProbe.close(resolve));
  const base = `http://127.0.0.1:${port}`;
  const env = {...process.env,ANPR_DATABASE_PATH:path.join(work,"test.db"),
    ANPR_UPLOAD_DIR:path.join(work,"uploads"),ANPR_REVIEW_DIR:path.join(work,"review")};
  delete env.ANPR_DATABASE_URL;
  const server = spawn(process.env.ANPR_TEST_PYTHON || "py",["-3.13","-B","-m","uvicorn","hotlist_server:app","--app-dir","tests","--host","127.0.0.1","--port",String(port)],
    {cwd:root,env,windowsHide:true,stdio:["ignore","pipe","pipe"]});
  let serverLog="";
  server.stdout.on("data",chunk=>serverLog+=chunk);
  server.stderr.on("data",chunk=>serverLog+=chunk);
  let browser;
  try {
    let ready=false;
    for(let i=0;i<60;i++) {
      try { const res=await fetch(base+"/api/hotlist"); if(res.ok) {ready=true;break;} } catch {}
      await new Promise(resolve=>setTimeout(resolve,250));
    }
    assert(ready,serverLog);
    browser = await chromium.launch({channel:"msedge",headless:true});
    for(const [i,viewport] of [{width:1366,height:900},{width:390,height:844}].entries()) {
      const camera = i ? "CAM02" : "CAM01";
      const plate = i ? "GJ01AB5678" : "GJ01AB1234";
      const context=await browser.newContext({viewport,ignoreHTTPSErrors:true});
      const page=await context.newPage();
      const errors=[];
      page.on("pageerror",error=>errors.push(error.message));
      await page.goto(base+"/hotlist.html");
      await page.click("#addEntry");
      await page.fill("#entryPlate",plate);
      await page.fill("#entryReason","Test listing: verify against case record");
      await page.fill("#entryReference","TEST-ONLY");
      await page.click("#saveEntry");
      await page.waitForFunction(()=>!document.querySelector("#entryDialog").open);
      await page.getByRole("button",{name:`Edit ${plate}`,exact:true}).click();
      await page.uncheck("#entryActive");
      await page.click("#saveEntry");
      await page.waitForFunction(()=>!document.querySelector("#entryDialog").open);
      const ignored=await context.request.post(base+"/api/scan",{multipart:{camera_id:camera,file:{name:"test.png",mimeType:"image/png",buffer:pixel}}});
      assert.equal((await ignored.json()).hotlist_alerts.length,0);
      await page.getByRole("button",{name:`Edit ${plate}`,exact:true}).click();
      await page.check("#entryActive");
      await page.click("#saveEntry");
      await page.waitForFunction(()=>!document.querySelector("#entryDialog").open);
      const observer=await context.newPage();
      await observer.goto(base+"/dashboard.html");
      await page.goto(base+"/scan.html");
      await page.selectOption("#cameraSelect",camera);
      await page.setInputFiles("#fileInput",{name:"plate.png",mimeType:"image/png",buffer:pixel});
      await page.click("#submitBtn");
      await page.waitForSelector(".hotlist-popup");
      await observer.waitForSelector(".hotlist-popup",{timeout:15000});
      assert.equal(await page.locator(".hotlist-popup strong").innerText(),plate);
      await page.screenshot({path:path.join(out,`hotlist-popup-${viewport.width}.png`),fullPage:true});
      await page.reload();
      await page.waitForSelector(".hotlist-popup");
      const repeated=await context.request.post(base+"/api/process-frame",{multipart:{camera_id:camera,frame:{name:"test.png",mimeType:"image/png",buffer:pixel}}});
      assert.equal((await repeated.json()).hotlist_alerts.length,1);
      assert.equal(await page.locator(".hotlist-popup").count(),1);
      await page.locator(".hotlist-popup button").click();
      await page.waitForSelector(".hotlist-popup",{state:"detached"});
      await observer.waitForSelector(".hotlist-popup",{state:"detached",timeout:15000});
      await page.goto(base+"/hotlist.html");
      await page.waitForFunction(()=>document.querySelector("#alertRows").textContent.includes("Acknowledged"));
      await page.screenshot({path:path.join(out,`hotlist-${viewport.width}.png`),fullPage:true});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
      await page.evaluate(()=>HotlistAlerts.sync());
      let heldRoute, release;
      const held = new Promise(resolve=>release=resolve);
      const inboxPattern="**/api/hotlist-alerts?unacknowledged=true**";
      await page.route(inboxPattern,route=>{heldRoute=route;release();});
      const syncCall=page.evaluate(()=>HotlistAlerts.sync());
      await held;
      const arriving={id:99999,revision:1,plate_text:plate,camera_id:camera,camera_label:"Test camera",
        category:"watchlist",reason:"<img src=x onerror=alert(1)>",reference:"RACE-TEST",
        seen_at:new Date().toISOString(),match_status:"review",acknowledged_at:null};
      await page.evaluate(alert=>HotlistAlerts.receive(alert),arriving);
      await heldRoute.fulfill({contentType:"application/json",body:JSON.stringify({items:[],total:0})});
      await syncCall;
      assert.equal(await page.locator('.hotlist-popup[data-alert-id="99999"]').count(),1,"in-flight inbox snapshot must not erase a newer socket alert");
      assert.equal(await page.locator(".hotlist-popup img").count(),0,"listing text must never become HTML");
      await page.evaluate(alert=>HotlistAlerts.receive({...alert,revision:2,acknowledged_at:new Date().toISOString()}),arriving);
      await page.evaluate(alert=>HotlistAlerts.receive(alert),arriving);
      assert.equal(await page.locator(".hotlist-popup").count(),0,"stale message must not undo acknowledgement");
      await page.unroute(inboxPattern);
      assert.deepEqual(errors,[]);
      console.log(JSON.stringify({viewport,crud:true,disabledNoMatch:true,scanPopup:true,dashboardBroadcast:true,durableReload:true,acknowledge:true,reconciliationRace:true,safeText:true,pageErrors:errors}));
      await context.close();
    }
  } finally {
    await browser?.close();
    // Terminate only this fixture's process tree; py.exe launches Python on Windows.
    if(process.platform==="win32") {
      const killer=spawn("taskkill",["/PID",String(server.pid),"/T","/F"],{windowsHide:true,stdio:"ignore"});
      await once(killer,"exit");
    } else server.kill();
    if(server.exitCode===null) await once(server,"exit");
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
