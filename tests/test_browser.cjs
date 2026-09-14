const {chromium} = require("playwright");
const assert = require("node:assert/strict");
const path = require("node:path");
const base = process.env.ANPR_TEST_URL || "http://127.0.0.1:8000";
const out = process.env.ANPR_SCREENSHOT_DIR || process.cwd();
const pixel = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l1sAAAAASUVORK5CYII=", "base64");

(async () => {
  const browser = await chromium.launch({channel:"msedge",headless:true});
  try {
    for (const viewport of [{width:1366,height:900},{width:390,height:844}]) {
      const context = await browser.newContext({viewport,ignoreHTTPSErrors:true});
      const page = await context.newPage();
      const errors = [];
      page.on("pageerror", e => errors.push(e.message));
      const cameras = [
        {camera_id:"CAM01",label:"North gate",lat:23.0395,lng:72.566,location_known:true},
        {camera_id:"CAM02",label:"South gate",lat:23.0325,lng:72.5145,location_known:true}
      ];
      const hops = cameras.map((c,i) => ({...c,camera_label:c.label,event_id:i+1,
        timestamp:"2026-09-10T04:00:00Z",confidence:.94,status:"ok",plausible:true}));
      const json = data => ({contentType:"application/json",body:JSON.stringify(data)});
      await page.route("**/api/auth/me", route=>route.fulfill(json({
        user:{id:1,username:"admin_demo",role:"SYSTEM_ADMIN",active:true},
        permissions:["*"]
      })));
      await page.route("**/api/cameras", route=>route.fulfill(json(cameras)));
      await page.route("**/api/cameras/*/snapshot?*", route=>route.fulfill({contentType:"image/png",body:pixel}));
      await page.route("**/api/stats", route=>route.fulfill(json({total_scans:2,unique_vehicles:1,needs_review:0})));
      await page.route("**/api/vehicles", route=>route.fulfill(json([{plate_text:"GJ01AB1234",sightings:2,last_camera:"CAM02"}])));
      await page.route("**/api/events", route=>route.fulfill(json([])));
      const trajectories=[];
      await page.route("**/api/trajectory/**", route=>{
        trajectories.push(new URL(route.request().url()));
        return route.fulfill(json({plate_text:"GJ01AB1234",hops}));
      });
      await page.route("**/api/search/plates?*", route=>route.fulfill(json({matches:[
        {event_id:1,plate_text:"GJ01AB1234",camera_id:"CAM01",camera_label:"North gate",
          timestamp:"2026-09-10T04:00:00Z",match_score:1,distance_m:0}
      ]})));
      await page.goto(base+"/dashboard.html");
      await page.waitForSelector(".leaflet-container");
      await page.waitForFunction(()=>document.querySelectorAll(".leaflet-overlay-pane path").length > 0);
      await page.fill("#searchPlate","GJ01AB1234");
      await page.fill("#searchStart","2026-09-10T08:00");
      await page.fill("#searchEnd","2026-09-10T10:00");
      await page.click("#searchBtn");
      await page.waitForFunction(()=>document.querySelectorAll(".search-hit").length===1);
      await page.waitForTimeout(300);
      const filtered=trajectories.find(u=>u.searchParams.has("start"));
      assert(filtered, "trajectory must carry date filter");
      assert.equal(filtered.searchParams.get("radius_m"),"5000");
      await page.screenshot({path:path.join(out,`dashboard-${viewport.width}.png`),fullPage:true});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,"dashboard overflow");
      await page.click("#resetSearch");
      await page.waitForTimeout(200);
      assert.equal(trajectories.at(-1).searchParams.has("start"),false);
      await page.goto(base+"/scan.html");
      await page.waitForFunction(()=>cameraSocket?.readyState===1);
      await page.evaluate(()=>cameraSocket.socket.close());
      await page.waitForFunction(()=>cameraSocket?.readyState===1,{},{timeout:8000});
      const detections = ["GJ01AB1234","GJ01AB5678"].map((plate,i)=>({
        event_id:i+1,plate_text:plate,confidence:.94,status:"ok",needs_review:false,duplicate:false
      }));
      await page.route("**/api/scan?*",route=>route.fulfill(json({
        ...detections[0],detections:new URL(route.request().url()).searchParams.get("selection")==="largest"
          ? [detections[0]] : detections
      })));
      await page.setInputFiles("#fileInput",{name:"plate.png",mimeType:"image/png",buffer:pixel});
      await page.click("#submitBtn");
      await page.waitForFunction(()=>document.querySelectorAll(".log-item").length===2);
      assert.equal(await page.locator("#resultPlate").innerText(),"GJ01AB1234");
      await page.route("**/api/process-frame?*",route=>route.fulfill(json({
        event_id:null,
        plate_text:"D",
        raw_text:"D",
        confidence:.79,
        status:"scanning",
        needs_review:false,
        detections:[]
      })));
      const autoData = await page.evaluate(async () => {
        const file = new File([new Blob(["x"], {type:"image/jpeg"})], "auto.jpg", {type:"image/jpeg"});
        return await uploadFiles([file], "auto frames", "auto");
      });
      assert.equal(autoData.results.length,0);
      assert.equal(await page.locator(".log-item").count(),2,"rejected auto OCR must not be logged");
      await page.screenshot({path:path.join(out,`scanner-${viewport.width}.png`),fullPage:true});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,"scanner overflow");
      assert.deepEqual(errors,[],"page errors");
      console.log(JSON.stringify({viewport,filteredRoute:true,multiPlateUI:true,reconnect:true,pageErrors:errors}));
      await context.close();
    }
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
