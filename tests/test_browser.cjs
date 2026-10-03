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
      const login = await context.request.post(base + "/api/auth/login", {
        data: {username: "admin_demo", password: "AdminDemo!2026"}
      });
      assert.equal(login.ok(), true, "browser test login must establish the session cookie");
      const page = await context.newPage();
      const errors = [];
      page.on("pageerror", e => errors.push(e.message));
      const cameras = [
        {camera_id:"CAM01",label:"North gate",lat:23.0395,lng:72.566,location_known:true},
        {camera_id:"CAM02",label:"South gate",lat:23.0325,lng:72.5145,location_known:true},
        {camera_id:"CAM03",label:"CG Road",lat:23.0258,lng:72.5644,location_known:true}
      ];
      const hops = [
        {...cameras[0],camera_label:cameras[0].label,event_id:1,timestamp:"2026-09-10T04:00:00Z",confidence:.94,status:"ok",plausible:true},
        {...cameras[0],camera_label:cameras[0].label,event_id:2,timestamp:"2026-09-10T04:01:00Z",confidence:.93,status:"ok",plausible:true},
        {...cameras[1],camera_label:cameras[1].label,event_id:3,timestamp:"2026-09-10T04:06:18Z",confidence:.94,status:"ok",plausible:true},
        {...cameras[1],camera_label:cameras[1].label,event_id:4,timestamp:"2026-09-10T04:07:00Z",confidence:.92,status:"ok",plausible:true},
        {...cameras[2],camera_label:cameras[2].label,event_id:5,timestamp:"2026-09-10T04:12:00Z",confidence:.94,status:"ok",plausible:true}
      ];
      const trajectoryHops = [
        {
          source_event_id:1,start_event_id:1,destination_event_id:3,end_event_id:3,
          source_camera_id:"CAM01",destination_camera_id:"CAM02",
          distance_meters:4200,travel_time_seconds:378,estimated_speed_kmh:40,
          routing_source:"osrm",distance_source:"osrm",
          route_geometry:[
            {lat:23.0395,lng:72.566},
            {lat:23.0365,lng:72.545},
            {lat:23.0325,lng:72.5145}
          ]
        },
        {
          source_event_id:3,start_event_id:3,destination_event_id:5,end_event_id:5,
          source_camera_id:"CAM02",destination_camera_id:"CAM03",
          distance_meters:3800,travel_time_seconds:342,estimated_speed_kmh:40,
          routing_source:"manual",distance_source:"manual",
          road_connection:{connection_id:7,routing_source:"manual",distance_meters:3800}
        }
      ];
      const json = data => ({contentType:"application/json",body:JSON.stringify(data)});
      const mapConfig = await context.request.get(base + "/api/map/config");
      assert.equal(mapConfig.ok(), true, "map config endpoint should be reachable");
      const mapConfigJson = await mapConfig.json();
      assert.equal(mapConfigJson.provider, "esri_world_street");
      assert.match(mapConfigJson.tile_url, /server\.arcgisonline\.com/);
      assert.doesNotMatch(mapConfigJson.tile_url, /basemaps\.cartocdn\.com/);
      await page.route("**/api/auth/me", route=>route.fulfill(json({
        user:{id:1,username:"admin_demo",role:"SYSTEM_ADMIN",active:true},
        permissions:["*"]
      })));
      await page.route("**/api/cameras", route=>route.fulfill(json(cameras)));
      await page.route("**/api/cameras/*/snapshot?*", route=>route.fulfill({contentType:"image/png",body:pixel}));
      await page.route("**/api/stats", route=>route.fulfill(json({total_scans:2,unique_vehicles:1,needs_review:0})));
      await page.route("**/api/vehicles", route=>route.fulfill(json([{plate_text:"GJ01AB1234",sightings:5,last_camera:"CAM03"}])));
      await page.route("**/api/events", route=>route.fulfill(json([])));
      const trajectories=[];
      await page.route("**/api/trajectory/**", route=>{
        trajectories.push(new URL(route.request().url()));
        return route.fulfill(json({plate_text:"GJ01AB1234",hops,trajectory_hops:trajectoryHops}));
      });
      await page.route("**/api/search/plates?*", route=>route.fulfill(json({matches:[
        {event_id:1,plate_text:"GJ01AB1234",camera_id:"CAM01",camera_label:"North gate",
          timestamp:"2026-09-10T04:00:00Z",match_score:1,distance_m:0}
      ]})));
      await page.goto(base+"/dashboard.html");
      await page.waitForSelector(".leaflet-container");
      await page.waitForFunction(()=>document.querySelector("#mapStatus")?.textContent.includes("esri_world_street"), null, {timeout:8000});
      const tileSrcs = await page.evaluate(() => Array.from(document.querySelectorAll(".leaflet-tile")).map(img => img.currentSrc || img.src));
      assert(tileSrcs.some(src => src.includes("server.arcgisonline.com")), "trajectory map should request the configured Esri tile provider");
      assert(tileSrcs.every(src => !src.includes("basemaps.cartocdn.com")), "dashboard must not request CARTO API-key tiles by default");
      assert.match(await page.locator("body").innerText(), /ANPR-derived traffic estimates/);
      assert.match(await page.locator("body").innerText(), /500\+ camera simulation/i);
      assert.match(await page.locator("body").innerText(), /NO REAL TRAFFIC SIGNAL CONTROL/);
      assert.match(await page.locator("body").innerText(), /DEMO enforcement workflow/);
      const markerIconUrl = await page.evaluate(() => L.Icon.Default.prototype.options.iconUrl);
      assert.match(markerIconUrl, /leaflet@1\.9\.4\/dist\/images\/marker-icon\.png/);
      await page.waitForFunction(()=>document.querySelectorAll(".leaflet-overlay-pane path").length > 0);
      await page.fill("#searchPlate","GJ01AB1234");
      await page.fill("#searchStart","2026-09-10T08:00");
      await page.fill("#searchEnd","2026-09-10T10:00");
      await page.click("#searchBtn");
      await page.waitForFunction(()=>document.querySelectorAll(".search-hit").length===1);
      await page.waitForTimeout(300);
      const mapRender = await page.evaluate(() => window.__ANPR_LAST_MAP_RENDER);
      assert.equal(mapRender.rawObservationCount, 5, "history keeps every persisted observation");
      assert.equal(mapRender.trajectoryStopCount, 3, "map summarizes repeated observations into camera stops");
      assert.equal(mapRender.displayedCameraMarkerCount, 3, "map should not draw all simulation cameras during a selected trajectory");
      assert.equal(mapRender.osrmRouteSegments, 1, "OSRM geometry must be rendered when supplied");
      assert.equal(mapRender.configuredRouteSegments, 1, "manual road graph fallback should still render when route geometry is unavailable");
      const stableLayerCount = mapRender.routeLayerCount;
      await page.click("#searchBtn");
      await page.waitForTimeout(300);
      assert.equal(await page.evaluate(() => window.__ANPR_LAST_MAP_RENDER.routeLayerCount), stableLayerCount,
        "repeated trajectory redraw must clear and reuse map layers");
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
