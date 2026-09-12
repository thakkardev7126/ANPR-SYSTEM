const test = require("node:test");
const assert = require("node:assert/strict");
const Socket = require("../frontend/reconnecting-socket.js");

class FakeWebSocket {
  static instances = [];
  constructor(url) { this.url=url; this.readyState=0; this.sent=[]; FakeWebSocket.instances.push(this); }
  open() { this.readyState=1; this.onopen?.(); }
  close() { this.readyState=3; this.onclose?.(); }
  send(data) { this.sent.push(data); }
  receive(data) { this.onmessage?.({data:JSON.stringify(data)}); }
}

test("reconnects, preserves URL and stops after explicit close", t => {
  t.mock.timers.enable({apis:["setTimeout","setInterval"]});
  FakeWebSocket.instances=[];
  const states=[], messages=[];
  const socket=new Socket("ws://test/ws/cameras/CAM02",{
    WebSocketClass:FakeWebSocket,onState:s=>states.push(s),onMessage:m=>messages.push(m)});
  const first=FakeWebSocket.instances[0];
  first.open();
  assert.equal(socket.readyState,1);
  first.receive({plate:"GJ01AB1234"});
  assert.equal(messages.length,1);
  first.close();
  t.mock.timers.tick(2000);
  assert.equal(FakeWebSocket.instances.length,2);
  const second=FakeWebSocket.instances[1];
  assert.equal(second.url,first.url);
  second.open();
  assert.equal(socket.send({action:"correct_event"}),true);
  socket.close();
  t.mock.timers.tick(60000);
  assert.equal(FakeWebSocket.instances.length,2);
  assert.equal(socket.send({action:"correct_event"}),false);
});

test("ignores stale camera callbacks and malformed messages", t => {
  t.mock.timers.enable({apis:["setTimeout","setInterval"]});
  FakeWebSocket.instances=[];
  const messages=[];
  const socket=new Socket("ws://test/ws/events",{WebSocketClass:FakeWebSocket,onMessage:m=>messages.push(m)});
  const first=FakeWebSocket.instances[0];
  first.open();
  socket.reconnect();
  const second=FakeWebSocket.instances[1];
  first.receive({wrong:true});
  second.open();
  second.onmessage({data:"not json"});
  second.receive({type:"pong"});
  second.receive({type:"event_created"});
  assert.deepEqual(messages,[{type:"event_created"}]);
  socket.close();
});

test("connection timeout schedules retry", t => {
  t.mock.timers.enable({apis:["setTimeout","setInterval"]});
  FakeWebSocket.instances=[];
  const socket=new Socket("ws://test/ws/events",{WebSocketClass:FakeWebSocket});
  t.mock.timers.tick(10000);
  t.mock.timers.tick(2000);
  assert.equal(FakeWebSocket.instances.length,2);
  socket.close();
});

