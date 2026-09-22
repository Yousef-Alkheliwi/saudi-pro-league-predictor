(function(){
"use strict";
var DATA = JSON.parse(document.getElementById("spl-data").textContent);
var CLUBS = DATA.clubs, META = DATA.meta, MODEL = DATA.model;
var BY_ID = {}, PAIR = {};
CLUBS.forEach(function(c){ BY_ID[c.id] = c; });
DATA.pairings.forEach(function(p){ PAIR[p.home + ">" + p.away] = p; });

var $ = function(id){ return document.getElementById(id); };
var pct = function(x){ return (100*x).toFixed(1) + "%"; };
var pct0 = function(x){ return Math.round(100*x) + "%"; };
var odds = function(x){ return x > 0.0005 ? (1/x).toFixed(2) : "–"; };
var sgn = function(x){ return (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2); };

/* ---------- tooltip ---------- */
var tip = $("tip");
function showTip(e, text){
  tip.textContent = text; tip.classList.add("on");
  var r = tip.getBoundingClientRect();
  var x = e.clientX + 12, y = e.clientY - r.height - 10;
  if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - 12;
  if (y < 4) y = e.clientY + 16;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip(){ tip.classList.remove("on"); }
function bindTip(el, text){
  el.addEventListener("mouseenter", function(e){ showTip(e, text); });
  el.addEventListener("mousemove", function(e){ showTip(e, text); });
  el.addEventListener("mouseleave", hideTip);
}

/* ---------- theme ---------- */
$("themebtn").addEventListener("click", function(){
  var root = document.documentElement;
  var explicit = root.getAttribute("data-theme");
  var dark = explicit ? explicit === "dark"
    : window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.setAttribute("data-theme", dark ? "light" : "dark");
  try { localStorage.setItem("spl-theme", dark ? "light" : "dark"); } catch(err){}
});
try {
  var savedTheme = localStorage.getItem("spl-theme");
  if (savedTheme) document.documentElement.setAttribute("data-theme", savedTheme);
} catch(err){}

/* ---------- tabs ---------- */
var TABS = [["tab-predict","view-predict"],["tab-ratings","view-ratings"],
            ["tab-method","view-method"]];
TABS.forEach(function(t){
  $(t[0]).addEventListener("click", function(){
    TABS.forEach(function(o){
      var on = o[0] === t[0];
      $(o[0]).setAttribute("aria-selected", on ? "true" : "false");
      $(o[1]).hidden = !on;
    });
  });
});

/* ---------- notice ---------- */
if (META.warnings && META.warnings.length){
  var joined = META.warnings.join(" ");
  var parts = [];
  if (META.plan_limited){
    parts.push("<p><strong>These are not live predictions.</strong> The API plan"
      + " behind this page only serves seasons " + META.seasons.join(", ")
      + ", so the newest match in the model is <strong>"
      + META.newest_match_label + "</strong>. Squads, managers and form have"
      + " moved on since.</p>");
  }
  if (!META.injuries_available){
    parts.push("<p>No injury feed is available on that plan either, so"
      + " <strong>every squad here is treated as fully fit</strong>. The injury"
      + " model is built and tested — it just has nothing to eat.</p>");
  }
  parts.push("<p>Everything else is real: " + META.n_matches.toLocaleString()
    + " actual Saudi Pro League matches and " + META.box_score_rows
    + " real box scores.</p>");
  $("noticebody").innerHTML = parts.join("");
  $("notice").hidden = false;
}

/* ---------- selects ---------- */
var homesel = $("homesel"), awaysel = $("awaysel");
CLUBS.forEach(function(c){
  [homesel, awaysel].forEach(function(sel){
    var o = document.createElement("option");
    o.value = c.id; o.textContent = c.name;
    sel.appendChild(o);
  });
});
function findId(frag){
  for (var i = 0; i < CLUBS.length; i++){
    if (CLUBS[i].name.toLowerCase().indexOf(frag) >= 0) return CLUBS[i].id;
  }
  return CLUBS[0].id;
}
var startHome = findId("hilal"), startAway = findId("nassr");
try {
  var saved = JSON.parse(localStorage.getItem("spl-fixture") || "null");
  if (saved && BY_ID[saved[0]] && BY_ID[saved[1]] && saved[0] !== saved[1]){
    startHome = saved[0]; startAway = saved[1];
  }
} catch(err){}
homesel.value = startHome; awaysel.value = startAway;

function guard(changed){
  if (homesel.value === awaysel.value){
    var other = CLUBS.filter(function(c){ return String(c.id) !== homesel.value; })[0];
    if (changed === "home") awaysel.value = other.id; else homesel.value = other.id;
  }
}
homesel.addEventListener("change", function(){ guard("home"); render(); });
awaysel.addEventListener("change", function(){ guard("away"); render(); });
$("swapbtn").addEventListener("click", function(){
  var h = homesel.value; homesel.value = awaysel.value; awaysel.value = h; render();
});

/* ---------- scoreline grid ---------- */
var RAMP = ["--h0","--h1","--h2","--h3","--h4","--h5","--h6"];
function buildGrid(p, homeName, awayName){
  var g = p.grid, n = g.length, host = $("sgrid"), peak = 0, pi = 0, pj = 0;
  for (var i = 0; i < n; i++) for (var j = 0; j < n; j++){
    if (g[i][j] > peak){ peak = g[i][j]; pi = i; pj = j; }
  }
  host.innerHTML = "";
  host.style.gridTemplateColumns = "auto repeat(" + n + ",minmax(30px,1fr))";
  var corner = document.createElement("div");
  corner.className = "sg-corner"; corner.textContent = "H↓";
  host.appendChild(corner);
  for (var c = 0; c < n; c++){
    var ax = document.createElement("div");
    ax.className = "sg-axis"; ax.textContent = c;
    host.appendChild(ax);
  }
  for (i = 0; i < n; i++){
    var rax = document.createElement("div");
    rax.className = "sg-axis"; rax.textContent = i;
    host.appendChild(rax);
    for (j = 0; j < n; j++){
      var v = g[i][j];
      var step = peak > 0 ? Math.min(6, Math.round(6 * Math.pow(v / peak, 0.55))) : 0;
      var cell = document.createElement("div");
      cell.className = "sg-cell" + (i === pi && j === pj ? " peak" : "");
      cell.style.background = "var(" + RAMP[step] + ")";
      cell.style.color = step >= 5 ? "#fff" : "var(--ink)";
      if (v >= 0.03) cell.textContent = Math.round(100 * v);
      bindTip(cell, homeName + " " + i + "–" + j + " " + awayName
        + "  ·  " + pct(v));
      host.appendChild(cell);
    }
  }
}

/* ---------- markets ---------- */
function buildMarkets(p, homeName, awayName){
  var m = p.markets;
  var rows = [
    ["Expected goals", p.xg.home.toFixed(2) + " – " + p.xg.away.toFixed(2), null],
    ["Expected total", p.xg.total.toFixed(2) + " goals", null],
    ["Over 2.5 goals", pct(m["over_2.5"]), m["over_2.5"]],
    ["Under 2.5 goals", pct(m["under_2.5"]), m["under_2.5"]],
    ["Over 1.5 goals", pct(m["over_1.5"]), m["over_1.5"]],
    ["Over 3.5 goals", pct(m["over_3.5"]), m["over_3.5"]],
    ["Both teams to score", pct(m.btts), m.btts],
    [homeName + " clean sheet", pct(m.home_cs), m.home_cs],
    [awayName + " clean sheet", pct(m.away_cs), m.away_cs]
  ];
  var host = $("mkt");
  host.innerHTML = "";
  rows.forEach(function(r){
    var d = document.createElement("div");
    d.className = "mrow";
    var nm = document.createElement("span"); nm.className = "nm"; nm.textContent = r[0];
    var vl = document.createElement("span"); vl.className = "vl num"; vl.textContent = r[1];
    d.appendChild(nm); d.appendChild(vl);
    if (r[2] !== null){
      var mt = document.createElement("div"); mt.className = "meter";
      var i = document.createElement("i"); i.style.width = (100*r[2]).toFixed(1) + "%";
      mt.appendChild(i); d.appendChild(mt);
    }
    host.appendChild(d);
  });
}

/* ---------- comparison bars ---------- */
function buildCompare(p, homeName, awayName){
  var t = p.tempo;
  var rows = [
    ["Possession", t.poss[0], t.poss[1], "%", 1],
    ["Total shots", t.shots[0], t.shots[1], "", 1],
    ["Shots on target", t.sot[0], t.sot[1], "", 1],
    ["Expected goals", p.xg.home, p.xg.away, "", 2]
  ];
  if (t.corners) rows.splice(3, 0, ["Corners", t.corners[0], t.corners[1], "", 1]);
  var host = $("cmp");
  host.innerHTML = "";
  rows.forEach(function(r){
    var a = r[1], b = r[2], dp = r[4], unit = r[3];
    var max = Math.max(a, b) || 1;
    var wrap = document.createElement("div");
    wrap.className = "cmp-row";
    wrap.innerHTML =
      '<div class="top"><span class="vh num">' + a.toFixed(dp) + unit + '</span>'
      + '<span class="nm">' + r[0] + '</span>'
      + '<span class="va num">' + b.toFixed(dp) + unit + '</span></div>'
      + '<div class="cmp-track">'
      + '<span class="cmp-half l"><i style="width:' + (100*a/max) + '%"></i></span>'
      + '<span class="cmp-half r"><i style="width:' + (100*b/max) + '%"></i></span>'
      + '</div>';
    bindTip(wrap, r[0] + ": " + homeName + " " + a.toFixed(dp) + unit
      + "  vs  " + awayName + " " + b.toFixed(dp) + unit);
    host.appendChild(wrap);
  });
}

/* ---------- why ---------- */
function formSpan(results){
  return results.map(function(r){
    return '<span class="' + r + '">' + r + '</span>';
  }).join("");
}
function buildWhy(p, home, away){
  var c = p.ctx;
  var h2h = c.h2h;
  var rest = function(v){ return v === null ? "n/a" : v.toFixed(1) + "d"; };
  var rows = [
    ["Home advantage", "×" + MODEL.home_adv_mult.toFixed(2) + " on goal rate"],
    ["Attack rating", sgn(home.attack) + "  vs  " + sgn(away.attack)],
    ["Defence rating", sgn(home.defence) + "  vs  " + sgn(away.defence)],
    ["Rest days", rest(c.rest[0]) + "  vs  " + rest(c.rest[1])],
    ["Matches in last 14d", c.cong[0] + "  vs  " + c.cong[1]],
    ["Squad available", pct0(c.avail[0]) + "  vs  " + pct0(c.avail[1])],
    ["Head to head", h2h.n ? (h2h.w + "W–" + h2h.d + "D–" + h2h.l
        + "L in " + h2h.n) : "no meetings"],
    ["H2H average score", h2h.n ? (h2h.gf.toFixed(1) + "–" + h2h.ga.toFixed(1))
        : "–"],
    ["Form, " + home.name.split(" ")[0], formSpan(c.form[0].r) + "  "
        + c.form[0].ppg.toFixed(2) + " ppg", true],
    ["Form, " + away.name.split(" ")[0], formSpan(c.form[1].r) + "  "
        + c.form[1].ppg.toFixed(2) + " ppg", true],
    ["Confidence", p.conf]
  ];
  var host = $("why");
  host.innerHTML = "";
  rows.forEach(function(r){
    var d = document.createElement("div");
    d.className = "wrow";
    d.innerHTML = '<span class="k">' + r[0] + '</span><span class="v '
      + (r[2] ? "form-str" : "num") + '">' + r[1] + '</span>';
    host.appendChild(d);
  });
}

/* ---------- render ---------- */
function render(){
  var hid = parseInt(homesel.value, 10), aid = parseInt(awaysel.value, 10);
  var p = PAIR[hid + ">" + aid];
  if (!p) return;
  var home = BY_ID[hid], away = BY_ID[aid];
  try { localStorage.setItem("spl-fixture", JSON.stringify([hid, aid])); } catch(err){}

  $("homename").textContent = home.name;
  $("awayname").textContent = away.name;
  var rec = function(c){
    var r = c.record || {};
    return r.played ? (r.w + "–" + r.d + "–" + r.l + "  ·  " + r.pts + " pts")
      : "";
  };
  $("homemeta").textContent = rec(home);
  $("awaymeta").textContent = rec(away);

  var top = p.scores[0];
  $("heroscore").textContent = top.s.replace("-", "–");
  $("heropct").textContent = pct(top.p) + " · next " + p.scores[1].s.replace("-","–");

  var ph = p.p.home, pd = p.p.draw, pa = p.p.away;
  $("seg-home").style.flexGrow = ph; $("seg-draw").style.flexGrow = pd;
  $("seg-away").style.flexGrow = pa;
  $("seg-home-t").textContent = ph > 0.13 ? pct0(ph) : "";
  $("seg-draw-t").textContent = pd > 0.13 ? pct0(pd) : "";
  $("seg-away-t").textContent = pa > 0.13 ? pct0(pa) : "";
  bindTip($("seg-home"), home.name + " win · " + pct(ph) + " · fair odds " + odds(ph));
  bindTip($("seg-draw"), "Draw · " + pct(pd) + " · fair odds " + odds(pd));
  bindTip($("seg-away"), away.name + " win · " + pct(pa) + " · fair odds " + odds(pa));
  $("k-home").innerHTML = home.name + " <b>" + pct(ph) + "</b> &middot; " + odds(ph);
  $("k-draw").innerHTML = "Draw <b>" + pct(pd) + "</b> &middot; " + odds(pd);
  $("k-away").innerHTML = away.name + " <b>" + pct(pa) + "</b> &middot; " + odds(pa);

  buildGrid(p, home.name, away.name);
  $("gridaxis").innerHTML = "&larr; " + home.name + " goals down  &middot;  "
    + away.name + " goals across &rarr;";
  buildMarkets(p, home.name, away.name);
  buildCompare(p, home.name, away.name);
  buildWhy(p, home, away);

  var aux = MODEL.aux && MODEL.aux.shots;
  $("tempo-sub").textContent = aux
    ? ("Fitted on " + aux.n + " real team-match box scores. Bars share a scale per row.")
    : (MODEL.aux_note || "League priors scaled by expected goals.");
}

/* ---------- ratings table ---------- */
var sortKey = "net", sortDir = -1;
function drawTable(){
  var rows = CLUBS.slice();
  var get = function(c){
    var r = c.record || {};
    if (sortKey === "name") return c.name;
    if (sortKey === "pts") return r.pts || 0;
    if (sortKey === "rec") return r.w || 0;
    if (sortKey === "gd") return (r.gf || 0) - (r.ga || 0);
    return c[sortKey];
  };
  rows.sort(function(a, b){
    var x = get(a), y = get(b);
    if (typeof x === "string") return sortDir * x.localeCompare(y);
    return sortDir * (x - y);
  });
  var maxAbs = 0;
  CLUBS.forEach(function(c){ maxAbs = Math.max(maxAbs, Math.abs(c.net)); });
  var tb = document.querySelector("#rtable tbody");
  tb.innerHTML = "";
  rows.forEach(function(c){
    var r = c.record || {};
    var tr = document.createElement("tr");
    var w = maxAbs ? (100 * Math.abs(c.net) / maxAbs).toFixed(1) : 0;
    tr.innerHTML = "<td>" + c.name + "</td>"
      + '<td class="num">' + sgn(c.attack) + "</td>"
      + '<td class="num">' + sgn(c.defence) + "</td>"
      + '<td class="num"><b>' + sgn(c.net) + "</b></td>"
      + '<td><span class="bipolar"><span class="neg">'
        + (c.net < 0 ? '<i style="width:' + w + '%"></i>' : "")
        + '</span><span class="pos">'
        + (c.net >= 0 ? '<i style="width:' + w + '%"></i>' : "")
        + "</span></span></td>"
      + '<td class="num">' + (r.pts != null ? r.pts : "–") + "</td>"
      + '<td class="num">' + (r.played ? r.w + "–" + r.d + "–" + r.l : "–") + "</td>"
      + '<td class="num">' + (r.played ? sgnInt(r.gf - r.ga) : "–") + "</td>";
    tb.appendChild(tr);
  });
  document.querySelectorAll("#rtable th[data-k]").forEach(function(th){
    if (th.getAttribute("data-k") === sortKey){
      th.setAttribute("aria-sort", sortDir < 0 ? "descending" : "ascending");
    } else {
      th.removeAttribute("aria-sort");
    }
  });
}
function sgnInt(v){ return (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v); }
document.querySelectorAll("#rtable th[data-k]").forEach(function(th){
  th.addEventListener("click", function(){
    var k = th.getAttribute("data-k");
    if (k === sortKey) sortDir = -sortDir;
    else { sortKey = k; sortDir = k === "name" ? 1 : -1; }
    drawTable();
  });
});

/* ---------- method tab ---------- */
(function(){
  var s = [
    ["Matches fitted", MODEL.n_matches.toLocaleString()],
    ["Clubs", META.n_clubs],
    ["Home advantage", "×" + MODEL.home_adv_mult.toFixed(2)],
    ["Decay half-life", MODEL.half_life_days + "d"],
    ["Rho", sgn(MODEL.rho)],
    ["Box scores", META.box_score_rows]
  ];
  $("modelstats").innerHTML = s.map(function(r){
    return '<div><div class="k">' + r[0] + '</div><div class="v">' + r[1] + "</div></div>";
  }).join("");

  $("provenance").innerHTML = "Every figure comes from " + META.source
    + ", covering season" + (META.seasons.length > 1 ? "s " : " ")
    + META.seasons.join(", ") + " — " + META.n_matches.toLocaleString()
    + " finished matches up to " + META.newest_match_label + ", plus "
    + META.box_score_rows + " team-match box scores for the shots, "
    + "shots-on-target, possession and corners models. The page itself does no "
    + "modelling: all " + DATA.pairings.length + " club pairings are precomputed "
    + "by the Python model that the test suite covers, so what you see here and "
    + "what the command line prints cannot drift apart.";

  var limits = [
    "<b>The rest-days effect is weakly identified.</b> One season gives a standard "
    + "error near 0.07 on that coefficient, so a Gaussian prior shrinks it rather "
    + "than letting the fit claim that more rest hurts. Here it landed at "
    + sgn(MODEL.b_rest) + ".",
    "<b>The injury elasticity is a documented prior, not a fitted coefficient.</b> "
    + "Historical per-fixture injury lists are not cheaply retrievable, so there is "
    + "nothing to fit it on.",
    "<b>No lineup or transfer-window awareness.</b> A club that sold its top scorer "
    + "looks unchanged until enough matches accumulate.",
    "<b>Box scores are a small sample.</b> Each one costs an API request, so the "
    + "shots and possession models rest on " + META.box_score_rows
    + " team-match rows — team-level effects are shrunk hard toward the league mean."
  ];
  if (META.plan_limited){
    limits.unshift("<b>The data is not current.</b> The API plan serves only seasons "
      + META.seasons.join(", ") + ", so nothing after " + META.newest_match_label
      + " is in the model. A paid plan would pick up the current season, the live "
      + "injury feed and upcoming fixtures with no code change.");
  }
  if (!META.injuries_available){
    limits.splice(1, 0, "<b>No injury data at all on this plan.</b> Every squad is "
      + "shown at 100% available because the feed returned nothing, not because "
      + "everyone is fit.");
  }
  $("limits").innerHTML = limits.map(function(t){ return "<li>" + t + "</li>"; }).join("");

  $("rfoot").innerHTML = "Ratings fitted on " + MODEL.n_matches.toLocaleString()
    + " matches with a " + MODEL.half_life_days + "-day decay half-life and ridge "
    + MODEL.ridge + ". Points and records are the "
    + META.seasons[META.seasons.length-1] + "–"
    + String(META.seasons[META.seasons.length-1]+1).slice(2)
    + " league season only; the ratings also use the earlier seasons, weighted down, "
    + "which is why the two orders differ.";

  $("footmeta").innerHTML = "Saudi Pro League Match Lab · model fitted on "
    + MODEL.n_matches.toLocaleString() + " matches from " + META.source
    + " · snapshot " + (META.snapshot || "").slice(0, 10)
    + " · probabilistic forecasts, not predictions of certainty";
})();

drawTable();
render();
})();
