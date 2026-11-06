// Site analytics for privateducksheep.github.io, sent to PostHog.
// Shaun reads the results in his own PostHog dashboard; nothing here shows them on the site.
// The project key below is PostHog's public write-only key: it can send events but not read them.
(() => {
  const KEY = "";                                          // phc_... from PostHog → Project settings
  const HOST = "https://us.i.posthog.com";     // or https://eu.i.posthog.com for an EU project
  if (!KEY || navigator.doNotTrack === "1" || /localhost|127\.0\.0\.1/.test(location.hostname) && !location.search.includes("ph_test")) return;

  // PostHog's standard loader
  !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.crossOrigin="anonymous",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="init capture register register_once unregister identify alias set_config reset opt_in_capturing opt_out_capturing has_opted_out_capturing getFeatureFlag isFeatureEnabled onFeatureFlags setPersonProperties startSessionRecording stopSessionRecording".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);

  posthog.init(KEY, {
    api_host: HOST,
    person_profiles: "always",
    capture_pageview: "history_change",   // also counts page changes inside the AI Hub app
    capture_pageleave: true,              // gives time on page
    autocapture: true,
    persistence: "localStorage+cookie",
    session_recording: { maskAllInputs: true },
  });

  // ?ref=grab-swe marks which application link a visitor came from.
  // It sticks to the visitor, so later visits without the tag still count under it.
  const ref = new URLSearchParams(location.search).get("ref");
  if (ref) {
    posthog.register({ ref });
    posthog.setPersonProperties({ ref }, { first_ref: ref });
  }

  const site = location.pathname.startsWith("/ai-hub") ? "ai-hub"
    : location.pathname.startsWith("/audio-fft") ? "audio-fft"
    : location.pathname.startsWith("/resume") ? "resume" : "portfolio";
  posthog.register({ site });

  const projectOf = (el) => el?.closest("article")?.querySelector("h3")?.textContent.trim() || null;

  // which project someone opened, and for how long
  let opened = null;
  document.addEventListener("click", (e) => {
    const card = e.target.closest(".card.has-trailer");
    if (card && !e.target.closest("a")) {
      opened = { project: projectOf(card), at: Date.now() };
      posthog.capture("project_opened", { project: opened.project });
    }
    const a = e.target.closest("a[href]");
    if (a) posthog.capture("link_clicked", {
      label: a.textContent.trim().replace(/\s+/g, " ").replace(/\s*→$/, ""),
      href: a.href,
      project: projectOf(a),
      section: a.closest("section[id]")?.id || null,
    });
    if (e.target.closest("#copy")) posthog.capture("email_copied");
    if (e.target.closest("#b-pdf")) posthog.capture("resume_pdf_downloaded");
  }, true);

  const viewer = document.getElementById("viewer");
  if (viewer) new MutationObserver(() => {
    if (viewer.hidden && opened) {
      posthog.capture("project_closed", { project: opened.project, seconds: Math.round((Date.now() - opened.at) / 1000) });
      opened = null;
    }
  }).observe(viewer, { attributes: true, attributeFilter: ["hidden"] });

  document.addEventListener("play", (e) => {
    // the viewer swaps in a seekable copy mid-play, so count one play per opening
    if (e.target.closest?.("#viewer") && opened && !opened.played) {
      opened.played = true;
      posthog.capture("trailer_played", { project: opened.project });
    }
  }, true);

  // how far down the page people get
  const seen = new Set();
  const io = new IntersectionObserver((entries) => entries.forEach((en) => {
    const id = en.target.id;
    if (en.isIntersecting && !seen.has(id)) { seen.add(id); posthog.capture("section_viewed", { section: id }); }
  }), { threshold: 0.25 });
  document.querySelectorAll("section[id], header[id]").forEach((s) => io.observe(s));
})();
