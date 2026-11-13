// Remove __tabbed anchors from URL when clicking tabs
document.addEventListener("click", function(e) {
  if (e.target.closest(".tabbed-labels")) {
    history.replaceState(null, "", window.location.pathname);
  }
});
