// Switch to the light favicon when the browser UI is dark. The href set by
// Material is already relative to the current page, so only the file name
// is swapped.
(function () {
  const query = window.matchMedia("(prefers-color-scheme: dark)");

  function updateFavicon() {
    const favicon = document.querySelector('link[rel="icon"]');
    if (!favicon) return;
    const name = query.matches ? "favicon-dark.png" : "favicon.png";
    favicon.href = favicon.href.replace(/favicon(-dark)?\.png$/, name);
  }

  updateFavicon();
  query.addEventListener("change", updateFavicon);
})();
