// Fix drawer visibility in intermediate resolutions
document.addEventListener('DOMContentLoaded', function () {
  const drawerToggle = document.querySelector('[data-md-toggle="drawer"]');
  const drawer = document.querySelector('.md-sidebar--primary');

  if (drawerToggle && drawer) {
    drawerToggle.addEventListener('change', function () {
      if (this.checked) {
        // Force drawer to be visible
        drawer.style.transform = 'translateX(0)';
        drawer.style.zIndex = '200';
        drawer.style.visibility = 'visible';
        drawer.style.display = 'block';
      } else {
        // Reset when closed
        drawer.style.transform = '';
        drawer.style.zIndex = '';
      }
    });
  }

  // Adaptive favicon for dark mode
  function updateFavicon() {
    const favicon = document.querySelector('link[rel="icon"]');
    if (!favicon) return;

    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    favicon.href = isDark ? 'images/favicon-dark.png' : 'images/favicon.png';
  }

  // Update on load
  updateFavicon();

  // Update when theme changes
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateFavicon);
});
