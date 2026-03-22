// Greek Gods Trading Arena — main.js
// Small enhancements: active nav, smooth scroll, number formatting

document.addEventListener('DOMContentLoaded', () => {

  // Highlight active nav link based on current URL
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(link => {
    if (link.getAttribute('href') === path) {
      link.classList.add('active');
    }
  });

  // Smooth scroll for anchor links on the study page
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // Auto-format: flash green/red on result values after form submit
  document.querySelectorAll('.ri-value, .sc-pnl').forEach(el => {
    const text = el.textContent;
    if (text.includes('-') && !el.classList.contains('highlight-price')) {
      el.style.color = 'var(--red)';
    }
  });

  // Simple fade-in on scroll for cards
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }
      });
    },
    { threshold: 0.1 }
  );

  document.querySelectorAll('.greek-card, .level-card, .ql-card, .sc-card, .greek-section').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(16px)';
    el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(el);
  });

});
