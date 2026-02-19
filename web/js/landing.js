// ── Scroll animations ──────────────────────────────────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach(el => {
    if (el.isIntersecting) {
      el.target.classList.add('visible');
      observer.unobserve(el.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll(
  '.pv-step, .agent-card, .math-card, .metric-item'
).forEach(el => {
  el.classList.add('fade-up');
  observer.observe(el);
});

// ── Stagger children ───────────────────────────────────────────────────
document.querySelectorAll('.pipeline-visual, .agent-grid, .math-grid, .metrics-grid')
  .forEach(grid => {
    [...grid.children].forEach((child, i) => {
      child.style.transitionDelay = `${i * 80}ms`;
    });
  });

// ── Nav active state ───────────────────────────────────────────────────
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');

window.addEventListener('scroll', () => {
  let current = '';
  sections.forEach(s => {
    if (window.scrollY >= s.offsetTop - 100) current = s.id;
  });
  navLinks.forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === `#${current}`);
  });
}, { passive: true });
