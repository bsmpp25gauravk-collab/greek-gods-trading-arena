// Prevent form data changing on second click — lock inputs after first submit
document.addEventListener('DOMContentLoaded', function () {
  const calcForm = document.getElementById('calc-form');
  if (calcForm) {
    // Store submitted values and re-apply them after page load
    // to prevent browser autofill from changing them
    const inputs = calcForm.querySelectorAll('input[type="number"]');
    inputs.forEach(inp => {
      inp.addEventListener('focus', () => {
        inp.select();
      });
    });
  }
});

// Study page tab switcher
function showTab(tab) {
  ['primary','advanced'].forEach(t => {
    const el = document.getElementById('tab-' + t);
    const btn = document.getElementById('tab-btn-' + t);
    if (el)  el.style.display  = (t === tab) ? '' : 'none';
    if (btn) btn.classList.toggle('active', t === tab);
  });
}
