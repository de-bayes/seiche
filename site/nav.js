/* Pill nav highlight: a single soft rectangle that slides between items on
   hover and glides back to the current page on leave. */
(function () {
  'use strict';
  var pill = document.querySelector('.pill');
  if (!pill) return;
  var hl = document.createElement('span');
  hl.className = 'pill__hl';
  pill.insertBefore(hl, pill.firstChild);
  var links = Array.prototype.slice.call(pill.querySelectorAll('a'));
  var active = pill.querySelector('a[aria-current="page"]') || links[0];

  function moveTo(el, instant) {
    if (instant) hl.style.transition = 'none';
    hl.style.width = el.offsetWidth + 'px';
    hl.style.transform = 'translateX(' + el.offsetLeft + 'px)';
    if (instant) {
      void hl.offsetWidth;  /* flush so the next move animates again */
      hl.style.transition = '';
    }
  }

  moveTo(active, true);
  links.forEach(function (a) {
    a.addEventListener('mouseenter', function () { moveTo(a); });
    a.addEventListener('focus', function () { moveTo(a); });
  });
  pill.addEventListener('mouseleave', function () { moveTo(active); });
  window.addEventListener('resize', function () { moveTo(active, true); });
  /* Lora loads async and reflows the items; re-seat the highlight after */
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(function () { moveTo(active, true); });
  }
})();
