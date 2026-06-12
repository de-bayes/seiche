/* Making-Software ruler rail: ticks, proportional section labels, scroll
   cursor with fraction readout. Pages set window.MS_SECTIONS first. */
    (function () {
      var rail = document.getElementById('ms-ruler');
      if (!rail) return;
      var ticksEl = document.getElementById('ms-ruler-ticks');
      var labelsEl = document.getElementById('ms-ruler-labels');
      var cursor = document.getElementById('ms-ruler-cursor');
      var fracEl = document.getElementById('ms-ruler-frac');

      // section id -> short mono label, set per page before this script loads
      var SECTIONS = window.MS_SECTIONS || [];

      // fine ticks down the whole rail height (one every ~14px), built once
      var TICK_N = 56;
      var tickFrag = document.createDocumentFragment();
      for (var i = 0; i <= TICK_N; i++) {
        var tk = document.createElement('div');
        tk.className = 'ms-ruler__tick' + (i % 5 === 0 ? ' is-major' : '');
        tk.style.top = (i / TICK_N * 100) + '%';
        tickFrag.appendChild(tk);
      }
      ticksEl.appendChild(tickFrag);

      // build label nodes (positions set in layout())
      var labelNodes = [];
      SECTIONS.forEach(function (sec) {
        var el = document.getElementById(sec[0]);
        if (!el) return;
        var a = document.createElement('a');
        a.className = 'ms-ruler__label';
        a.href = '#' + sec[0];
        a.textContent = sec[1];
        a.addEventListener('click', function (ev) {
          ev.preventDefault();
          var target = document.getElementById(sec[0]);
          if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
        labelsEl.appendChild(a);
        labelNodes.push({ el: el, node: a });
      });

      function docHeight() {
        return Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);
      }

      // place each label at the vertical fraction of its section's top in the doc
      function layout() {
        var h = docHeight() - window.innerHeight;
        var total = docHeight();
        labelNodes.forEach(function (rec) {
          var top = rec.el.getBoundingClientRect().top + window.scrollY;
          var frac = total > 0 ? top / total : 0;
          rec.node.style.top = (frac * 100) + '%';
        });
        update();
      }

      function update() {
        var max = docHeight() - window.innerHeight;
        var frac = max > 0 ? window.scrollY / max : 0;
        if (frac < 0) frac = 0; if (frac > 1) frac = 1;
        cursor.style.top = (frac * 100) + '%';
        fracEl.textContent = frac.toFixed(2);
        // highlight the label whose section is current
        var mid = window.scrollY + window.innerHeight * 0.33;
        var activeIdx = 0;
        for (var i = 0; i < labelNodes.length; i++) {
          var top = labelNodes[i].el.getBoundingClientRect().top + window.scrollY;
          if (top <= mid) activeIdx = i;
        }
        for (var j = 0; j < labelNodes.length; j++) {
          labelNodes[j].node.classList.toggle('is-active', j === activeIdx);
        }
      }

      window.addEventListener('scroll', update, { passive: true });
      window.addEventListener('resize', layout);
      if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(layout);
      }
      // late relayout once canvases/images settle their heights
      window.addEventListener('load', function () { setTimeout(layout, 60); });
      layout();
    })();
