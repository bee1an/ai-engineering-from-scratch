/**
 * Language switcher for AI Engineering from Scratch.
 *
 * State: persisted in localStorage('lang') with values 'en' | 'zh'.
 * Default: detect from browser, fallback to 'en'.
 *
 * Pages opt in by including this script and exposing a `.lang-toggle`
 * button (or the script will inject one into `.header-nav` if missing).
 *
 * On change: dispatches a `lang:change` CustomEvent on `document` with
 * detail `{ lang }` so pages can re-render content.
 *
 * Static text translation: any element with `data-i18n="key"` will have
 * its text replaced from the I18N dictionary below. Keys with the suffix
 * `:html` are inserted as innerHTML (use sparingly, only for trusted
 * static strings).
 */
(function () {
  var STORAGE_KEY = 'lang';
  var SUPPORTED = ['en', 'zh'];

  var I18N = {
    'nav.contents':   { en: 'Contents',  zh: '目录' },
    'nav.catalog':    { en: 'Catalog',   zh: '课程列表' },
    'nav.roadmap':    { en: 'Roadmap',   zh: '学习路线' },
    'nav.glossary':   { en: 'Glossary',  zh: '术语表' },

    'index.tagline':  { en: '473 lessons. 20 phases. Every algorithm built from raw math before a single framework gets imported.',
                        zh: '473 节课。20 个阶段。每个算法都从原始数学起步，所有框架在那之后才登场。' },
    'index.attribution': { en: 'Maintained by Rohit Ghumare and contributors. Run on your own machine.',
                           zh: '由 Rohit Ghumare 与贡献者们维护。在你自己的机器上运行。' },
    'index.cta.star': { en: 'Star on GitHub', zh: 'GitHub 点星' },
    'index.cta.follow': { en: 'Follow @bee1an', zh: '关注 @bee1an' },
    'index.preface.eyebrow': { en: 'How this works', zh: '它是如何工作的' },
    'index.preface.p1': {
      en: "Most AI material teaches in scattered pieces. A paper here, a fine-tuning post there, a flashy agent demo somewhere else. The pieces rarely line up. You ship a chatbot but can't explain its loss curve. You hook a function to an agent but can't say what attention does inside the model that's calling it.",
      zh: '大多数 AI 资料都是碎片化的：这里一篇论文，那里一篇微调博客，再加上一个炫酷的 agent demo。这些碎片很少对得齐。你能上线一个聊天机器人，但说不清它的 loss 曲线；你能把函数挂到 agent 上，却讲不出调用它的模型内部 attention 在做什么。'
    },
    'index.preface.p2': {
      en: 'This curriculum is the spine. 20 phases, 473 lessons, four languages: Python, TypeScript, Rust, Julia. Linear algebra at one end, autonomous swarms at the other. Every algorithm gets built from raw math first. Backprop. Tokenizer. Attention. Agent loop. By the time PyTorch shows up, you already know what it\'s doing under the hood.',
      zh: '这门课程就是那根脊梁。20 个阶段、473 节课、四种语言：Python、TypeScript、Rust、Julia。一头是线性代数，另一头是自主群体。每个算法都先从原始数学搭出来。Backprop、Tokenizer、Attention、Agent loop。等到 PyTorch 登场，你已经知道它在底下做什么了。'
    },
    'index.preface.p3': {
      en: 'Each lesson runs the same loop: read the problem, derive the math, write the code, run the test, keep the artifact. No five-minute videos, no copy-paste deploys, no hand-holding. Free, open source, and built to run on your own laptop.',
      zh: '每节课跑同样一套流程：读问题、推数学、写代码、跑测试、留产出。没有五分钟视频，没有复制粘贴部署，没有手把手保姆。免费、开源，且能在你自己的笔记本上跑。'
    },

    'index.stats.title':       { en: 'Current Progress',  zh: '当前进度' },
    'index.stats.finished':    { en: 'Finished Lessons',  zh: '已完成课程' },
    'index.stats.phases':      { en: 'Phases',            zh: '阶段' },
    'index.stats.languages':   { en: 'Languages',         zh: '编程语言' },
    'index.stats.glossary':    { en: 'Glossary Terms',    zh: '术语数量' },

    'index.toc.title':    { en: 'Curriculum · 20 phases · 473 lessons', zh: '课程 · 20 个阶段 · 473 节课' },
    'index.toc.subtitle': { en: 'Tap a phase to expand its lessons. Each one ships when its math, code, and test are all written.',
                            zh: '点击某个阶段查看它的课程列表。每节课的数学、代码、测试都写完之后才会发布。' },
    'index.legend.complete':    { en: 'Complete',     zh: '已完成' },
    'index.legend.inprogress':  { en: 'In progress',  zh: '进行中' },
    'index.legend.planned':     { en: 'Planned',      zh: '计划中' },

    'index.colophon.eyebrow': { en: 'Colophon', zh: '版本说明' },
    'index.colophon.body': {
      en: 'The entire curriculum is on GitHub. Clone it, fork it, learn at your own pace. No paywall, no signup. Every lesson has runnable code in Python, TypeScript, Rust, or Julia, depending on what fits the concept best.',
      zh: '整套课程都在 GitHub 上。Clone、fork 都行，按自己的节奏学。不收钱、不注册。每节课都有可跑的代码——Python、TypeScript、Rust 或 Julia 中最适合那个概念的那种。'
    },
    'index.modal.note':  { en: 'Progress saved in browser only', zh: '进度只保存在你的浏览器里' },
    'index.modal.reset': { en: 'Reset progress', zh: '清除进度' },

    'footer.copy':     { en: '© 2026 · open source · free forever', zh: '© 2026 · 开源 · 永远免费' },
    'footer.report':   { en: 'Report', zh: '反馈问题' },
    'footer.report.long':   { en: 'Report / Suggest', zh: '反馈 / 建议' },

    'lesson.loading':  { en: 'Loading lesson...', zh: '课程加载中...' },
    'lesson.notfound.title': { en: 'Lesson not found', zh: '课程未找到' },
    'lesson.notfound.msg': { en: 'Could not fetch the lesson at', zh: '无法获取课程' },
    'lesson.notfound.written': { en: 'It may not have been written yet.', zh: '该课程可能尚未编写。' },
    'lesson.back':     { en: 'Back to Home', zh: '返回首页' },
    'lesson.prev':     { en: 'Previous', zh: '上一课' },
    'lesson.next':     { en: 'Next', zh: '下一课' },
    'lesson.quiz.correct': { en: 'correct', zh: '正确' },
    'lesson.quiz.complete': { en: 'Complete all questions to see your score', zh: '完成所有题目后查看分数' },
    'lesson.diagram':   { en: 'Diagram', zh: '图表' },
    'lesson.close':     { en: 'Close', zh: '关闭' },
    'lesson.expand':    { en: 'Expand', zh: '展开' },
    'lesson.copy':      { en: 'Copy', zh: '复制' },
    'lesson.copied':    { en: 'Copied!', zh: '已复制！' },
    'lesson.copycmd':   { en: 'Copy command', zh: '复制命令' },

    'glossary.title':    { en: 'Glossary', zh: '术语表' },
    'glossary.subtitle': { en: 'What people say. What it actually means.', zh: '大家怎么说。它实际上是什么。' },
    'glossary.search':   { en: 'Search terms...', zh: '搜索术语...' },

    'catalog.title':     { en: 'Catalog', zh: '课程目录' },
    'catalog.subtitle':  { en: 'Every lesson, every phase, every language.', zh: '所有课程、所有阶段、所有语言。' },
    'catalog.search':    { en: 'Search lessons...', zh: '搜索课程...' },

    'prereqs.title':     { en: 'Roadmap', zh: '学习路线' },
    'prereqs.subtitle':  { en: 'Prerequisites and learning paths.', zh: '前置依赖与学习路径。' },
  };

  function detectInitial() {
    try {
      var stored = localStorage.getItem(STORAGE_KEY);
      if (stored && SUPPORTED.indexOf(stored) >= 0) return stored;
    } catch (e) { /* localStorage may be disabled */ }
    var nav = (navigator.language || '').toLowerCase();
    if (nav.indexOf('zh') === 0) return 'zh';
    return 'en';
  }

  var current = detectInitial();
  document.documentElement.setAttribute('data-lang', current);
  document.documentElement.setAttribute('lang', current === 'zh' ? 'zh-CN' : 'en');

  function persist(lang) {
    try { localStorage.setItem(STORAGE_KEY, lang); } catch (e) {}
  }

  function translate(key, lang) {
    var entry = I18N[key];
    if (!entry) return null;
    return entry[lang] != null ? entry[lang] : entry.en;
  }

  function applyTranslations() {
    var els = document.querySelectorAll('[data-i18n]');
    for (var i = 0; i < els.length; i++) {
      var el = els[i];
      var raw = el.getAttribute('data-i18n');
      var asHtml = false;
      var key = raw;
      if (raw.indexOf(':html') === raw.length - 5) {
        asHtml = true;
        key = raw.slice(0, -5);
      }
      var val = translate(key, current);
      if (val == null) continue;
      if (asHtml) el.innerHTML = val;
      else el.textContent = val;
    }
    var attrEls = document.querySelectorAll('[data-i18n-attr]');
    for (var j = 0; j < attrEls.length; j++) {
      var ae = attrEls[j];
      // Format: "attr1:key1;attr2:key2"
      var spec = ae.getAttribute('data-i18n-attr');
      var pairs = spec.split(';');
      for (var k = 0; k < pairs.length; k++) {
        var bits = pairs[k].split(':');
        if (bits.length !== 2) continue;
        var attr = bits[0].trim();
        var keyA = bits[1].trim();
        var v = translate(keyA, current);
        if (v != null) ae.setAttribute(attr, v);
      }
    }
  }

  function setLang(lang, opts) {
    if (SUPPORTED.indexOf(lang) < 0) return;
    if (lang === current && !(opts && opts.force)) return;
    current = lang;
    persist(lang);
    document.documentElement.setAttribute('data-lang', lang);
    document.documentElement.setAttribute('lang', lang === 'zh' ? 'zh-CN' : 'en');
    paintToggleButtons();
    applyTranslations();
    document.dispatchEvent(new CustomEvent('lang:change', { detail: { lang: lang } }));
  }

  function getLang() { return current; }

  function paintToggleButtons() {
    var btns = document.querySelectorAll('.lang-toggle');
    for (var i = 0; i < btns.length; i++) {
      var btn = btns[i];
      btn.textContent = current === 'zh' ? 'EN' : '中';
      btn.setAttribute('aria-label', current === 'zh' ? 'Switch to English' : '切换到中文');
      btn.setAttribute('title', current === 'zh' ? 'Switch to English' : '切换到中文');
    }
  }

  function injectToggleIfMissing() {
    if (document.querySelector('.lang-toggle')) return;
    var nav = document.querySelector('.header-inner');
    if (!nav) return;
    var themeBtn = nav.querySelector('.theme-toggle');
    var btn = document.createElement('button');
    btn.className = 'lang-toggle';
    btn.type = 'button';
    nav.insertBefore(btn, themeBtn || null);
  }

  function bind() {
    document.addEventListener('click', function (e) {
      var t = e.target;
      while (t && t !== document) {
        if (t.classList && t.classList.contains('lang-toggle')) {
          setLang(current === 'zh' ? 'en' : 'zh');
          return;
        }
        t = t.parentNode;
      }
    });
  }

  function init() {
    injectToggleIfMissing();
    paintToggleButtons();
    applyTranslations();
    bind();
    document.dispatchEvent(new CustomEvent('lang:ready', { detail: { lang: current } }));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.AIFS_LANG = { get: getLang, set: setLang, t: function (k) { return translate(k, current); } };
})();
