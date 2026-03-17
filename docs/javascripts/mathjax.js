window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams",                          // auto equation numbering via \label/\ref
    packages: { "[+]": ["boldsymbol"] },  // \boldsymbol support
    macros: {
      // Convenience macros for AI/ML notation
      R:       "\\mathbb{R}",
      E:       "\\mathbb{E}",
      N:       "\\mathcal{N}",
      bx:      "\\mathbf{x}",
      bW:      "\\mathbf{W}",
      btheta:  "\\boldsymbol{\\theta}",
      softmax: "\\operatorname{softmax}",
      sigmoid: "\\operatorname{\\sigma}",
      attn:    "\\operatorname{Attention}",
      norm:    ["{\\|#1\\|}", 1],
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  loader: {
    load: ["[tex]/boldsymbol"]
  },
  startup: {
    // Re-typeset on instant navigation (Material's custom event)
    ready() {
      MathJax.startup.defaultReady();
      document$.subscribe(() => MathJax.typesetPromise());
    }
  }
};
