# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

# Add parent directory to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Project information -----

project = "SPINOTOR"
copyright = "2026, Amine Khettat"
author = "Amine Khettat"
release = "0.12.0"

# -- General configuration ---

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

rst_prolog = """
.. note::

   License reminder: this project is distributed under a Custom restricted license (no redistribution). See the repository LICENSE file.
   Disclaimer: this simulator is provided as-is for research, calibration, and educational use. Users assume all risks,
   including any direct or indirect damage, data loss, hardware damage, injury, or regulatory non-compliance arising
   from use or misuse.
"""

# Autosummary
autosummary_generate = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ---

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2c3e50",
}

# -- Options for intersphinx extension ---

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
