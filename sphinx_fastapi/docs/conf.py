import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
project = "Energy Model API"
author = "Your Name"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinxcontrib.openapi",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]
autodoc_member_order = "bysource"
