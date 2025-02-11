# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mink"
copyright = "2024, Kevin Zakka"
author = "Kevin Zakka"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx-mathjax-offline",
    "sphinx.ext.napoleon",
    "sphinx_favicon",
]

# Autodoc settings
autodoc_typehints = "both"
autodoc_class_signature = "separated"
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'inherited-members': False,
    'exclude-members': "__init__, __post_init__, __new__",
}

autodoc_type_aliases = {
    "npt.ArrayLike": "ArrayLike",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

htmlhelp_basename = "minkdoc"