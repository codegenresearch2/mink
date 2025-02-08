# Configuration file for the Sphinx documentation builder. This file is intended to provide configuration settings for generating documentation using Sphinx. It includes settings for project information, general configuration, and options for HTML output. The documentation is aimed at enhancing understanding and clarity for users.

# -- Project information ----------------------------------------------------- This section sets up basic information about the project, such as the project name, copyright, and author.

project = "mink"
copyright = "2024, Kevin Zakka"
author = "Kevin Zakka"

# -- General configuration --------------------------------------------------- This section configures various extensions and settings used by Sphinx to build the documentation.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx-mathjax-offline",
    "sphinx.ext.napoleon",
    "sphinx_favicon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = { ".rst": "restructuredtext" }

pygments_style = "sphinx"

napoleon_numpy_docstring = False
napoleon_use_rtype = False

# -- Options for HTML output ------------------------------------------------- This section configures the appearance and output format of the HTML documentation.

html_theme = "sphinx_rtd_theme"

htmlhelp_basename = "minkdoc"
