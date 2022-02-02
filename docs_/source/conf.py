# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.append(os.path.join(os.pardir, os.pardir, "hnmf"))


# -- Project information -----------------------------------------------------

project = "hNMF"
copyright = "2019 - 2022, Eric Stasney"
author = "Eric Stasney"

# The full version, including alpha/beta/rc tags
release = "0.2.2"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "numpydoc"]

# Napoleon settings
napoleon_include_private_with_doc = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {"analytics_id": "UA-132355416-4"}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["./custom_styles"]

html_context = {
    "display_github": True,
    "github_user": "estasney",
    "github_repo": "hNMF",
    "github_version": "master/docs_/source/",
    "github_url": "https://github.com/estasney/hNMF",
}


rst_epilog = """
.. |project| replace:: hNMF
"""
