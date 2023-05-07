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
import shutil
import importlib.metadata
from pathlib import Path
import nbconvert

# -- Project information (unique to each project) -------------------------------------

project = "rydiqule"
author = "Quantum Technology Center, DEVCOM Army Research Laboratory and Naval Air Warfare Center - Weapons Division"
latex_author = r"""Quantum Technology Center\and
DEVCOM Army Research Laboratory\and
Naval Air Warfare Center - Weapons Division\\
\\
\Large Unclassified - Approved for Public Release\\
\\
\normalsize Dr. Kevin C Cox - kevin.c.cox29.civ@army.mil\\
\normalsize Dr. Christopher O'brien - christopher.m.obrien22.civ@us.navy.mil\\
\normalsize Dr. David H Meyer - david.h.meyer3.civ@army.mil\\
\normalsize Mr. Benjamin Miller - benjamin.n.miller@navy.mil\\
"""

release = importlib.metadata.version('rydiqule')

version = release

# HTML icons
img_path = 'img'
html_logo = img_path + '/Rydiqule_Icon_64.svg'
html_favicon = img_path + '/Rydiqule_Icon_Transparent.ico'

# -- General configuration (should be identical across all projects) ------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
]

autosummary_generate = True
autodoc_typehints = 'signature'
autoclass_content = 'class'  # options: 'both', 'class', 'init'
numfig = True

# mathjax options for autonumbering equations
mathjax3_config = {'tex': {'tags': 'all', 'useLabelIds': True},}

# Prefix each autosectionlabel with the name of the document it is in and a colon
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The suffix(es) of source filenames.
source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'

# intersphinx allows us to link directly to other repos sphinxdocs.
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numbakitode': ('https://numbakit-ode.readthedocs.io/en/latest/', None),
}

# Make `some code` equivalent to :code:`some code`
default_role = 'code'

# hide todo notes if on readthedocs and not building the latest
if os.environ.get('READTHEDOCS') and os.environ.get('READTHEDOCS_VERSION') != 'latest':
    todo_include_todos = False
else:
    todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_title = "rydiqule"
html_short_title = "rydiqule"
html_show_copyright = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Customize the html_theme
html_theme_options = {'navigation_depth': 3}

# -- Options for LaTeX output ---------------------------------------------

latex_engine = 'lualatex'

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
'papersize': 'a4paper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
'preamble': r'''\usepackage{braket}''',

# Latex figure (float) alignment
'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'rydiqule.tex', project,
    latex_author,
    'manual')
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = img_path + '/Rydiqule_Logo_Transparent_300.png'

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
latex_use_parts = True

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True

def setup(app):

    app.connect('builder-inited', run_nbconvert)

def run_nbconvert(_):
    """
    Runs nbconvert on all example notebooks, for inclusion in docs.
    """

    if os.environ.get('READTHEDOCS'):
        rel_path = '../..'
        imgs_path = os.path.abspath('./img')
    else:
        rel_path = '..'
        imgs_path = os.path.abspath('./source/img')

    examples_path = os.path.join(os.path.abspath(rel_path),'examples')

    out_base_path = os.path.join(os.path.dirname(Path(__file__)),'_examples')
    intro_out_base_path = os.path.join(os.path.dirname(Path(__file__)), '_intro_nbs')

    examples = Path(examples_path).glob('*.ipynb')
    intro_nbs = ['Introduction_To_Rydiqule']
    img_assets = ['Rydiqule_Logo_Transparent_300.png']

    rst_exporter = nbconvert.exporters.RSTExporter()
    writer = nbconvert.writers.FilesWriter()

    for ex in examples:
        name = ex.stem
        if name in intro_nbs:  
            out_path = os.path.join(intro_out_base_path,name)
        else:
            out_path = os.path.join(out_base_path,name)
        writer.build_directory = out_path
        (body, resources) = rst_exporter.from_filename(ex)
        writer.write(body,resources,name)
        # copy image assets to every directory
        for img in img_assets:
            shutil.copy(os.path.join(imgs_path,img), out_path)
