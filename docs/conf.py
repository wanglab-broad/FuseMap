import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../fusemap'))

project = 'FuseMap'
copyright = '2025, Yichun He'
author = 'Yichun He'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_gallery.load_style",]



# autodoc configuration
autodoc_typehints = "description"
autodoc_mock_imports = ["anndata",
                        "dgl",
                        # "hnswlib",
                        # "captum",
                        # "circlify",
                        "matplotlib",
                        # "networkx",
                        # "numba",
                        # "numcodecs",
                        "numpy",
                        # "obonet",
                        "pandas",
                        # "pegasusio",
                        # "pytorch_lightning",
                        "scanpy",
                        "scipy",
                        "seaborn",
                        # "tiledb",
                        "tqdm",
                        "torch",
                        # "zarr",
                        ]

# todo configuration
todo_include_todos = True


# nbsphinx configuration
nbsphinx_thumbnails = {
    'notebooks/1_spatial_integration_imaging' : '_static/test.png',
    'notebooks/2_spatial_integration_cross_tech': '_static/test.png',
    'notebooks/3_gene_spatial_imputation': '_static/test.png',
    'notebooks/4_map_new_dataset_customized': '_static/test.png',
    'notebooks/5_map_new_dataset_molCCF': '_static/test.png',
    'notebooks/6_cell_to_cell_interaction': '_static/test.png'

}


templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# If this is True, todo emits a warning for each TODO entries. The default is False.
todo_emit_warnings = True


# html_theme = 'sphinx_rtd_theme'

html_theme = "sphinx_book_theme"
# autodoc_class_signature = "separated"

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

# Custom CSS
html_css_files = ['custom.css']


# Output file base name for HTML help builder.
htmlhelp_basename = "fusemap-doc"



# Prevent DGL from actually loading
import sys
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = ['dgl', 'dgl.data', 'dgl.nn', 'dgl.function']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)