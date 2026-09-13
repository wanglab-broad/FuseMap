import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../fusemap'))

project = 'FuseMap'
copyright = '2026, Yichun He'
author = 'Yichun He'
release = '1.1.3'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
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
    "sphinx_gallery.load_style",
    "sphinx_design",
    "sphinx_copybutton",
]

# autodoc / autosummary configuration
autodoc_typehints = "description"
autosummary_generate = True
autodoc_mock_imports = [
    "anndata", "dgl", "harmonypy", "igraph", "leidenalg", "matplotlib",
    "networkx", "numpy", "pandas", "scanpy", "scipy", "seaborn", "sklearn",
    "sparse", "torch", "tqdm", "umap",
]

# copybutton: strip prompts when copying code blocks
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
}
if os.environ.get("FUSEMAP_DOCS_OFFLINE") == "1":
    intersphinx_mapping = {}

# todo configuration
todo_include_todos = True

# nbsphinx configuration: never execute notebooks at build time
nbsphinx_execute = "never"

# Editorial artwork for every tutorial; notebook result figures stay in the notebooks.
nbsphinx_thumbnails = {
    'notebooks/1_spatial_integration_imaging': '_static/thumbs/art/1_spatial_integration_imaging-cellular.webp',
    'notebooks/2_spatial_integration_cross_tech': '_static/thumbs/art/2_spatial_integration_cross_tech-cellular.webp',
    'notebooks/7_cross_condition_integration': '_static/thumbs/art/7_cross_condition_integration-cellular.webp',
    'notebooks/9_embryo_cross_resolution': '_static/thumbs/art/9_embryo_cross_resolution-cellular.webp',
    'notebooks/4_map_new_dataset_customized': '_static/thumbs/art/4_map_new_dataset_customized-cellular.webp',
    'notebooks/5_map_new_dataset_molCCF': '_static/thumbs/art/5_map_new_dataset_molCCF-cellular.webp',
    'notebooks/12_cross_species_marmoset': '_static/thumbs/art/12_cross_species_marmoset-cellular.webp',
    'notebooks/3_gene_spatial_imputation': '_static/thumbs/art/3_gene_spatial_imputation-cellular.webp',
    'notebooks/6_cell_to_cell_interaction': '_static/thumbs/art/6_cell_to_cell_interaction-cellular.webp',
    'notebooks/11_subregion_discovery': '_static/thumbs/art/11_subregion_discovery-cellular.webp',
    'notebooks/10_gene_panel_selection': '_static/thumbs/art/10_gene_panel_selection-cellular.webp',
}

templates_path = []
source_suffix = ".rst"
master_doc = "index"
pygments_style = "sphinx"
todo_emit_warnings = True

html_theme = "sphinx_book_theme"
html_title = "FuseMap"
html_theme_options = {
    "repository_url": "https://github.com/wanglab-broad/FuseMap",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": False,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
}

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_static_path = ['_static']
html_css_files = ['custom.css']
htmlhelp_basename = "fusemap-doc"

# Prevent DGL from actually loading
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = ['dgl', 'dgl.data', 'dgl.nn', 'dgl.function']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
