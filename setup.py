#!/usr/bin/env python

from os import path, walk, system
import atexit

import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

NAME = "Orange3-Argument"

VERSION = "0.1.0"

AUTHOR = 'Ji Qi, Netherlands eScience Center, NL'
AUTHOR_EMAIL = 'j.qi@esciencecenter.nl'

URL = 'https://github.com/EyeofBeholder-NLeSC/orange3-argument'
DESCRIPTION = "Argument mining and reasoning add-on for Orange3."
LONG_DESCRIPTION = open(path.join(
    path.dirname(__file__), 'README.pypi'), 'r', encoding='utf-8').read()

LICENSE = "Apache License 2.0"

KEYWORDS = (
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    'orange3 add-on',
    'argument mining', 
    'network visualization'
)

PACKAGES = find_packages()

PACKAGE_DATA = {
    'orangecontrib.argument.widgets': ['icons/*'],
}

DATA_FILES = [
    # Data files that will be installed outside site-packages folder
]

INSTALL_REQUIRES = [
    'AnyQt', 
    'pyqt6', 
    'pyqtgraph', 
    'gensim', 
    'networkx', 
    'numpy', 
    'Orange3', 
    'pandas', 
    'pot', 
    'scikit_learn', 
    'spacy_readability', 
    'pytextrank', 
    'flair'
    ]

ENTRY_POINTS = {
    # Entry points that marks this package as an orange add-on. If set, addon will
    # be shown in the add-ons manager even if not published on PyPi.
    'orange.addons': (
       'argument = orangecontrib.argument',
    ),
    # # Entry point used to specify packages containing tutorials accessible
    # # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    # 'orange.widgets.tutorials': (
    #     # Syntax: any_text = path.to.package.containing.tutorials
    #     'exampletutorials = orangecontrib.example.tutorials',
    # ),

    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/example/widgets/__init__.py
        'Argument = orangecontrib.argument.widgets',
    ),

    # # Register widget help
    # "orange.canvas.help": (
    #     'html-index = orangecontrib.example.widgets:WIDGET_HELP_PATH',)
}

NAMESPACE_PACKAGES = ["orangecontrib"]

# TEST_SUITE = "orangecontrib.example.tests.suite"


def include_documentation(local_dir, install_dir):
    global DATA_FILES
    if 'bdist_wheel' in sys.argv and not path.exists(local_dir):
        print("Directory '{}' does not exist. "
              "Please build documentation before running bdist_wheel."
              .format(path.abspath(local_dir)))
        sys.exit(0)

    doc_files = []
    for dirpath, dirs, files in walk(local_dir):
        doc_files.append((dirpath.replace(local_dir, install_dir),
                          [path.join(dirpath, f) for f in files]))
    DATA_FILES.extend(doc_files)

def _post_install():
    print('POST INSTALL: update spacy to the newest version...')
    system('pip install -U spacy pytextrank')

class PostInstallCommand(install):
    def __init__(self, *args, **kwargs):
        super(PostInstallCommand, self).__init__(*args, **kwargs)
        atexit.register(_post_install)

if __name__ == '__main__':
    # include_documentation('doc/_build/html', 'help/orange3-example')
    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        license=LICENSE,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=NAMESPACE_PACKAGES,
        zip_safe=False,
        cmdclass={'install': PostInstallCommand,}
    )
