# Orange3 Argument Mining Add-on

[![github build badge](https://github.com/EyeofBeholder-NLeSC/orange3-argument/actions/workflows/build.yml/badge.svg?branch=dev)](https://github.com/EyeofBeholder-NLeSC/orange3-argument/actions/workflows/build.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=EyeofBeholder-NLeSC_orange3-argument&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=EyeofBeholder-NLeSC_orange3-argument)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=EyeofBeholder-NLeSC_orange3-argument&metric=coverage)](https://sonarcloud.io/summary/new_code?id=EyeofBeholder-NLeSC_orange3-argument)
[![read the docs badge](https://readthedocs.org/projects/pip/badge/)](https://orange3-argument.readthedocs.io/en/latest/)
[![code style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![supported python versions](https://img.shields.io/badge/Python-3.8_%7C_3.9_%7C_3.10_%7C_3.11-blue)](./setup.cfg)

![image](./screenshot.png)

This is a Python package that implement a selection of argument mining techniques for argument classification and attacking relationship visualization. Modules are wrapped up with GUIs implemented on [Orange3](https://orangedatamining.com/), a powerful open-source platform to perform data analysis and visualization.  


## Installation

To install, first navigate to the project folder in terminal. We recommand you to create a virtual environment and install everything there. You can choose whatever tool you prefer to do so.

After activating your newly created virtual environment, you can install the add-on, together with Orange3 and all the other dependencies by running

```
pip install -e .
```

This will register the add-on but keep the code in the development directory (will not copy it to Python's site-packages directory).


## Usage

After the installation, the widget from this add-on is registered with Orange. To run Orange from the terminal,
use

```
python -m Orange.canvas
```

This will also allow you to see what's going on in the background from terminal.

A demo workflow together with a sample dataset are provided alongside this codebase.

After loading the workflow, you should be able to see the Orange interface like this:


## Credits

This package was created with the [Orange3 Example Add-on](https://github.com/biolab/orange3-example-addon).