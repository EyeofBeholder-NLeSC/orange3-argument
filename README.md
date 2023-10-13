# Orange3 Argument Mining Add-on

[![github build badge](https://github.com/EyeofBeholder-NLeSC/orange3-argument/actions/workflows/build.yml/badge.svg?branch=dev)](https://github.com/EyeofBeholder-NLeSC/orange3-argument/actions/workflows/build.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=EyeofBeholder-NLeSC_orange3-argument&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=EyeofBeholder-NLeSC_orange3-argument)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=EyeofBeholder-NLeSC_orange3-argument&metric=coverage)](https://sonarcloud.io/summary/new_code?id=EyeofBeholder-NLeSC_orange3-argument)
[![read the docs badge](https://readthedocs.org/projects/pip/badge/)](https://orange3-argument.readthedocs.io/en/latest/)
[![code style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/EyeofBeholder-NLeSC/orange3-argument/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/orangearg)](https://pypi.org/project/orangearg/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/orangearg)](https://pypi.org/project/orangearg/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8383636.svg)](https://doi.org/10.5281/zenodo.8383636)
[![Research Software Directory](https://img.shields.io/badge/RSD-Orange3_Argument-blue)](https://research-software-directory.org/software/orange3-argument-add-on)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)

<div align="center">
    <img src="https://github.com/EyeofBeholder-NLeSC/orange3-argument/blob/main/screenshot.png">
</div>

This work is an open-source Python package that implements a pipeline of processing, analyzing, and visualizing an argument corpus and the attacking relationship inside the corpus. It also implements the corresponding GUIs on a scientific workflow platform named [Orange3](https://orangedatamining.com/), so that users with little knowledge of Python programming can also benefit from it.

## Table of Contents

- [Why](#why)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Credits](#credits)

## Why

This package is designed with a clear mission: to empower researchers in building their own argument mining workflows effortlessly. Leveraging the capabilities of state-of-the-art, pre-trained language models for natural language processing, this tool facilitates the process of understanding arguments from text data. At its core, this work is committed to transparency and interpretability throughout the analysis process. We believe that clarity and comprehensibility are paramount when working with complex language data. As such, the tool not only automates the task but also ensures that the results are easily interpretable, allowing researchers to gain valuable insights from their data.

Please cite this work if you use it for scientific or commercial purpose.

<div align="center">
    <img src="https://github.com/EyeofBeholder-NLeSC/orange3-argument/blob/main/docs/_static/flowchart.png">
</div>

## Installation

This package requires Python version >= 3.8. We recommand installing this package in a new virtual environment to avoid dependency conflicts. The package can be installed from PyPI via `pip`:

```console
pip install orangearg
```

Executing the above command will install both the necessary dependencies and the graphical user interface (GUI) components of Orange3.

Further details can be found in the [installation guide](https://orange3-argument.readthedocs.io/en/latest/installation.html).


## Getting Started

If you would like to learn how to use this package for scripting, take a look at our example [notebook](./examples/example.ipynb).

To build and run workflows on Orange3, run the following command in your terminal to launch the Orange3 GUI, known as the 'canvas'.

```console
python -m Orange.canvas
```

A sample [workflow](./examples/demo_workflow.ows) and [dataset](./examples/example_dataset.json) have been provided to illustrate the effective utilization of this package within Orange3.

For additional information, please refer to the [guidance](https://orange3-argument.readthedocs.io/en/latest/widget_guis.html#) on using this package through widgets in Orange3.

## Documentation

The documentation of this work can be found on [Read the Docs](https://orange3-argument.readthedocs.io/en/latest/index.html).

## Contributing

If you want to contribute to the development of this work, have a look at the [contribution guidelines](./CONTRIBUTING.md).

## Credits

This work is being developed by the [Netherlands eScience Center](https://www.esciencecenter.nl/) in collaboration with the [Human Centered Data Analysis group](https://www.cwi.nl/en/groups/human-centered-data-analytics/) at Centrum Wiskunde & Informatica.

This package was created with the [Orange3 Example Add-on](https://github.com/biolab/orange3-example-addon).

<hr>

[Go to Top](#table-of-contents)