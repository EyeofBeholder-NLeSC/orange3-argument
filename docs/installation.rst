Installation
============

Preparation
-----------

To install this package, we assume that you have Python installed on your computer. However, if that is not the case, we highly recommend that you first consult the `installation guides of Python <https://docs.python-guide.org/starting/installation/>`_. You should install Python 3.8 or higher versions to use this package. Additionally, while it's not necessary to be familiar with shell commands, if you're interested, you can explore this helpful `list of commonly used shell commands <https://guide.esciencecenter.nl/#/best_practices/language_guides/bash?id=commonly-used-command-line-tools>`_.

Once you have Python installed, open the terminal on your computer:

* **Windows**: If you runs Windows 11 on your computer, press the ``Win`` key, search for "PowerShell" and then open it. In case of Windows 10, you need to first download it from the `Microsoft Store <https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701>`_.

* **Linux**: You can press the ``Ctrl + Alt + T`` key to fire up the terminal.

* **MacOS**: Click the Launchpad icon in the Doc, or press the ``Cmd + Space`` type "Terminal" in the search field, then click Terminal.


Installation
------------

To install, we recommend to first navigate to your working directory by running this command:

.. code-block:: console
    
    cd /path/to/your/working/directory

We recommend to install our package in a new virtual environment to avoid dependency conflicts, and we recommand to use `venv` to do this:

.. code-block:: console

    python -m venv venv

To activate the virtual environment just created, on Windows, run:

.. code-block:: console

    venv\Scripts\activate

And on Linux and MacOS, run:

.. code-block:: console

    source venv/bin/activate

Then, to install this package, run:

.. code-block:: console

    pip install orangearg