# Orange3 Argument Mining Add-on

This is the Orange3 argument mining add-on for [the Eye of the Beholder project](https://research-software-directory.org/projects/the-eye-of-the-beholder). 


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

A demo workflow together with a sample dataset are provided alongside this codebase:
- demo workflow: https://github.com/EyeofBeholder-NLeSC/orange3-argument/blob/main/example/workflows/demo_workflow.ows
- sample dataset: https://raw.githubusercontent.com/EyeofBeholder-NLeSC/orange3-argument/main/example/data/data_processed_1prod_sample.json

After loading the workflow, you should be able to see the Orange interface like this:

![image](https://user-images.githubusercontent.com/92043159/218071751-2da27971-625f-409a-9d0b-1fc7452af9ba.png)
