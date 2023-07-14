Argument Processor
==================

<img src="./icons/OWArgProcessor.svg" width="100" height="100">

Read tabular data in JSON format as Orange table.

## Signals

**Inputs**

- (None)

**Outputs**:

- `Data`: Output data table

## Description

**JSON File Reader** reads contents of a JSON file from either a local path or an URL, and output the content as a able.

## Control

- `File`: File browser that helps finding the target file.
- `URL`: Edit box for inputing the URL to the target file.

## Example

This example reads a local JSON file and output its content as a Orange data table. The JSON file can be found [here](https://raw.githubusercontent.com/EyeofBeholder-NLeSC/orange3-argument/main/example/data/data_processed_1prod_sample.json). 

![image](./images/OWJSONReader.png)