# BMT-2022-MCG-Quality-Classes

The official repository for the paper _Towards Analytically Computable Quality Classes
for MCG Sensor Systems_ (https://doi.org/10.1515/cdbme-2022-1176).

**What you will find here:**

| Folder                | Description                                                             |
| --------------------- | ----------------------------------------------------------------------- |
| [figures](./figures/) | All signals pairs evaluated by the cardiologists.                       |
| [data](./data/)       | The expert evaulation, as well as the SNR and ASC for each signal pair. |
| [code](./code/)       | The code used for signal generation and the evaluation of the survey.   |

## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
- [Built With](#built-with)
- [Authors](#authors)
- [License](#license)

## Getting Started

These instructions will get you a copy of the application up and running on your
local machine for development and testing purposes.

### Prerequisites

This project requires the following to run:

- Python >= 3.6
- Jupyter

### Installation

Clone project with

```console
git clone https://github.com/CRC1261/BMT-2022-MCG-Quality-Classes.git
```

Install requirements with

```console
pip install -r code/requirements.txt
```

## Usage

You can run the signal generation with

```console
jupyter notebook
202202211353 - Exp - MCG Quality Classes Survey - Signal Generation.ipynb
```

and the evaluation with

```console
jupyter notebook
202204101251 - Exp - MCG Quality Classes Survey - Evaluation.ipynb
```

## Built With

- [Jupyter](https://jupyter.org/)

## Authors

- **Erik Engelhardt** - _Initial work_

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.
