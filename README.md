# Setups:

This project consists of two primary components. The first and main component is implemented in C++ using the deal.II libraries, which includes the computational model for running simulations and generating results. The second component is written in Python3, offering a modern, user-friendly interface to adjust model parameters and visualize the results.

**Note:** The current version of this project is optimized for macOS, supporting both Intel and Apple Silicon processors. If you'd like to run it on Linux or Windows, please refer to the respective branches of the project.

Below, you'll find a step-by-step guide for setting up, installing, and configuring the project. If you don't plan to use the second component (the user interface), you may skip directly to the "Configuration without Python3" section.

**Note:** This project is based on the model introduced in the paper [[Multi-field_2024]].

## Setup and Installation

The following packages and libraries must be installed before running the project:

- **Xcode**  
	Install Xcode from the App Store. You may also need to install command-line tools by running:
	
	```
	xcode-select --install   
	```
	
	You can verify the installation with `xcode-select --version`, which should return `xcode-select version 2396`. If not, the installation was unsuccessful.

- **CMake**  
	Install CMake by first installing [Homebrew](https://brew.sh/). Open a new terminal and run:
	
	 ```
	/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
	 ```
	
	Then install CMake:
	
	```
	   brew install cmake
	```
	
	
	Verify the installation with `cmake --version`, which should return `cmake version 3.24.1`. If not, the installation failed.

- **Compiler and MPI**  
    Install these with Homebrew:
    
    ```
    brew install cmake open-mpi gcc@11
    ```
    
- **Python3**  
    Install Python3 using Homebrew:
    
    ```
    brew install python@3.10
    ```
    
    Verify the installation with `which python3`, which should return `/opt/homebrew/bin/python3`.
    
    Then, set up the Tkinter package:
    
    ```
    brew install python-tk@3.10
    ```
    
    Ensure PIP is installed for managing Python3 packages. Check with `pip --version`. If PIP is not installed, follow the instructions [here](https://www.groovypost.com/howto/install-pip-on-a-mac/#:~:text=To%20install%20PIP%20using%20ensurepip,instructions%20to%20complete%20this%20process.).
    
- **deal.II**  
    To install and set up the [deal.II](https://www.dealii.org/) library:
    
    1. Open the terminal and type `clang` to trigger installation of command-line tools.
    2. Download the deal.II library:
    
    ```
    git clone https://github.com/dealii/candi.git
    ```
    
    3. Navigate to the downloaded folder: `cd candi`.
    4. Set the environment variables:
    
    ```
    export OMPI_FC=gfortran-11; export OMPI_CC=clang; export OMPI_CXX=clang++
    ```
    
    5. Install deal.II:
    
    ```
    ./candi.sh --packages="dealii"
    ```
    
    6. Follow on-screen instructions (you can cancel the process by pressing `Ctrl+C`).

If you encounter issues, please check the [deal.II installation guide](https://github.com/dealii/candi) or the [MacOSX wiki](https://github.com/dealii/dealii/wiki/MacOSX).

**Note:** We recommend using deal.II version 9.4 to avoid compatibility issues with other versions.

## Configuration

To download and configure the BFSimulator project:

1. Navigate to your desired download directory and run:

```
	git clone https://github.com/SaeedZarzor/BFSimulator.git
```

2. Enter the project folder:

	`cd BFSimulator`

3. Install necessary Python3 packages:

```
	pip install -r requirements.txt
```

4. To make Python files executable, find the Python3 path with:

	`which python3`
	
	Copy the path and paste it into the first line of `BFSimulator.py` after `#!`.

5. Check file permissions with:

	`ls -lh BFSimulator.py`
	
	If the response shows `-rwx------@`, the file is executable. If not, use:
	
	`chmod 700 BFSimulator.py`
	
	Repeat this step for `save.py`, `make_run.py`, and `progress.py`.
	
	To run the project, use:
	
```
	./BFSimulator.py
```

## Configuration without Python3

If you'd like to run only the first part of the project, follow these steps:

1. Ensure that Xcode with command-line tools, CMake, compiler and MPI, and deal.II are installed (see above). Paraview is also recommended for visualizing results.
    
2. Download the project:
    
```
	git clone https://github.com/SaeedZarzor/BFSimulator.git
```

3. Navigate to the project folder:

	`cd BFSimulator`

4. Generate the Makefile with:

```
	cmake CMakeLists.txt
```

5. Run the following command:

	`make`

To start the simulation:

```
./Brain_growth Parameters.prm 2
```

For 3D simulations, replace `2` with `3`. Modify simulation parameters directly in the `Parameters.prm` file.
