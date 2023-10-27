# design_of_multi_agent_systems_braess_paradox

We won't go into the details of the project here, please read the report for those \
todo: include report in the repository

### instalation
Dependencies should all be included in the provided conda environment.yml and requirements.txt

To install run \
`conda env create -f environment.yml` and activate with `conda activate dmas_braess` \
or \
`pip install -r requirements.txt` 

### experiments
experiments.ipynb was used to create all data, plots and tables presented in the report. \
It is largely documented so you can follow along easily (for the most part, no promises!)

### gui
gui.py contains a simple GUI created in tKinter, which should be included in your python instalation. \
The website notes Tkinter is not included in the default distribution for OSX, although it seems to be when installing using `brew install python@10` \
The GUI wasn't really used during the project, but it can be helpful to understand what the project is about. \
Experiments where run for 200_000 steps, which takes around 5-7 minutes depending on your system. Use this information if you want to see how the model performs after letting the agents settle their strategies. \
note: during testing of the gui, buttons often don't respond to clicks very well. Resizing the window seems to help with this problem.
