# P2
The entry point for this project is `pset2.ipynb`. It is a [Jupyter notebook](https://jupyter.org/) and we recommend installing [JupyterLab](https://jupyter.org/install.html) to run it. You can find a preview of the notebook [here](https://github.com/slab-cmu/11-711-fall-20-projects/blob/master/p2/pset2.ipynb). You will heavily use PyTorch in this project. `pytorch_tensors_tutorial.ipynb` briefly describes some basic operations on Tensors in PyTorch. If you are not familiar with PyTorch, make sure to go through them before starting the project.

**Due: Oct 16, 11:59 PM EST**

**Summary:** This project focuses on sequence labeling with Hidden Markov Models and Deep Learning models. The target domain is part-of-speech tagging on English and Norwegian from the Universal Dependencies dataset. You will:
- Do some basic preprocessing of the data
- Build a naive classifier that tags each word with its most common tag
- Implement a `Viterbi` Tagger using `Hidden Markov Model` in PyTorch
- Build a `Bi-LSTM` deep learning model using PyTorch
- Build a `Bi-LSTM_CRF` model using the above components (`Viterbi` and `Bi-LSTM`) 
- Implement techniques to improve your tagger

**Submission:** To submit this assignment, run the script `make-submission.sh`, and submit the tarball `pset2-submission.tgz` on Canvas.

**Late Policy:** Each student will be granted 5 late days to use over the duration of the semester. You can use a maximum of 3 late days on any one project. **Weekends and holidays are also counted as late days.** Late submissions are automatically considered as using late days. Using late days will not affect your grade. However, projects submitted late after all late days have been used will receive no credit. Be careful!

**Academic honesty:** Homework assignments are to be completed individually and **you should not share your code**. E.g., it is not allowed to fork the public Github repo and push your progress there since others may see it. Verbal collaboration on homework assignments is acceptable, as well as re-implementation of relevant algorithms from research papers, but everything you turn in must be your own work, and you must note the names of anyone you collaborated with on each problem and cite resources that you used to learn about the problem. Suspected violations of academic integrity rules will be handled in accordance with the [CMU guidelines on collaboration and cheating](https://www.cmu.edu/policies/student-and-student-life/academic-integrity.html).
