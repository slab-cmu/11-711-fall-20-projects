# P3
The entry point for this project is `pset3.ipynb`. It is a [Jupyter notebook](https://jupyter.org/) and we recommend installing [JupyterLab](https://jupyter.org/install.html) to run it. You can find a preview of the notebook [here](https://github.com/slab-cmu/11-711-fall-20-projects/blob/master/p3/pset3.ipynb).

**Due: Nov 6, 11:59 PM EST**

**Summary:** In this problem set, you will implement a deep transition dependency parser in [PyTorch](https://pytorch.org). You will:
- Implement an arc-standard transition-based dependency parser in PyTorch
- Implement neural network components for choosing actions and combining stack elements
- Train your network to parse English and Norwegian sentences
- Implement techniques to improve your parser

**Submission:** To submit this assignment, run the script `make-submission.sh`, and submit the tarball `pset3-submission.tgz` on Canvas.

**Late Policy:** Each student will be granted 5 late days to use over the duration of the semester. You can use a maximum of 3 late days on any one project. **Weekends and holidays are also counted as late days.** Late submissions are automatically considered as using late days. Using late days will not affect your grade. However, projects submitted late after all late days have been used will receive no credit. Be careful!

**Academic honesty:** Homework assignments are to be completed individually and **you should not share your code**. E.g., it is not allowed to fork the public Github repo and push your progress there since others may see it. Verbal collaboration on homework assignments is acceptable, as well as re-implementation of relevant algorithms from research papers, but everything you turn in must be your own work, and you must note the names of anyone you collaborated with on each problem and cite resources that you used to learn about the problem. Suspected violations of academic integrity rules will be handled in accordance with the [CMU guidelines on collaboration and cheating](https://www.cmu.edu/policies/student-and-student-life/academic-integrity.html).