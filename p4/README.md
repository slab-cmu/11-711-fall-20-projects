# P4
The entry point for this project is `pset4.ipynb`. It is a [Jupyter notebook](https://jupyter.org/) and we recommend installing [JupyterLab](https://jupyter.org/install.html) to run it. You can find a preview of the notebook [here](https://github.com/slab-cmu/11-711-fall-20-projects/blob/master/p4/pset4.ipynb).

**Due: Dec 11, 11:59 PM EST**

**Summary:** In this problem set, you will venture into the challenging NLP task of **coreference resolution**. You will:
- Implement a simple rule-based system that achieve results which are surprisingly difficult to beat.
- Get acquainted with the trickiness of evaluating coref systems, and the current solutions in the field.
- Experiment with two neural approaches for coref to be implemented in PyTorch:
  * A feedforward network that only looks at boolean mention-pair features
  * A fully-neural architecture with embeddings all the way down
- Get a glimpse at domain adaptation in the wild, by trying to run a system trained on news against a narrative corpus and vice-versa.

**Submission:** To submit this assignment, run the script `make-submission.sh`, and submit the tarball `pset4-submission.tgz` on Canvas.

**Late Policy:** Each student will be granted 5 late days to use over the duration of the semester. You can use a maximum of 3 late days on any one project. **Weekends and holidays are also counted as late days.** Late submissions are automatically considered as using late days. Using late days will not affect your grade. However, projects submitted late after all late days have been used will receive no credit. Be careful!

**Academic honesty:** Homework assignments are to be completed individually and **you should not share your code**. E.g., it is not allowed to fork the public Github repo and push your progress there since others may see it. Verbal collaboration on homework assignments is acceptable, as well as re-implementation of relevant algorithms from research papers, but everything you turn in must be your own work, and you must note the names of anyone you collaborated with on each problem and cite resources that you used to learn about the problem. Suspected violations of academic integrity rules will be handled in accordance with the [CMU guidelines on collaboration and cheating](https://www.cmu.edu/policies/student-and-student-life/academic-integrity.html).