when you respond to me, start with "Hi, Lejun:"

Always reason from first principles: identify the real objective, separate facts from assumptions, and trace problems to their root cause before editing. Prefer the smallest change that preserves system invariants and matches user intent.

# Project overview
The project is (should be) described in @README.md, @proposal.md, @_plans, and @_progress.
- README.md is the standard overview that prepares for the public;
- proposal.md is the research proposal / plan to outlines the research questions, methods, and expected outcomes;
    - if you need to revise the proposal, prompt the user for approval (with a reasonable justification);
- You **make plans** for a chunk of work, and store them under _plans/ with descriptive names + "_plan.md".
- You **track progress** in _progress/ with detailed notes on completed tasks, known issues, performance analysis, and next steps.
    - one progress file per plan, with the same name + "_progress.md" for easy reference.

You should keep these documents updated as your project evolves, ensuring that they reflect the current state of your research and development efforts.
After you've done some work, update them accordingly.
When the user prompts "check the logs", it refers to all the documents mentioned above, and you should review them to provide a comprehensive update on the project's status.

We provide source files of a list of reference projects under @ref/


# Environment
you should activate a conda environment with the following command:
```
conda activate AM_AlphaGoZero
```
you may also need to run:
```
pip install -e .
```

When working with torch-relevant implementation, keep in mind that users may have no GPU access.
Make sure the code can also run on CPU for quick testing and debugging.
