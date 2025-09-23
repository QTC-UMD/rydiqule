Contributing
============

We actively encourage contributions and collaboration in the development of rydiqule.
Unforunately, rydiqule's development is done privately, *for reasons*.
If you would like to submit a PR for all but the most trivial of changes,
please e-mail us directly (david.h.meyer3.civ@army.mil or kevin.c.cox29.civ@army.mil)
so we can discuss the collaboration.

Tips for a successful contribution
++++++++++++++++++++++++++++++++++

If you would like to submit a pull request that improves or fixes rydiqule,
first off, thank you!
A wider set of contributors to a project significantly improves the quality
of the project as well as its pace of development.

Secondly, because of limitations placed on rydiqule's primary developers,
development deviates from a typical open-source project in that day-to-day work is done privately,
with bulk public code releases.
The most important difference for a potential contributor is that early communication
with the core developers is even more important so we can advise how to move forward.

Finally, writing code, especially a PR to someone else's project,
is similar to writing a journal article for peer-review.
The (code) reviewer needs to understand what you have done and why.
Ultimately you need to *persuade* them to accept.
Below is a general list of tips for producing a PR that can be merged quickly.

Ask *before* doing a lot of work
--------------------------------

Unlike paper reviews, the code review process is not blind.
As such, and as is common to any open-source project, it is good form to reach out to the developers
*before* submitting a PR with significant changes, via an issue on github or a direct message to the devs.
This can save you a lot of time as it allows the devs
to provide high-level guidance on what approach is likely to work and/or be accepted.
It avoids duplicated effort if we are already working on a solution.
It avoids wasted effort on things we have already rejected as not viable.
It also provides an opportunity for us to directly collaborate and help.
Finally, it allows us to set expectations on scope,
such as clarifying when a unit test isn't necessary.

Make things digestible
----------------------

In much the same way a paper with multiple pages of non-stop equations can be hard to follow,
submitting a PR with 1000s of changed lines of code, especially without having reached out first,
makes the PR very difficult to digest.
And we will not accept code we do not understand.

The key to digestible PRs is breaking changes down into small, logical, digestible steps.

- PRs are most like sections of a paper.
  Ensure that each PR addresses a single task.
  For example, don't implement unrelated features in a single PR,
  or undertake large refactors alongside a new feature.
  It is OK (even expected) to have a PR depend on another to be merged first.
- Commits are most like paragraphs in a paper.
  You should tend to use many small commits within a PR,
  ensuring each commit has a single purpose and a sufficiently detailed commit message.
  For example, moving a file and editing its contents should be separate commits.
  Updating many inter-related type hints should be a single commit.
- Careful application of git to edit commit history to organize edits after the fact
  should be considered.
  After all, it is rare the the order of discovery matches the best order for communicating the result.
  But note that editing history *after* creating the PR leads to annoyance for
  others who have already checked out your changes.
- Finally, don't overthink it.
  If the idea can be communicated in a concise single page,
  it is generally best to keep it that way instead of fluffing out the manuscript.
  The same goes for PRs: learn to recognize time-wasting bloat and low-value perfection chasing.
  If the change is straight-forward, don't waste time checking every box or revising commit messages.

Follow style guides
-------------------

Submitting a PR that does not follow standard conventions
makes it much harder for the reviewer to follow.
Even if the underlying work is fundamentally correct,
the differences in (ultimately arbitrary) styles consumes the bulk of the review effort.

Please ensure that your code follows the general style guides and practices for rydiqule.
This ensures code review focuses on important things, rather than minor details like formatting etc.
Aspects include :doc:`code linting <linting>`, :doc:`docstrings and higher-level docs <docs>`,
and :doc:`type hinting <types>`.

Unit tests and examples
-----------------------

Submitting a PR that lacks context or is missing supporting justifications
is much akin to submitting an incomplete manuscript.
Even if the work represents a significant advancement,
the missing details make it hard for the reviewer to understand and therefore accept.

For code, these supporting elements are provided by clear tests and examples,
especially in the associated github issue or pull request descriptions.
We also expect unit tests and examples to be included into the repository, where appropriate.
Please consult :doc:`our unit test documentation <tests>` to see what conventions we use.
Example notebooks should largely follow the style of existing example notebooks in the documentation.
Those notebooks live in the `/docs/source/examples <https://github.com/QTC-UMD/rydiqule/tree/main/docs/source/examples>`_
and `/docs/source/intro_nbs <https://github.com/QTC-UMD/rydiqule/tree/main/docs/source/intro_nbs>`_ directories of the repository.

If you are fixing a bug, unless it is very trivial,
we will expect a unit test to accompany the fix that demonstrates the issue and proves the fix works.
Ideally, the unit test should be the first commit so it can easily be shown what the issue is.

If you are adding a new feature, appropriate unit testing demonstrating the new feature works will be expected.
Depending on the scope of the enhancement, an example added to an existing notebook
or more likely a new example notebook will also be expected.

Please be patient
-----------------

We try very hard to outpace Physical Review timelines.
However, like most open-source project developers/maintainers,
this is not our primary responsibility.
While we strive to at least acknowledge messages quickly,
we are busy and it may take time to respond.
If things seem stuck, don't be afraid to ping again.

We also strive to be direct in our communication.
Especially in code reviews, this can come across as impatient or unappreciative.
That is not our intention and we request you not take it personally.
We strive to hold code released as a part of rydiqule to a high standard,
as it is a general tool with a wide variety of users and use cases.
We have to ensure contributions don't have unintended consequences for others,
and that they are maintainable *by us* after you have moved on.

Please be prepared for any PR to have many comments, questions, and requested changes before being merged.
Niche enhancements that break usability for other applications are unlikely to be merged.
While such contributions are important for science,
we will likely direct them elsewhere (i.e. a fork, paper supplemental material).
