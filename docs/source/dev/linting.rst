Linting
=======

We use the `ruff` linting package to help enforce code style and consistent readability.
It can be run locally from the project root folder by calling the command `ruff check .`.
It pulls options for running the command from the `ruff.toml` configuration file.

If you intend to work on the rydiqule codebase,
it is good practice to incorporate automatic linting within your code editor.

Linting can be disabled for a single line by using the `# noqa` tag at the end of the line.
Specific error codes can be specified by `# noqa: E501`, which ignores only line length errors.