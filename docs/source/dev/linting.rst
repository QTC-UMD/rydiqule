Linting
=======

We use the `flake8` linting package to help enforce code style and consistent readability.
It can be run locally from the project root folder by calling the command `flake8`.
It pulls options for running the command from the `.flake8` configuration file.

If you intend to work on the rydiqule codebase,
it is good practice to incorporate automatic linting within your code editor.

Linting can be disabled for a single line by using the `# noqa` tag at the end of the line.