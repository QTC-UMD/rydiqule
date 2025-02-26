from pathlib import Path

root = Path(__file__).parent.parent.parent
if (root / '.git').is_dir():
    # if .git exists, we are using an editable install
    try:
        import setuptools_scm
        import shutil
        if shutil.which('git') is None:
            # if git not on path somehow, fall back to importlib instead of failure
            scm_version = None
        else:
            scm_version = setuptools_scm.get_version(root,
                                                    version_scheme='release-branch-semver',
                                                    local_scheme='node-and-date')

    except ImportError:
        # if setuptools_scm not installed, fall back on importlib instead
        scm_version = None

else:
    scm_version = None

if scm_version is not None:
    # if introspecting a local version worked, use it
    __version__ = scm_version
else:
    # use the hard-coded version of the installed package
    import importlib.metadata
    try:
        assert __package__ is not None
        __version__ = importlib.metadata.version(__package__)
    except importlib.metadata.PackageNotFoundError:
        # this really shouldn't ever happen
        __version__ = None
