# Building the package
Before building a package, **Don't forget** to change the version accordingly in the file `pyproject.toml` under section `[project]`. 
Follow the convention of Major.Minor.Patch.
Then Execute the build process by
```bash
python -m build
```
or
```bash
uv build
```
The resulting packages will be in `dist/` as `glowtracker-[version].tar.gz` and `glowtracker-[version]-py3-none-any.whl`.
**Don't forget** to change the version accordingly. Following the convention of Major.Minor.Patch.

# Testing installing the package
Test the package by installing the local source distribution:
```bash
python -m pip install dist/glowtracker-[version].tar.gz
```
or
```bash
uv pip install dist/glowtracker-[version].tar.gz
```
Add `--python 3.12` to select the recommended Python version.


# Running
Print something 
Once the package is installed, the application can be started by
```bash
python -m glowtracker
```
or running the executable wrapper by simply type
```bash
glowtracker
```
