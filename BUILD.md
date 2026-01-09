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
The resulting packages will be in `dist/`, `glowtracker-[version].tar.gz` and `glowtracker-[version]-py3-none-any.why`.
**Don't forget** to change the version accordingly. Following the convention of Major.Minor.Patch.

# Testing installing the package
We can test the package by install directly and locally form
```bash
python -m pip install dist/glowtracker-[version].tar.gz
```

# Running
Once the package is installed, the application can be started by
```bash
python -m glowtracker
```
or running the executable wrapper by simply type
```bash
glowtracker
```