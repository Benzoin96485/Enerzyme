# Enerzyme Documentation

This directory contains the documentation for the Enerzyme package.

## Building the Documentation

To build the documentation, run:

```bash
cd docs
make html
```

The built documentation will be available in `_build/html/`.

## Documentation Structure

The documentation follows a hierarchical structure:

### Level 1: Main API Documentation (`api.rst`)
- Contains the top-level table of contents for all `enerzyme.*` modules
- Each module links to its dedicated documentation page

### Level 2: Module Documentation (`modules/*.rst`)
- Individual documentation files for each top-level module
- Shows the module's contents, classes, functions, and submodules
- Each submodule links to its own documentation page

### Level 3: Submodule Documentation (Auto-generated)
- Documentation for `enerzyme.*.*` submodules
- Generated automatically by Sphinx autosummary
- Uses custom templates for consistent formatting

## Custom Templates

The documentation uses custom Sphinx templates located in `_templates/`:

- `custom-module-template.rst`: Template for module documentation
- `custom-class-template.rst`: Template for class documentation  
- `custom-function-template.rst`: Template for function documentation

## Features

- **Hierarchical Navigation**: Easy navigation between different levels of the API
- **Auto-generated Content**: Uses Sphinx autosummary for automatic documentation generation
- **Consistent Formatting**: Custom templates ensure consistent appearance
- **Comprehensive Coverage**: Documents all public modules, classes, and functions

## Adding New Modules

To add documentation for a new module:

1. Create a new `.rst` file in the `modules/` directory
2. Add the module to `modules/index.rst`
3. Add the module to `api.rst` if it's a top-level module
4. Rebuild the documentation

The hierarchical structure will automatically handle the navigation and organization.

