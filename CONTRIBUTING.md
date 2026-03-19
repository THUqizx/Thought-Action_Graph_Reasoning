# Contributing to TAG

Thank you for your interest in contributing to TAG! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Create a branch for your feature or bug fix
3. Make your changes
4. Test your changes
5. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/TAG_Open_Source_v2.git
cd TAG_Open_Source_v2

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Contribution Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Testing

- Add tests for new features
- Ensure existing tests pass
- Test with multiple datasets (WebQSP, CWQ, GrailQA)

### Documentation

- Update README.md for new features
- Add inline comments for complex logic
- Update this file with new contribution guidelines

### Commit Messages

Use conventional commit format:

```
type: description

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat: add support for GPT-4o-mini
fix: resolve issue with SPARQL queries
docs: update README with installation instructions
```

## Types of Contributions

### 1. Bug Fixes

- Open an issue describing the bug
- Create a fix with tests
- Reference the issue in your PR

### 2. New Features

- Open an issue to discuss the feature
- Implement the feature with tests
- Update documentation

### 3. Performance Improvements

- Profile the code to identify bottlenecks
- Implement optimizations with benchmarks
- Document performance improvements

### 4. Documentation

- Fix typos and unclear explanations
- Add examples for common use cases
- Improve code comments

## Pull Request Process

1. Update README.md with details of changes (if applicable)
2. Update documentation in docstrings
3. Add tests for new functionality
4. Ensure all tests pass
5. Request review from maintainers

## Code Review

- All PRs must be reviewed by at least one maintainer
- Address review comments promptly
- Maintain code quality and style

## Questions?

- Open an issue for questions
- Check existing issues and documentation
- Join discussions in PRs

## Acknowledgments

We appreciate all contributions! Contributors will be acknowledged in the project README.
