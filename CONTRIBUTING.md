# Contributing to SmartNotes

Thank you for your interest in contributing to SmartNotes! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and professional in all interactions. We aim to maintain a welcoming and inclusive community.

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/Kanike-tea/SmartNotes.git
cd SmartNotes
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 4. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Guidelines

### Code Style

- **Language**: Python 3.8+
- **Formatter**: Black (100 character line length)
- **Linter**: Flake8
- **Type Hints**: Required for all functions
- **Docstrings**: Google-style docstrings for all public functions

### Before Committing

```bash
# Format code
black src/ preprocessing/ tests/

# Check linting
flake8 src/ preprocessing/ tests/

# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=html
```

### Type Checking

```bash
mypy src/ preprocessing/
```

## Code Organization

### Module Structure

```
smartnotes/
├── src/
│   ├── model/          # Neural network architectures
│   ├── training/       # Training scripts
│   ├── inference/      # Inference scripts
│   └── dataloader/     # Data loading and preprocessing
├── preprocessing/      # Data preprocessing pipeline
├── tests/              # Test suite
├── config.py          # Centralized configuration
└── utils.py           # Utility functions
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `CRNN`, `TextTokenizer`)
- **Functions**: snake_case (e.g., `clean_text`, `calculate_cer`)
- **Constants**: UPPER_CASE (e.g., `ALLOWED_CHARS`)
- **Private**: Prefix with `_` (e.g., `_load_iam`)

## Writing Tests

All new features should include tests. Place tests in the `tests/` directory.

### Test Structure

```python
class TestFeature:
    """Tests for new feature."""
    
    def test_basic_functionality(self):
        """Test basic behavior."""
        # Arrange
        data = prepare_test_data()
        
        # Act
        result = function_under_test(data)
        
        # Assert
        assert result == expected_value
    
    def test_error_handling(self):
        """Test error cases."""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run specific test
pytest tests/test_model.py::TestModel::test_forward_pass -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Documentation

### Docstring Format

```python
def function(arg1: str, arg2: int) -> bool:
    """
    Short description of function.
    
    Longer description if needed. Explain what the function does,
    its use cases, and any important behavior.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
        RuntimeError: When operation fails
        
    Example:
        >>> result = function("test", 42)
        >>> print(result)
        True
    """
    pass
```

### Inline Comments

```python
# Good: Explains why, not what
if learning_rate < 1e-6:
    # Skip very small learning rates as they won't converge
    continue

# Bad: Obvious from code
if x == 1:  # Set x to 1
    x = 1
```

## Submitting Changes

### 1. Commit Messages

Use clear, descriptive commit messages:

```
feat: Add language model support to inference engine

- Implement pyctcdecode integration
- Add configuration options for LM decoding
- Update inference tests

Fixes #123
```

### Commit Message Format

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `perf:` Performance improvement
- `chore:` Build, dependencies, etc.

### 2. Create Pull Request

1. Push your branch: `git push origin feature/your-feature-name`
2. Go to GitHub and create a Pull Request
3. Fill in the PR template with:
   - Description of changes
   - Type of change (feature/bug fix/etc.)
   - Related issues
   - Testing performed

### 3. Code Review

- Address reviewer comments
- Keep commits clean and organized
- Don't force push after review starts (unless requested)

## Areas for Contribution

### High Priority

- [ ] Improve documentation and examples
- [ ] Add more comprehensive tests
- [ ] Performance optimizations
- [ ] Support for more datasets
- [ ] Better error messages and validation

### Medium Priority

- [ ] Additional evaluation metrics (CER, WER, confidence scores)
- [ ] Model quantization and pruning
- [ ] ONNX export support
- [ ] Web API/REST endpoints
- [ ] Interactive visualizations

### Nice to Have

- [ ] Multi-language support
- [ ] Attention mechanisms
- [ ] Transformer-based models
- [ ] Mobile deployment support
- [ ] Real-time processing pipeline

## Setting Up Your Development Environment

### VS Code Configuration

Create `.vscode/settings.json`:

```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": ["--max-line-length=100"],
    "[python]": {
        "editor.defaultFormatter": "ms-python.python",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

### Pre-commit Hooks

Install pre-commit to automatically format and lint:

```bash
pip install pre-commit
pre-commit install
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

## Release Process

1. Update version in `setup.py` and `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create GitHub release with tag
4. Build and publish package

## Questions or Need Help?

- Create an issue for bugs or features
- Check existing issues for similar questions
- Reach out to maintainers directly

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to SmartNotes! 🎉
