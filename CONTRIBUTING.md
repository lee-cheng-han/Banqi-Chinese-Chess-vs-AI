# Contributing to Banqi AI Game

Thank you for your interest in contributing to the Banqi AI Game! This document provides guidelines and information for contributors.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion for improvement:

1. **Check existing issues** first to avoid duplicates
2. **Create a new issue** with a clear title and description
3. **Include steps to reproduce** the problem if it's a bug
4. **Add screenshots** if they help explain the issue
5. **Specify your environment** (Python version, OS, etc.)

### Submitting Changes

1. **Fork the repository** to your GitHub account
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** with clear, focused commits
4. **Test thoroughly** to ensure your changes work correctly
5. **Update documentation** if needed
6. **Submit a pull request** with a clear description

## Development Guidelines

### Code Style

- Follow **PEP 8** Python style guidelines
- Use **meaningful variable and function names**
- Add **docstrings** for classes and functions
- Keep **line length under 100 characters**
- Use **type hints** where appropriate

### Code Organization

- **Separate concerns**: Keep game logic, AI, and GUI in distinct areas
- **Use enums** for constants (like piece types, colors)
- **Handle edge cases** and invalid inputs gracefully
- **Add comments** for complex algorithms or game rules

### Testing

Before submitting your changes:

1. **Test the basic game flow**:
   - Start new games
   - Reveal pieces correctly
   - Move pieces according to rules
   - AI responds appropriately

2. **Test edge cases**:
   - Invalid moves
   - Game over conditions
   - AI with no valid moves
   - All pieces revealed scenarios

3. **Test different scenarios**:
   - Both AI difficulty levels
   - Different piece combinations
   - Various board states

### Areas for Contribution

#### 🐛 Bug Fixes
- Game rule implementation errors
- AI decision-making issues
- GUI display problems
- Edge case handling

#### ✨ Features
- **Game Enhancements**:
  - Save/load game functionality
  - Game replay system
  - Move history tracking
  - Undo/redo functionality

- **AI Improvements**:
  - Opening book for early game
  - Advanced evaluation functions
  - Dynamic difficulty adjustment
  - Performance optimizations

- **UI/UX Enhancements**:
  - Animations for piece movements
  - Sound effects
  - Alternative board themes
  - Keyboard shortcuts
  - Improved accessibility

- **New Game Modes**:
  - Puzzle mode with preset positions
  - Tournament bracket system
  - Time controls
  - Online multiplayer foundation

#### 📚 Documentation
- Code documentation improvements
- Tutorial for new players
- Strategy guides
- API documentation for extensibility

#### 🧪 Testing
- Unit tests for game logic
- AI behavior tests
- GUI interaction tests
- Performance benchmarks

## Technical Details

### Project Structure
```
banqi-ai/
├── banqi.py              # Main game file
├── README.md             # Project documentation
├── LICENSE               # MIT license
├── requirements.txt      # Dependencies
├── .gitignore           # Git ignore rules
├── CONTRIBUTING.md      # This file
└── tests/               # Test files (future)
```

### Key Classes and Their Responsibilities

- **`Piece`**: Represents individual game pieces
  - Piece type, color, revealed state
  - Ranking system and capture rules
  - String representation with Chinese characters

- **`BanqiGame`**: Main game controller
  - Board state management
  - Move validation and execution
  - GUI event handling
  - Game flow control

- **`AIPlayer`**: Artificial intelligence opponent
  - Move generation and evaluation
  - Minimax algorithm implementation
  - Difficulty level handling
  - Board position assessment

- **`GameStateData`**: Lightweight state representation
  - Used for AI calculations
  - Enables efficient game tree search
  - Isolated from GUI concerns

### AI Implementation Notes

The AI uses a **minimax algorithm** with the following features:
- **Evaluation function** considering piece values and positions
- **Move ordering** prioritizing captures
- **Configurable search depth** based on difficulty
- **Efficient state copying** for game tree exploration

### GUI Implementation

The interface uses **tkinter** with:
- **Canvas-based board** with circular pieces
- **Event-driven interaction** for piece selection
- **Dynamic color assignment** based on first reveal
- **Real-time status updates** and move validation

## Getting Started with Development

1. **Set up your environment**:
   ```bash
   git clone https://github.com/yourusername/banqi-ai.git
   cd banqi-ai
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Run the game** to understand current functionality:
   ```bash
   python banqi.py
   ```

3. **Explore the code** structure and identify areas for improvement

4. **Make small changes first** to get familiar with the codebase

5. **Test thoroughly** before submitting pull requests

## Questions and Support

- **Open an issue** for questions about the codebase
- **Check existing issues** for similar questions
- **Be specific** about what you're trying to achieve
- **Include code snippets** when asking about implementation details

## Recognition

Contributors will be recognized in the project README and release notes. Significant contributions may earn maintainer status.

Thank you for helping make the Banqi AI Game better! 🎯
