# Banqi (Chinese Chess) vs AI

A Python implementation of Banqi (Chinese Dark Chess) with an AI opponent. Play the classic Chinese board game against an intelligent computer opponent with multiple difficulty levels.

[![Banqi Game Screenshot](screenshot.png)](https://github.com/lee-cheng-han/Banqi-Chinese-Chess-vs-AI/blob/31bb3bfbd97ec5b1079ab14ec913d312081c70cb/demo.png)

## Features

- **Classic Banqi Gameplay**: Authentic Chinese Dark Chess rules and mechanics
- **AI Opponent**: Play against an intelligent AI with two difficulty levels
- **Modern GUI**: Clean, intuitive interface with circular pieces
- **Dynamic Color Assignment**: Player gets whatever color they reveal first
- **Strategic Depth**: Includes all traditional Banqi rules including special cannon mechanics
- **Real-time Feedback**: Clear status messages and move validation

## Game Rules

Banqi is played on a 4×8 board with 32 pieces (16 per player). All pieces start face-down and players take turns either revealing pieces or moving already revealed pieces.

### Piece Hierarchy (High to Low)
- **General (帥/將)** - Highest rank, but cannot capture Soldiers
- **Advisor (仕/士)** - High-ranking defensive piece
- **Elephant (相/象)** - Strong defensive piece
- **Chariot (俥/車)** - Powerful attacking piece
- **Horse (傌/馬)** - Mobile attacking piece
- **Cannon (炮/砲)** - Special movement and capture rules
- **Soldier (兵/卒)** - Lowest rank, but can capture the General

### Special Rules
- **Soldiers can capture Generals** (only exception to hierarchy)
- **Cannons** can move one square orthogonally OR jump over exactly one piece to capture
- **Movement** is strictly orthogonal (no diagonal moves)
- **Win condition**: Opponent cannot make any legal moves

## Installation

### Prerequisites
- Python 3.7 or higher
- tkinter (usually included with Python)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/banqi-ai.git
cd banqi-ai
```

2. Install dependencies (if needed):
```bash
pip install -r requirements.txt
```

3. Run the game:
```bash
python banqi.py
```

## How to Play

1. **Start**: Click "New Game" to begin
2. **First Move**: Click any face-down piece to reveal it - this determines your color
3. **Moving**: Click on your revealed pieces to select them, then click where you want to move
4. **Capturing**: Move to an enemy piece's position to capture (if rules allow)
5. **Winning**: Eliminate the opponent's ability to make legal moves

### Controls
- **Left Click**: Select pieces and make moves
- **New Game**: Start a fresh game
- **AI Difficulty**: Choose between Easy and Hard AI
- **Rules**: View detailed game rules
- **Quit**: Exit the application

## AI Features

### Easy Mode
- Uses greedy strategy focusing on immediate captures
- Good for beginners learning the game

### Hard Mode
- Implements minimax algorithm with 3-ply search depth
- Evaluates board position, piece values, and tactical threats
- Provides challenging gameplay for experienced players

## Technical Details

### Architecture
- **Object-Oriented Design**: Clean separation of game logic, AI, and GUI
- **Modular Components**: Separate classes for pieces, game state, and AI player
- **Efficient AI**: Lightweight game state copying for fast move evaluation

### Key Classes
- `Piece`: Represents individual game pieces with type, color, and state
- `BanqiGame`: Main game controller handling rules and GUI
- `AIPlayer`: Intelligent opponent with configurable difficulty
- `GameStateData`: Lightweight state representation for AI calculations

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Feature enhancements
- AI improvements
- UI/UX improvements
- Documentation updates

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Traditional Chinese Dark Chess (Banqi) rules and gameplay
- Chinese character representations for authentic piece display
- Minimax algorithm implementation for AI opponent

## Screenshots

### Game Interface
The game features a circular board layout with traditional Chinese characters for pieces:
- Red pieces: 帥 仕 相 俥 傌 兵 炮
- Black pieces: 將 士 象 車 馬 卒 砲

### AI Difficulty Levels
- **Easy**: Perfect for learning the game mechanics
- **Hard**: Challenging opponent for competitive play

## Future Enhancements

- [ ] Online multiplayer support
- [ ] Game replay system
- [ ] Advanced AI with opening book
- [ ] Tournament mode
- [ ] Save/load game functionality
- [ ] Sound effects and animations
- [ ] Multiple board themes

