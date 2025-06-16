"""
Banqi (Chinese Dark Chess) Game with AI Opponent

A complete implementation of the traditional Chinese board game Banqi,
featuring a graphical user interface and AI opponent with adjustable difficulty.

Features:
- 4x8 board with 32 pieces (16 per side)
- All pieces start face-down and are revealed during play
- Orthogonal movement only (no diagonals)
- Special cannon jumping mechanics
- Minimax AI with tactical evaluation
- Circular piece display on wooden board theme

Author: Lee Cheng Han (Hank Lee)
Date: 2025-06-15
"""

import tkinter as tk
from tkinter import messagebox, ttk
import random
import time
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set
import copy


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class PieceType(Enum):
    """Enumeration of all piece types in Banqi"""
    GENERAL = "general"    # 帥/將 - Highest rank, but vulnerable to soldiers
    ADVISOR = "advisor"    # 仕/士 - High rank support piece
    ELEPHANT = "elephant"  # 相/象 - Medium-high rank piece
    CHARIOT = "chariot"    # 俥/車 - Medium rank, strong piece
    HORSE = "horse"        # 傌/馬 - Medium rank piece
    CANNON = "cannon"      # 炮/砲 - Special movement, can jump
    SOLDIER = "soldier"    # 兵/卒 - Lowest rank, but can capture general


class PieceColor(Enum):
    """Player colors in the game"""
    RED = "red"      # Traditional red player (帥 side)
    BLACK = "black"  # Traditional black player (將 side)


class GameState(Enum):
    """Current state of the game"""
    SELECTING_PIECE = 1  # Player is selecting a piece to move/reveal
    PIECE_SELECTED = 2   # Player has selected a piece and choosing destination
    GAME_OVER = 3        # Game has ended


# ============================================================================
# PIECE CLASS
# ============================================================================

class Piece:
    """
    Represents a single game piece with its type, color, and reveal status.
    
    Each piece has a rank for capture rules and strategic value for AI evaluation.
    """
    
    def __init__(self, piece_type: PieceType, color: PieceColor):
        """
        Initialize a new piece.
        
        Args:
            piece_type: The type of piece (general, advisor, etc.)
            color: The color/player this piece belongs to
        """
        self.piece_type = piece_type
        self.color = color
        self.is_revealed = False  # Pieces start face-down
    
    def __str__(self) -> str:
        """
        Return the Chinese character representation of this piece.
        
        Returns:
            Single Chinese character representing the piece
        """
        # Red pieces use traditional characters
        if self.color == PieceColor.RED:
            char_map = {
                PieceType.GENERAL: "帥",   # General (Commander)
                PieceType.ADVISOR: "仕",   # Advisor/Guard
                PieceType.ELEPHANT: "相",  # Elephant/Minister
                PieceType.CHARIOT: "俥",   # Chariot/Rook
                PieceType.HORSE: "傌",     # Horse/Knight
                PieceType.SOLDIER: "兵",   # Soldier/Pawn
                PieceType.CANNON: "炮"     # Cannon
            }
        else:  # BLACK pieces use alternate traditional characters
            char_map = {
                PieceType.GENERAL: "將",   # General (same meaning, different char)
                PieceType.ADVISOR: "士",   # Scholar/Advisor
                PieceType.ELEPHANT: "象",  # Elephant (different char)
                PieceType.CHARIOT: "車",   # Chariot (different char)
                PieceType.HORSE: "馬",     # Horse (different char)
                PieceType.SOLDIER: "卒",   # Soldier (different char)
                PieceType.CANNON: "砲"     # Cannon (different char)
            }
        return char_map[self.piece_type]
    
    def get_rank(self) -> int:
        """
        Get the hierarchical rank of this piece for capture rules.
        
        Returns:
            Integer rank (higher number = higher rank)
        """
        rank_map = {
            PieceType.GENERAL: 6,   # Highest rank
            PieceType.ADVISOR: 5,
            PieceType.ELEPHANT: 4,
            PieceType.CHARIOT: 3,
            PieceType.HORSE: 2,
            PieceType.CANNON: 1,
            PieceType.SOLDIER: 0    # Lowest rank (but can capture general)
        }
        return rank_map[self.piece_type]
    
    def get_value(self) -> int:
        """
        Get the strategic value of this piece for AI evaluation.
        
        Returns:
            Point value representing piece importance
        """
        value_map = {
            PieceType.GENERAL: 1000,  # Most valuable (losing = game over)
            PieceType.ADVISOR: 200,
            PieceType.ELEPHANT: 180,
            PieceType.CHARIOT: 160,
            PieceType.HORSE: 140,
            PieceType.CANNON: 120,
            PieceType.SOLDIER: 100    # Valuable for threatening general
        }
        return value_map[self.piece_type]
    
    def can_capture(self, target: 'Piece') -> bool:
        """
        Check if this piece can capture the target piece based on game rules.
        
        Args:
            target: The piece to potentially capture
            
        Returns:
            True if capture is allowed, False otherwise
        """
        # Both pieces must be revealed to interact
        if not target.is_revealed or not self.is_revealed:
            return False
        
        # Cannot capture own pieces
        if self.color == target.color:
            return False
        
        # Special rule: General cannot capture Soldiers
        if self.piece_type == PieceType.GENERAL and target.piece_type == PieceType.SOLDIER:
            return False
        
        # Special rule: Soldiers can capture General (David vs Goliath)
        if self.piece_type == PieceType.SOLDIER and target.piece_type == PieceType.GENERAL:
            return True
        
        # Cannons can capture any enemy piece (movement rules checked elsewhere)
        if self.piece_type == PieceType.CANNON:
            return True
        
        # Standard rule: can capture equal or lower rank pieces
        return self.get_rank() >= target.get_rank()


# ============================================================================
# GAME STATE DATA CLASS
# ============================================================================

class GameStateData:
    """
    Lightweight representation of game state for AI calculations.
    
    This class provides a snapshot of the board state without GUI dependencies,
    allowing the AI to perform deep searches efficiently.
    """
    
    def __init__(self, board, current_player, first_piece_revealed, game_over):
        """
        Initialize game state snapshot.
        
        Args:
            board: 2D array of pieces
            current_player: Current player's color
            first_piece_revealed: Whether first piece has been revealed
            game_over: Whether game has ended
        """
        self.board = board
        self.current_player = current_player
        self.first_piece_revealed = first_piece_revealed
        self.game_over = game_over
        self.board_height = 4  # Standard Banqi board dimensions
        self.board_width = 8


# ============================================================================
# AI PLAYER CLASS
# ============================================================================

class AIPlayer:
    """
    AI opponent using minimax algorithm with alpha-beta pruning concepts.
    
    The AI evaluates positions based on material balance, positional factors,
    and tactical opportunities. Difficulty affects search depth and strategy.
    """
    
    def __init__(self, color: PieceColor, difficulty: str = "hard"):
        """
        Initialize AI player.
        
        Args:
            color: AI's piece color
            difficulty: "easy" or "hard" - affects search depth and strategy
        """
        self.color = color
        self.difficulty = difficulty
        # Search deeper on hard difficulty for better play
        self.search_depth = 3 if difficulty == "hard" else 2
    
    def create_game_state(self, game) -> GameStateData:
        """
        Create a lightweight copy of the current game state for AI analysis.
        
        Args:
            game: The main game object
            
        Returns:
            GameStateData object for AI calculations
        """
        # Deep copy the board to avoid modifying the original
        board_copy = []
        for row in game.board:
            row_copy = []
            for piece in row:
                if piece is None:
                    row_copy.append(None)
                else:
                    # Create new piece with same properties
                    piece_copy = Piece(piece.piece_type, piece.color)
                    piece_copy.is_revealed = piece.is_revealed
                    row_copy.append(piece_copy)
            board_copy.append(row_copy)
        
        return GameStateData(
            board=board_copy,
            current_player=game.current_player,
            first_piece_revealed=game.first_piece_revealed,
            game_over=game.game_over
        )
    
    def evaluate_board(self, game_state: GameStateData) -> float:
        """
        Evaluate the current board position from AI's perspective.
        
        Considers:
        - Material balance (piece values)
        - Positional factors (center control, mobility)
        - Tactical opportunities (captures, threats)
        
        Args:
            game_state: Current position to evaluate
            
        Returns:
            Evaluation score (positive = good for AI, negative = bad for AI)
        """
        score = 0
        my_pieces = []
        enemy_pieces = []
        
        # Collect all revealed pieces
        for row in range(game_state.board_height):
            for col in range(game_state.board_width):
                piece = game_state.board[row][col]
                if piece and piece.is_revealed:
                    if piece.color == self.color:
                        my_pieces.append((piece, row, col))
                    else:
                        enemy_pieces.append((piece, row, col))
        
        # Material balance - most important factor
        for piece, _, _ in my_pieces:
            score += piece.get_value()
        for piece, _, _ in enemy_pieces:
            score -= piece.get_value()
        
        # Positional bonuses
        for piece, row, col in my_pieces:
            # Center control bonus - pieces in center are more flexible
            if 1 <= row <= 2 and 2 <= col <= 5:
                score += 20
            
            # Mobility bonus - pieces with more moves are more valuable
            moves = self.get_piece_moves(game_state, row, col)
            score += len(moves) * 5
        
        # Threat evaluation - value pieces we can capture
        for piece, row, col in my_pieces:
            captures = self.get_capture_moves(game_state, row, col)
            for tr, tc in captures:
                target = game_state.board[tr][tc]
                if target:
                    # Add fraction of target's value as threat bonus
                    score += target.get_value() * 0.3
        
        return score
    
    def get_piece_moves(self, game_state: GameStateData, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Get all valid moves for a piece at the given position.
        
        Args:
            game_state: Current game state
            row, col: Position of piece to check
            
        Returns:
            List of (row, col) tuples representing valid destination squares
        """
        moves = []
        piece = game_state.board[row][col]
        
        if not piece or not piece.is_revealed:
            return moves
        
        # Standard adjacent moves for all pieces (orthogonal only)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < game_state.board_height and 
                0 <= new_col < game_state.board_width):
                if self.is_valid_move(game_state, row, col, new_row, new_col):
                    moves.append((new_row, new_col))
        
        # Special cannon jump moves
        if piece.piece_type == PieceType.CANNON:
            for target_row in range(game_state.board_height):
                for target_col in range(game_state.board_width):
                    if (target_row != row or target_col != col):
                        if self.is_cannon_jump_move(game_state, row, col, target_row, target_col):
                            target = game_state.board[target_row][target_col]
                            # Can jump to capture enemies or empty squares after jumping
                            if target is None or (target.color != piece.color):
                                moves.append((target_row, target_col))
        
        return moves
    
    def get_capture_moves(self, game_state: GameStateData, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Get all valid capture moves for a piece.
        
        Args:
            game_state: Current game state
            row, col: Position of piece to check
            
        Returns:
            List of (row, col) tuples where piece can capture
        """
        captures = []
        piece = game_state.board[row][col]
        
        if not piece or not piece.is_revealed:
            return captures
        
        # Check all possible moves for capture opportunities
        moves = self.get_piece_moves(game_state, row, col)
        for mr, mc in moves:
            target = game_state.board[mr][mc]
            if target and target.color != piece.color and piece.can_capture(target):
                captures.append((mr, mc))
        
        return captures
    
    def get_all_moves(self, game_state: GameStateData) -> List[Tuple]:
        """
        Generate all possible moves for the AI player.
        
        Move types:
        - "capture": Capture an enemy piece
        - "move": Move to empty square
        - "reveal": Reveal a face-down piece
        
        Args:
            game_state: Current game state
            
        Returns:
            List of move tuples: (move_type, from_pos, to_pos)
        """
        moves = []
        
        # Capture moves (highest priority)
        for row in range(game_state.board_height):
            for col in range(game_state.board_width):
                piece = game_state.board[row][col]
                if piece and piece.is_revealed and piece.color == self.color:
                    captures = self.get_capture_moves(game_state, row, col)
                    for tr, tc in captures:
                        moves.append(("capture", (row, col), (tr, tc)))
        
        # Regular moves to empty squares
        for row in range(game_state.board_height):
            for col in range(game_state.board_width):
                piece = game_state.board[row][col]
                if piece and piece.is_revealed and piece.color == self.color:
                    piece_moves = self.get_piece_moves(game_state, row, col)
                    for mr, mc in piece_moves:
                        target = game_state.board[mr][mc]
                        if target is None:
                            moves.append(("move", (row, col), (mr, mc)))
        
        # Reveal moves (when no better options available)
        for row in range(game_state.board_height):
            for col in range(game_state.board_width):
                piece = game_state.board[row][col]
                if piece and not piece.is_revealed:
                    moves.append(("reveal", (row, col), None))
        
        return moves
    
    def is_cannon_jump_move(self, game_state: GameStateData, from_row: int, from_col: int, 
                           to_row: int, to_col: int) -> bool:
        """
        Check if a move is a valid cannon jump (over exactly one piece).
        
        Cannons can jump over exactly one piece in a straight line to capture
        or move to the square beyond.
        
        Args:
            game_state: Current game state
            from_row, from_col: Starting position
            to_row, to_col: Destination position
            
        Returns:
            True if this is a valid cannon jump move
        """
        selected_piece = game_state.board[from_row][from_col]
        if selected_piece is None or selected_piece.piece_type != PieceType.CANNON:
            return False
        
        # Must be in same row or column (straight line)
        if from_row != to_row and from_col != to_col:
            return False
        
        # Must have a target piece at destination for capture
        target_piece = game_state.board[to_row][to_col]
        if target_piece is None:
            return False
        
        # Count pieces between cannon and target
        pieces_between = 0
        
        if from_row == to_row:  # Horizontal jump
            start_col = min(from_col, to_col) + 1
            end_col = max(from_col, to_col)
            
            for col in range(start_col, end_col):
                if game_state.board[from_row][col] is not None:
                    pieces_between += 1
        else:  # Vertical jump
            start_row = min(from_row, to_row) + 1
            end_row = max(from_row, to_row)
            
            for row in range(start_row, end_row):
                if game_state.board[row][from_col] is not None:
                    pieces_between += 1
        
        # Must have exactly one piece to jump over
        return pieces_between == 1
    
    def is_orthogonal_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """
        Check if move is orthogonal (up/down/left/right only, no diagonals).
        
        Args:
            from_row, from_col: Starting position
            to_row, to_col: Destination position
            
        Returns:
            True if move is exactly one square orthogonally
        """
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        
        # Must move exactly one square in exactly one direction
        return (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)
    
    def is_valid_move(self, game_state: GameStateData, from_row: int, from_col: int, 
                     to_row: int, to_col: int) -> bool:
        """
        Check if a move is valid according to game rules.
        
        Args:
            game_state: Current game state
            from_row, from_col: Starting position
            to_row, to_col: Destination position
            
        Returns:
            True if move is legal
        """
        selected_piece = game_state.board[from_row][from_col]
        target = game_state.board[to_row][to_col]
        
        # Basic validation
        if selected_piece is None or not selected_piece.is_revealed:
            return False
        
        if selected_piece.color != game_state.current_player:
            return False
        
        # Check board boundaries
        if not (0 <= to_row < game_state.board_height and 
                0 <= to_col < game_state.board_width):
            return False
        
        # Special cannon movement rules
        if selected_piece.piece_type == PieceType.CANNON:
            # Cannons can move to adjacent empty squares like other pieces
            if target is None and self.is_orthogonal_move(from_row, from_col, to_row, to_col):
                return True
            # Cannons can also capture by jumping (if target exists)
            if target is not None:
                return self.is_cannon_jump_move(game_state, from_row, from_col, to_row, to_col)
            return False
        
        # For non-cannon pieces: must be orthogonal and adjacent
        if not self.is_orthogonal_move(from_row, from_col, to_row, to_col):
            return False
        
        # Can always move to empty adjacent squares or interact with adjacent pieces
        return True
    
    def minimax(self, game_state: GameStateData, depth: int, maximizing: bool) -> Tuple[float, Optional[Tuple]]:
        """
        Minimax algorithm for optimal move selection.
        
        Recursively evaluates future positions to find the best move.
        
        Args:
            game_state: Current position
            depth: How many moves ahead to search
            maximizing: True if maximizing AI score, False if minimizing
            
        Returns:
            Tuple of (best_score, best_move)
        """
        # Base case: reached search depth or game over
        if depth == 0 or game_state.game_over:
            return self.evaluate_board(game_state), None
        
        moves = self.get_all_moves(game_state)
        if not moves:
            return self.evaluate_board(game_state), None
        
        best_move = None
        
        if maximizing:
            # AI's turn - maximize evaluation
            max_eval = float('-inf')
            for move in moves:
                # Create copy and apply move
                game_copy = self.apply_move(self.create_game_state_copy(game_state), move)
                eval_score, _ = self.minimax(game_copy, depth - 1, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
            return max_eval, best_move
        else:
            # Opponent's turn - minimize evaluation
            min_eval = float('inf')
            for move in moves:
                game_copy = self.apply_move(self.create_game_state_copy(game_state), move)
                eval_score, _ = self.minimax(game_copy, depth - 1, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
            return min_eval, best_move
    
    def create_game_state_copy(self, game_state: GameStateData) -> GameStateData:
        """
        Create a deep copy of game state for move simulation.
        
        Args:
            game_state: State to copy
            
        Returns:
            Independent copy of the game state
        """
        board_copy = []
        for row in game_state.board:
            row_copy = []
            for piece in row:
                if piece is None:
                    row_copy.append(None)
                else:
                    piece_copy = Piece(piece.piece_type, piece.color)
                    piece_copy.is_revealed = piece.is_revealed
                    row_copy.append(piece_copy)
            board_copy.append(row_copy)
        
        return GameStateData(
            board=board_copy,
            current_player=game_state.current_player,
            first_piece_revealed=game_state.first_piece_revealed,
            game_over=game_state.game_over
        )
    
    def apply_move(self, game_state: GameStateData, move):
        """
        Apply a move to a game state (used in move simulation).
        
        Args:
            game_state: State to modify
            move: Move tuple to apply
            
        Returns:
            Modified game state
        """
        move_type, from_pos, to_pos = move
        
        if move_type == "reveal":
            # Reveal a face-down piece
            row, col = from_pos
            piece = game_state.board[row][col]
            if piece:
                piece.is_revealed = True
                
        elif move_type in ["move", "capture"]:
            # Move or capture with a piece
            from_row, from_col = from_pos
            to_row, to_col = to_pos
            piece = game_state.board[from_row][from_col]
            target = game_state.board[to_row][to_col]
            
            # Reveal target if hidden
            if target and not target.is_revealed:
                target.is_revealed = True
            
            # Execute move if valid
            if target is None or piece.can_capture(target):
                game_state.board[to_row][to_col] = piece
                game_state.board[from_row][from_col] = None
        
        return game_state
    
    def make_move(self, game) -> bool:
        """
        Make the AI's move using the selected strategy.
        
        Args:
            game: Main game object to modify
            
        Returns:
            True if move was made successfully
        """
        if game.game_over:
            return False
        
        # Create lightweight game state for analysis
        game_state = self.create_game_state(game)
        
        # Choose strategy based on difficulty
        if self.difficulty == "hard":
            _, best_move = self.minimax(game_state, self.search_depth, True)
        else:
            best_move = self.get_greedy_move(game_state)
        
        if not best_move:
            return False
        
        return self.execute_move(game, best_move)
    
    def get_greedy_move(self, game_state: GameStateData):
        """
        Simple greedy move selection for easy difficulty.
        
        Prioritizes captures, then moves, then reveals.
        
        Args:
            game_state: Current game state
            
        Returns:
            Selected move tuple or None
        """
        moves = self.get_all_moves(game_state)
        if not moves:
            return None
        
        # Prioritize captures of valuable pieces
        captures = [m for m in moves if m[0] == "capture"]
        if captures:
            best_capture = max(captures, 
                             key=lambda m: game_state.board[m[2][0]][m[2][1]].get_value())
            return best_capture
        
        # Then regular moves
        regular_moves = [m for m in moves if m[0] == "move"]
        if regular_moves:
            return random.choice(regular_moves)
        
        # Finally reveals
        reveals = [m for m in moves if m[0] == "reveal"]
        if reveals:
            return random.choice(reveals)
        
        return None
    
    def execute_move(self, game, move) -> bool:
        """
        Execute the selected move on the actual game board.
        
        Args:
            game: Main game object
            move: Move tuple to execute
            
        Returns:
            True if move was executed successfully
        """
        move_type, from_pos, to_pos = move
        
        if move_type == "reveal":
            # Reveal a face-down piece
            row, col = from_pos
            piece = game.board[row][col]
            if piece and not piece.is_revealed:
                piece.is_revealed = True
                if not game.first_piece_revealed:
                    game.current_player = piece.color
                    game.first_piece_revealed = True
                return True
        
        elif move_type in ["move", "capture"]:
            # Execute move or capture
            from_row, from_col = from_pos
            to_row, to_col = to_pos
            
            piece = game.board[from_row][from_col]
            target = game.board[to_row][to_col]
            
            if piece and piece.color == self.color:
                # Reveal target if hidden
                if target and not target.is_revealed:
                    target.is_revealed = True
                
                # Execute move if legal
                if target is None or piece.can_capture(target):
                    game.board[to_row][to_col] = piece
                    game.board[from_row][from_col] = None
                    return True
        
        return False


# ============================================================================
# MAIN GAME CLASS
# ============================================================================

class BanqiGame:
    """
    Main game class for Banqi (Chinese Dark Chess) vs AI.
    
    This class handles:
    - Game initialization and board setup
    - GUI creation and management
    - Player input and move validation
    - Game state management
    - AI integration
    - Win condition checking
    """
    
    def __init__(self, ai_difficulty="hard"):
        """
        Initialize a new Banqi game.
        
        Args:
            ai_difficulty (str): AI difficulty level ("easy" or "hard")
        """
        # Board dimensions
        self.board_width = 8
        self.board_height = 4
        
        # Game board - 2D list of Piece objects or None
        self.board: List[List[Optional[Piece]]] = [
            [None for _ in range(self.board_width)] 
            for _ in range(self.board_height)
        ]
        
        # Game state variables
        self.current_player = None  # Will be set when first piece is revealed
        self.game_state = GameState.SELECTING_PIECE
        self.selected_pos: Optional[Tuple[int, int]] = None  # Currently selected piece position
        self.first_piece_revealed = False  # Track if any piece has been revealed yet
        self.game_over = False
        
        # Player and AI management
        self.human_color = None  # Will be set when first piece is revealed
        self.ai_player = None    # Will be created when human color is determined
        self.ai_thinking = False  # Flag to prevent input during AI calculations

        # Initialize game components
        self.init_pieces()
        
        # Create and setup GUI
        self.root = tk.Tk()
        self.root.title("Banqi vs AI (Chinese Dark Chess)")
        self.root.geometry("800x600")
        self.root.configure(bg='#8B4513')
        
        self.setup_gui()
        self.update_display()
    
    def init_pieces(self):
        """
        Initialize and randomly place all 32 pieces on the board.
        
        Creates the complete set of pieces for both players:
        - 1 General, 2 Advisors, 2 Elephants, 2 Chariots, 
          2 Horses, 2 Cannons, 5 Soldiers per color
        - Shuffles and places them face-down on the board
        """
        pieces = []
        
        # Create pieces for both players (Red and Black)
        for color in [PieceColor.RED, PieceColor.BLACK]:
            pieces.append(Piece(PieceType.GENERAL, color))  # 1 General
            pieces.extend([Piece(PieceType.ADVISOR, color) for _ in range(2)])  # 2 Advisors
            pieces.extend([Piece(PieceType.ELEPHANT, color) for _ in range(2)])  # 2 Elephants
            pieces.extend([Piece(PieceType.CHARIOT, color) for _ in range(2)])  # 2 Chariots
            pieces.extend([Piece(PieceType.HORSE, color) for _ in range(2)])  # 2 Horses
            pieces.extend([Piece(PieceType.CANNON, color) for _ in range(2)])  # 2 Cannons
            pieces.extend([Piece(PieceType.SOLDIER, color) for _ in range(5)])  # 5 Soldiers
        
        # Randomly shuffle all pieces
        random.shuffle(pieces)
        
        # Place shuffled pieces on the board
        piece_index = 0
        for row in range(self.board_height):
            for col in range(self.board_width):
                self.board[row][col] = pieces[piece_index]
                piece_index += 1
    
    def setup_gui(self):
        """
        Create and configure all GUI components.
        
        Sets up:
        - Main window layout
        - Title and status labels
        - Circular game board with canvas
        - Control buttons and difficulty selector
        - Event bindings for mouse clicks
        """
        # Main container frame
        main_frame = tk.Frame(self.root, bg='#8B4513')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Game title
        title_label = tk.Label(
            main_frame, 
            text="Banqi", 
            font=('Arial', 20, 'bold'), 
            bg='#8B4513', 
            fg='white'
        )
        title_label.pack(pady=10)
        
        # Game information display
        info_frame = tk.Frame(main_frame, bg='#8B4513')
        info_frame.pack(pady=10)
        
        # Current player indicator
        self.current_player_label = tk.Label(
            info_frame, 
            text="Current Player: You (Red)", 
            font=('Arial', 14), 
            bg='#8B4513', 
            fg='white'
        )
        self.current_player_label.pack(side=tk.LEFT, padx=20)
        
        # Game status messages
        self.game_status_label = tk.Label(
            info_frame, 
            text="Select a piece to reveal or move", 
            font=('Arial', 12), 
            bg='#8B4513', 
            fg='white'
        )
        self.game_status_label.pack(side=tk.LEFT, padx=20)
        
        # Board container with raised border
        board_frame = tk.Frame(main_frame, bg='#654321', relief='raised', bd=2)
        board_frame.pack(pady=20)
        
        # Canvas for drawing circular pieces
        self.canvas = tk.Canvas(board_frame, width=640, height=320, bg='#654321')
        self.canvas.pack(padx=10, pady=10)
        
        # Calculate circle layout parameters
        self.circle_size = 35      # Radius of each piece circle
        self.grid_spacing = 80     # Distance between circle centers
        self.board_start_x = 40    # Left margin
        self.board_start_y = 40    # Top margin
        
        # Storage for canvas objects (circles and text)
        self.canvas_circles = {}
        self.canvas_texts = {}
        
        # Create visual elements for each board position
        for row in range(self.board_height):
            for col in range(self.board_width):
                # Calculate center position for this grid square
                x = self.board_start_x + col * self.grid_spacing
                y = self.board_start_y + row * self.grid_spacing
                
                # Create circular piece representation
                circle = self.canvas.create_oval(
                    x - self.circle_size, y - self.circle_size,
                    x + self.circle_size, y + self.circle_size,
                    fill='#DEB887', outline='black', width=2
                )
                
                # Create text for piece symbols
                text = self.canvas.create_text(
                    x, y, text="", font=('Arial', 10, 'bold'), fill='black'
                )
                
                # Store references for later updates
                self.canvas_circles[(row, col)] = circle
                self.canvas_texts[(row, col)] = text
                
                # Bind click events to both circle and text
                self.canvas.tag_bind(
                    circle, '<Button-1>', 
                    lambda event, r=row, c=col: self.on_square_click(r, c)
                )
                self.canvas.tag_bind(
                    text, '<Button-1>', 
                    lambda event, r=row, c=col: self.on_square_click(r, c)
                )
        
        # Control panel with buttons
        control_frame = tk.Frame(main_frame, bg='#8B4513')
        control_frame.pack(pady=20)
        
        # New game button
        new_game_btn = tk.Button(
            control_frame, 
            text="New Game", 
            command=self.new_game, 
            font=('Arial', 12)
        )
        new_game_btn.pack(side=tk.LEFT, padx=10)
        
        # AI difficulty selector
        difficulty_frame = tk.Frame(control_frame, bg='#8B4513')
        difficulty_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(difficulty_frame, text="AI:", bg='#8B4513', fg='white').pack(side=tk.LEFT)
        self.difficulty_var = tk.StringVar(value="hard")
        difficulty_combo = ttk.Combobox(
            difficulty_frame, 
            textvariable=self.difficulty_var, 
            values=["easy", "hard"], 
            state="readonly", 
            width=8
        )
        difficulty_combo.pack(side=tk.LEFT, padx=5)
        
        # Rules display button
        rules_btn = tk.Button(
            control_frame, 
            text="Rules", 
            command=self.show_rules, 
            font=('Arial', 12)
        )
        rules_btn.pack(side=tk.LEFT, padx=10)
        
        # Quit button
        quit_btn = tk.Button(
            control_frame, 
            text="Quit", 
            command=self.root.quit, 
            font=('Arial', 12)
        )
        quit_btn.pack(side=tk.LEFT, padx=10)
    
    def on_square_click(self, row: int, col: int):
        """
        Handle mouse clicks on board squares.
        
        Manages the complete interaction flow:
        - Piece selection and deselection
        - Piece revelation
        - Move validation and execution
        - Turn management
        
        Args:
            row (int): Board row (0-3)
            col (int): Board column (0-7)
        """
        # Ignore clicks during game over or AI thinking
        if self.game_over or self.ai_thinking:
            return
        
        # Ensure it's the human player's turn (after first piece reveal)
        if self.human_color is not None and self.current_player != self.human_color:
            return
        
        piece = self.board[row][col]
        
        # CASE 1: No piece currently selected
        if self.selected_pos is None:
            # Can't select empty squares
            if piece is None:
                return
            
            # If piece is unrevealed, reveal it
            if not piece.is_revealed:
                piece.is_revealed = True
                
                # Handle first piece revelation
                if not self.first_piece_revealed:
                    # Player gets the color they reveal
                    self.human_color = piece.color
                    self.current_player = piece.color
                    
                    # Create AI with opposite color
                    ai_color = PieceColor.BLACK if piece.color == PieceColor.RED else PieceColor.RED
                    self.ai_player = AIPlayer(ai_color, self.difficulty_var.get())
                    self.first_piece_revealed = True
                
                self.end_turn()
                return
            
            # Select revealed piece if it belongs to current player
            if self.human_color is None or piece.color == self.human_color:
                self.selected_pos = (row, col)
                self.update_display()
            
            return
        
        # CASE 2: A piece is already selected
        selected_row, selected_col = self.selected_pos
        selected_piece = self.board[selected_row][selected_col]
        
        # Clicking same piece deselects it
        if (row, col) == self.selected_pos:
            self.selected_pos = None
            self.update_display()
            return
        
        # Clicking another friendly piece selects it instead
        if piece and piece.is_revealed and piece.color == self.current_player:
            self.selected_pos = (row, col)
            self.update_display()
            return
        
        # Prevent moves to unrevealed pieces
        if piece and not piece.is_revealed:
            self.game_status_label.config(
                text="Cannot move to unrevealed pieces! Click on empty squares or revealed enemy pieces."
            )
            self.update_display()
            return
        
        # Attempt to execute the move
        if self.is_valid_move(selected_row, selected_col, row, col):
            target_piece = self.board[row][col]
            move_successful = False
            
            # Handle move to occupied square
            if target_piece is not None:
                # Double-check that target is revealed (should always be true here)
                if not target_piece.is_revealed:
                    self.game_status_label.config(text="Error: Cannot interact with unrevealed pieces!")
                    self.update_display()
                    return
                
                # Attempt capture
                if selected_piece.can_capture(target_piece):
                    self.board[row][col] = selected_piece
                    self.board[selected_row][selected_col] = None
                    move_successful = True
                else:
                    # Invalid capture attempt
                    self.game_status_label.config(text="Cannot capture that piece! Try another move.")
                    move_successful = False
            else:
                # Move to empty square
                self.board[row][col] = selected_piece
                self.board[selected_row][selected_col] = None
                move_successful = True
            
            # Complete turn if move was successful
            if move_successful:
                self.selected_pos = None
                self.end_turn()
            else:
                # Keep piece selected for retry
                self.update_display()
        else:
            # Invalid move - provide feedback
            self.game_status_label.config(text="Invalid move! Try a different position.")
            self.update_display()
    
    def update_display(self):
        """
        Update the visual display of the circular board.
        
        Handles complete visual state management:
        - Board square appearance (empty, revealed, hidden pieces)
        - Selection highlighting
        - Player information display
        - Game status messaging
        - Win/loss condition display
        """
        
        # SECTION 1: Update board visual state
        for row in range(self.board_height):
            for col in range(self.board_width):
                piece = self.board[row][col]
                circle = self.canvas_circles[(row, col)]
                text = self.canvas_texts[(row, col)]
                
                # CASE 1A: Empty square
                if piece is None:
                    self.canvas.itemconfig(circle, fill='#DEB887', outline='black', width=2)
                    self.canvas.itemconfig(text, text="", fill='black')
                
                # CASE 1B: Revealed piece
                elif piece.is_revealed:
                    color = '#FFE4B5' if piece.color == PieceColor.RED else '#D3D3D3'
                    text_color = 'red' if piece.color == PieceColor.RED else 'black'
                    self.canvas.itemconfig(circle, fill=color, outline='black', width=2)
                    self.canvas.itemconfig(text, text=str(piece), fill=text_color)
                
                # CASE 1C: Hidden piece
                else:
                    self.canvas.itemconfig(circle, fill='#8B4513', outline='black', width=2)
                    self.canvas.itemconfig(text, text="?", fill='white')
                
                # Apply selection highlighting
                if self.selected_pos == (row, col):
                    self.canvas.itemconfig(circle, outline='yellow', width=4)
                else:
                    self.canvas.itemconfig(circle, outline='black', width=2)
        
        # SECTION 2: Update player information display
        if self.human_color is None:
            # Before first piece revelation - always human's turn
            self.current_player_label.config(text="Current Player: You")
        else:
            # After first piece revelation - show color-specific information
            player_color_name = "Red" if self.human_color == PieceColor.RED else "Black"
            ai_color_name = "Black" if self.human_color == PieceColor.RED else "Red"
            
            if self.current_player == self.human_color:
                self.current_player_label.config(text=f"Current Player: You ({player_color_name})")
            else:
                self.current_player_label.config(text=f"Current Player: AI ({ai_color_name})")
        
        # SECTION 3: Update game status display
        if self.game_over:
            # Count remaining pieces to determine winner
            red_pieces = 0
            black_pieces = 0
            
            for row in range(self.board_height):
                for col in range(self.board_width):
                    piece = self.board[row][col]
                    if piece is not None:
                        if piece.color == PieceColor.RED:
                            red_pieces += 1
                        elif piece.color == PieceColor.BLACK:
                            black_pieces += 1
            
            # CASE 3A: Victory by elimination
            if red_pieces == 0:
                # Black wins by eliminating all red pieces
                if self.human_color == PieceColor.BLACK:
                    self.game_status_label.config(text="Game Over! You win!")
                else:
                    self.game_status_label.config(text="Game Over! AI wins!")
            
            elif black_pieces == 0:
                # Red wins by eliminating all black pieces
                if self.human_color == PieceColor.RED:
                    self.game_status_label.config(text="Game Over! You win!")
                else:
                    self.game_status_label.config(text="Game Over! AI wins!")
            
            # CASE 3B: Victory by stalemate (no valid moves)
            else:
                if self.current_player == self.human_color:
                    self.game_status_label.config(text="Game Over! AI wins! (No moves available)")
                else:
                    self.game_status_label.config(text="Game Over! You win! (AI has no moves)")
        
        # CASE 3C: Game in progress - various states
        elif self.ai_thinking:
            self.game_status_label.config(text="AI is thinking...")
        
        elif self.human_color is not None and self.current_player != self.human_color:
            self.game_status_label.config(text="AI's turn")
        
        elif self.selected_pos:
            # Provide context-specific guidance for selected piece
            selected_piece = self.board[self.selected_pos[0]][self.selected_pos[1]]
            if selected_piece and selected_piece.piece_type == PieceType.CANNON:
                self.game_status_label.config(text="Cannon: Move adjacent OR jump over exactly one piece to capture")
            else:
                self.game_status_label.config(text="Select where to move (only up/down/left/right)")
        
        else:
            # Default state - waiting for piece selection
            self.game_status_label.config(text="Select a piece to reveal or move")

    def is_cannon_jump_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """
        Validate cannon jump moves (special cannon capture mechanic).
        
        Cannons can capture by jumping over exactly one piece in a straight line.
        
        Args:
            from_row, from_col: Starting position
            to_row, to_col: Target position
            
        Returns:
            bool: True if this is a valid cannon jump move
        """
        selected_piece = self.board[from_row][from_col]
        if selected_piece is None or selected_piece.piece_type != PieceType.CANNON:
            return False
        
        # Must move in straight line (same row or column)
        if from_row != to_row and from_col != to_col:
            return False
        
        # Must have a target piece to capture
        target_piece = self.board[to_row][to_col]
        if target_piece is None:
            return False
        
        # Count pieces between cannon and target
        pieces_between = 0
        
        if from_row == to_row:  # Horizontal jump
            start_col = min(from_col, to_col) + 1
            end_col = max(from_col, to_col)
            
            for col in range(start_col, end_col):
                if self.board[from_row][col] is not None:
                    pieces_between += 1
        else:  # Vertical jump
            start_row = min(from_row, to_row) + 1
            end_row = max(from_row, to_row)
            
            for row in range(start_row, end_row):
                if self.board[row][from_col] is not None:
                    pieces_between += 1
        
        # Valid jump requires exactly one piece in between
        return pieces_between == 1
    
    def is_orthogonal_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """
        Check if move is orthogonal (up/down/left/right only, no diagonals).
        
        Args:
            from_row, from_col: Starting position
            to_row, to_col: Target position
            
        Returns:
            bool: True if move is exactly one square in orthogonal direction
        """
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        
        # Must move exactly one square in exactly one direction
        return (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)
    
    def is_valid_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """
        Comprehensive move validation for all piece types.
        
        Checks:
        - Basic validity (piece exists, belongs to current player)
        - Board boundaries
        - Movement rules (orthogonal for most pieces)
        - Special cannon rules (adjacent move or jump capture)
        
        Args:
            from_row, from_col: Starting position
            to_row, to_col: Target position
            
        Returns:
            bool: True if move is legal
        """
        selected_piece = self.board[from_row][from_col]
        target = self.board[to_row][to_col]
        
        # Basic validation
        if selected_piece is None or not selected_piece.is_revealed:
            return False
        
        if selected_piece.color != self.current_player:
            return False
        
        # Boundary check
        if not (0 <= to_row < self.board_height and 0 <= to_col < self.board_width):
            return False
        
        # Special cannon movement rules
        if selected_piece.piece_type == PieceType.CANNON:
            # Cannons can make normal adjacent moves to empty squares
            if target is None and self.is_orthogonal_move(from_row, from_col, to_row, to_col):
                return True
            # Cannons can also capture by jumping over pieces
            if target is not None:
                return self.is_cannon_jump_move(from_row, from_col, to_row, to_col)
            return False
        
        # All other pieces: must be adjacent and orthogonal
        if not self.is_orthogonal_move(from_row, from_col, to_row, to_col):
            return False
        
        # Valid move to adjacent square
        return True
    
    def end_turn(self):
        """
        End current turn and switch players.
        
        Checks for game over conditions, determines winner if game is over,
        and handles AI turn scheduling.
        """
        # Check for game over conditions first
        if self.check_game_over():
            self.game_over = True
            self.update_display()
            
            # NEW: Determine winner by checking who has pieces left
            red_pieces = 0
            black_pieces = 0
            
            # Count remaining pieces for each color
            for row in range(self.board_height):
                for col in range(self.board_width):
                    piece = self.board[row][col]
                    if piece is not None:
                        if piece.color == PieceColor.RED:
                            red_pieces += 1
                        elif piece.color == PieceColor.BLACK:
                            black_pieces += 1
            
            # Determine winner based on remaining pieces
            if red_pieces == 0:
                # Black wins - all red pieces eliminated
                if self.human_color == PieceColor.BLACK:
                    messagebox.showinfo("Game Over", "You win!")
                else:
                    messagebox.showinfo("Game Over", "AI wins!")
            elif black_pieces == 0:
                # Red wins - all black pieces eliminated
                if self.human_color == PieceColor.RED:
                    messagebox.showinfo("Game Over", "You win!")
                else:
                    messagebox.showinfo("Game Over", "AI wins!")
            else:
                # No pieces eliminated, so current player lost due to no moves available
                if self.current_player == self.human_color:
                    messagebox.showinfo("Game Over", "AI wins! (No moves available)")
                else:
                    messagebox.showinfo("Game Over", "You win! (AI has no moves)")
            return

        # Switch to the other player
        self.current_player = PieceColor.BLACK if self.current_player == PieceColor.RED else PieceColor.RED
        self.update_display()

        # If it's AI's turn and AI player exists, schedule AI move after a short delay
        if (self.ai_player is not None and 
            self.current_player == self.ai_player.color and 
            not self.game_over):
            self.ai_thinking = True  # Set thinking flag for UI feedback
            self.update_display()
            self.root.after(500, self.make_ai_move)  # 500ms delay for better UX

    def make_ai_move(self):
        """
        Execute AI player's move and handle the result.
        
        Called after a short delay to provide better user experience.
        """
        if self.ai_player.make_move(self):
            self.ai_thinking = False
            self.end_turn()
        else:
            # AI couldn't make a move (shouldn't happen in normal play)
            self.ai_thinking = False
            self.update_display()

    def check_game_over(self) -> bool:
        """
        Check if the game is over due to elimination or no available moves.
        
        Returns True if:
        - One color has been completely eliminated, OR
        - All pieces are revealed AND current player has no valid moves
        
        Returns False if game should continue.
        """
        # Game cannot be over before first piece is revealed
        if not self.first_piece_revealed:
            return False
        
        # NEW: Check if either color has been completely eliminated
        red_pieces = 0
        black_pieces = 0
        
        # Count remaining pieces for each color on the board
        for row in range(self.board_height):
            for col in range(self.board_width):
                piece = self.board[row][col]
                if piece is not None:
                    if piece.color == PieceColor.RED:
                        red_pieces += 1
                    elif piece.color == PieceColor.BLACK:
                        black_pieces += 1
        
        # Game over if one color has no pieces left (elimination victory)
        if red_pieces == 0 or black_pieces == 0:
            return True
        
        # Check if there are any unrevealed pieces that can be flipped
        has_unrevealed_pieces = False
        for row in range(self.board_height):
            for col in range(self.board_width):
                piece = self.board[row][col]
                if piece is not None and not piece.is_revealed:
                    has_unrevealed_pieces = True
                    break
            if has_unrevealed_pieces:
                break
        
        # If there are still unrevealed pieces, game continues
        # (current player can always reveal a piece as a valid move)
        if has_unrevealed_pieces:
            return False
        
        # All pieces are revealed - check if current player can move any of their pieces
        current_player_has_moves = False
        current_player_has_pieces = False
        
        # Scan board for current player's pieces and check for valid moves
        for row in range(self.board_height):
            for col in range(self.board_width):
                piece = self.board[row][col]
                if piece is not None and piece.color == self.current_player:
                    current_player_has_pieces = True
                    # Check if this piece has any valid moves available
                    if self.has_valid_moves(row, col):
                        current_player_has_moves = True
                        break
            if current_player_has_moves:
                break
        
        # Game over if current player has no pieces or no valid moves 
        # (only applies when all pieces are revealed)
        return not current_player_has_pieces or not current_player_has_moves

    def has_valid_moves(self, row: int, col: int) -> bool:
        """
        Check if a specific piece has any legal moves available.
        
        Used for game over detection and AI planning.
        
        Args:
            row, col: Position of piece to check
            
        Returns:
            bool: True if piece has at least one valid move
        """
        piece = self.board[row][col]
        if piece is None or not piece.is_revealed or piece.color != self.current_player:
            return False
        
        # Special handling for cannons
        if piece.piece_type == PieceType.CANNON:
            # Check normal adjacent moves
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < self.board_height and 0 <= new_col < self.board_width:
                    target = self.board[new_row][new_col]
                    
                    # Can move to empty square
                    if target is None:
                        return True
                    
                    # Can capture adjacent enemy
                    if target.is_revealed and target.color != piece.color and piece.can_capture(target):
                        return True
            
            # Check cannon jump moves to all positions
            for target_row in range(self.board_height):
                for target_col in range(self.board_width):
                    target_piece = self.board[target_row][target_col]
                    
                    # Skip if no target or same position
                    if target_piece is None or (target_row == row and target_col == col):
                        continue
                    
                    # Skip friendly pieces
                    if target_piece.color == piece.color:
                        continue
                    
                    # Check for valid jump
                    if self.is_cannon_jump_move(row, col, target_row, target_col):
                        return True
            return False
        
        # Check moves for all other piece types
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < self.board_height and 0 <= new_col < self.board_width:
                target = self.board[new_row][new_col]
                
                # Can move to empty square
                if target is None:
                    return True
                
                # Can capture revealed enemy
                if target.is_revealed and target.color != piece.color and piece.can_capture(target):
                    return True
        
        return False

    def new_game(self):
        """
        Reset all game state and start a fresh game.
        
        Preserves the selected AI difficulty setting.
        """
        difficulty = self.difficulty_var.get()
        
        # Reset game state
        self.board = [[None for _ in range(self.board_width)] for _ in range(self.board_height)]
        self.current_player = PieceColor.RED  # Player always starts first
        self.selected_pos = None
        self.first_piece_revealed = False
        self.game_over = False
        self.human_color = None  # Will be determined by first reveal
        self.ai_player = None    # Will be created after first reveal
        self.ai_thinking = False
        
        # Reinitialize board and display
        self.init_pieces()
        self.update_display()

    def show_rules(self):
        """
        Display comprehensive game rules in a popup window.
        
        Covers all aspects of Banqi gameplay including:
        - Basic setup and flow
        - Piece rankings and capture rules
        - Movement restrictions
        - Special cannon mechanics
        - Win conditions
        - AI difficulty explanations
        """
        rules_text = """
    Banqi (Chinese Dark Chess) vs AI Rules:

    1. The game is played on a 4x8 board with 32 pieces (16 per player). 
    
    2. Player moves first and is whatever colour they reveal

    3. All pieces start face-down. Player and computer take turns either:
    - Revealing a face-down piece
    - Moving a revealed piece one square orthogonally (up/down/left/right only)
    - Capturing an opponent's piece

    4. Piece Rankings (high to low):
    General > Advisor > Elephant > Chariot > Horse > Cannon > Soldier

    6. Capture Rules:
    - Higher ranked pieces can capture lower ranked pieces
    - Exception: Soldiers can capture the General
    - Exception: General cannot capture Soldiers
    - Cannons can capture any piece

    7. Movement Rules:
    - Most pieces can only move one square in orthogonal directions (up, down, left, right)
    - No diagonal movement allowed for any piece
    - Can move to any adjacent empty square
    - Can capture adjacent enemy pieces (if rules allow)
    - CANNON SPECIAL RULES: 
        * Cannons can move one square orthogonally like other pieces
        * Cannons can also capture by jumping over exactly one piece (revealed or unrevealed) in a straight line (horizontal or vertical)
        * There can be multiple empty spaces between the cannon and the jumping piece, and between the jumping piece and the target piece

    8. Win Condition:
    - The game ends when a player cannot make any legal moves
    - This usually happens when all of a player's pieces are captured

    9. AI Difficulty:
    - Easy: AI uses basic strategy
    - Hard: AI uses advanced minimax algorithm with tactical planning

    10. Strategy Tips:
    - Soldiers are valuable for threatening the General
    - Cannons are powerful but vulnerable due to their movement restrictions
    - Keep track of revealed pieces to plan your strategy
    - The AI will adapt to your playstyle, so vary your tactics!
        """
        
        messagebox.showinfo("Banqi vs AI Rules", rules_text)

    def run(self):
        """Start the main game loop"""
        # Begin the Tkinter event loop to handle user interactions
        self.root.mainloop()


# Entry point of the program
if __name__ == "__main__":
    # Create a new game instance with default AI difficulty
    game = BanqiGame()
    
    # Start the game's main loop
    game.run()