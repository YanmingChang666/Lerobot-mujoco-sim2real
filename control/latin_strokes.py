"""Original monoline block capitals, for pen trajectories rather than glyph outlines.

ASCII lowercase input is rendered as capitals. The whole text is uniformly
scaled to fit size_m x size_m and centered; spaces create pen-up gaps.
"""
import math
import string

# Each semicolon begins a new stroke; glyph cell is 0..4 by 0..6.
_GLYPHS = {
    'A': '0,0 2,6 4,0;1,3 3,3',
    'B': '0,0 0,6 3,6 4,5 4,4 3,3 0,3;3,3 4,2 4,1 3,0 0,0',
    'C': '4,5 3,6 1,6 0,5 0,1 1,0 3,0 4,1',
    'D': '0,0 0,6 2,6 4,4 4,2 2,0 0,0',
    'E': '4,6 0,6 0,0 4,0;0,3 3,3',
    'F': '0,0 0,6 4,6;0,3 3,3',
    'G': '4,5 3,6 1,6 0,5 0,1 1,0 3,0 4,1 4,3 2,3',
    'H': '0,0 0,6;4,0 4,6;0,3 4,3',
    'I': '0,6 4,6;2,6 2,0;0,0 4,0',
    'J': '0,6 4,6;3,6 3,1 2,0 1,0 0,1',
    'K': '0,0 0,6;4,6 0,3 4,0',
    'L': '0,6 0,0 4,0',
    'M': '0,0 0,6 2,3 4,6 4,0',
    'N': '0,0 0,6 4,0 4,6',
    'O': '1,0 0,1 0,5 1,6 3,6 4,5 4,1 3,0 1,0',
    'P': '0,0 0,6 3,6 4,5 4,4 3,3 0,3',
    'Q': '1,0 0,1 0,5 1,6 3,6 4,5 4,1 3,0 1,0;2,2 4,0',
    'R': '0,0 0,6 3,6 4,5 4,4 3,3 0,3;2,3 4,0',
    'S': '4,5 3,6 1,6 0,5 0,4 1,3 3,3 4,2 4,1 3,0 1,0 0,1',
    'T': '0,6 4,6;2,6 2,0',
    'U': '0,6 0,1 1,0 3,0 4,1 4,6',
    'V': '0,6 2,0 4,6',
    'W': '0,6 1,0 2,3 3,0 4,6',
    'X': '0,6 4,0;0,0 4,6',
    'Y': '0,6 2,3 4,6;2,3 2,0',
    'Z': '0,6 4,6 0,0 4,0',
}


def latin_to_strokes(text, size_m=.08):
    """Return centered strokes in meters, preserving inter-letter/word gaps."""
    if not math.isfinite(size_m) or size_m <= 0:
        raise ValueError('Text size must be finite and positive')
    if not text.strip() or any(c not in string.ascii_letters + ' ' for c in text):
        raise ValueError('English writing supports A-Z, a-z and spaces only; lowercase is rendered as uppercase')
    text = text.strip().upper()
    width = (len(text)-1)*6 + 4
    scale = size_m/max(width, 6)
    strokes = []
    for index, char in enumerate(text):
        if char == ' ':
            continue
        for stroke in _GLYPHS[char].split(';'):
            points = [tuple(map(float, p.split(','))) for p in stroke.split()]
            strokes.append([((x+index*6-width/2)*scale, (y-3)*scale) for x, y in points])
    return strokes
