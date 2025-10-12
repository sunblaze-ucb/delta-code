# Manufactoria Solution DSL

A Domain Specific Language for describing Manufactoria puzzle solutions in text format.

## Overview

Manufactoria is a puzzle game where you build automated factories to sort robots based on their colored tape patterns. Robots enter your factory carrying sequences of colored tape, and you must route them to the correct destinations based on the given criteria.

## Game Mechanics

### Robots and Tape
- **Robots**: Each robot carries a sequence of colored tapes
- **Tape Colors**: Primary colors are Blue (B) and Red (R), with additional Yellow (Y) and Green (G) for advanced puzzles
- **Tape Representation**: Sequences are represented as strings (e.g., `RBRR`, `BBR`, or empty string `""`)

### Operations
- **Pull**: Remove tape from the front of the robot's sequence
- **Paint**: Add colored tape to the end of the robot's sequence
- **Route**: Direct robots through the factory based on their current tape state

### Objective
Route robots to the correct destinations based on their final tape configuration and the puzzle requirements:
- **Accepted**: Robot reaches the END node
- **Rejected**: Robot is not routed to the END node or caught in an infinite loop. 

### Objective (output check)
Route robots to the correct destinations based on their final tape configuration and the puzzle requirements:
- **Accepted**: Robot reaches the END node and meets the puzzle's acceptance criteria
- **Rejected**: Robot is routed to the NONE node, or caught in an infinite loop, or robot reaches the END node but fails to meet the puzzle's acceptance criteria


## DSL Syntax

### Program Structure

Every solution must start with a `START` directive and end with an `END` directive, wrapped with ``` markers:

```
START start:
    NEXT <next_node_id>

# Factory logic goes here

END end
```

### Node Types

#### 1. Puller Nodes

Pullers remove specific colors from the front of the robot's tape sequence and route based on the current front color.

**Red/Blue Puller:**
```
PULLER_RB <node_id>:
    [R] <next_node_id>      # Route and remove color if front tape is Red
    [B] <next_node_id>      # Route and remove color if front tape is Blue
    [EMPTY] <next_node_id>  # Route if no tape or front tape is neither red nor blue
```

**Yellow/Green Puller:**
```
PULLER_YG <node_id>:
    [Y] <next_node_id>      # Route and remove color if front tape is Yellow
    [G] <next_node_id>      # Route and remove color if front tape is Green
    [EMPTY] <next_node_id>  # Route if no tape or front tape is neither yellow nor green
```

**Note**: Unspecified branches default to `NONE`, which rejects the robot.

#### 2. Painter Nodes

Painters add colored tape to the end of the robot's sequence and continue to the next node.

```
PAINTER_RED <node_id>:
    NEXT <next_node_id>

PAINTER_BLUE <node_id>:
    NEXT <next_node_id>

PAINTER_YELLOW <node_id>:
    NEXT <next_node_id>

PAINTER_GREEN <node_id>:
    NEXT <next_node_id>
```

## Syntax Rules

1. **Node IDs**: Must be unique identifiers (alphanumeric characters and underscores only)
2. **Comments**: Lines starting with `#` are comments (single-line only)
3. **Indentation**: Use consistent spaces or tabs for route definitions
4. **Case Sensitivity**: Colors must be uppercase (R, B, Y, G)
5. **Termination**: 
   - Robots routed to `NONE` are rejected
   - Robots routed to the END node are accepted
6. **Code Blocks**: Final factory code should be wrapped in triple backticks with ``` markers


## Syntax Rules (Output check)

1. **Node IDs**: Must be unique identifiers (alphanumeric characters and underscores only)
2. **Comments**: Lines starting with `#` are comments (single-line only)
3. **Indentation**: Use consistent spaces or tabs for route definitions
4. **Case Sensitivity**: Colors must be uppercase (R, B, Y, G)
5. **Termination**: 
   - Robots routed to `NONE` are rejected
   - Robots routed to the END node are accepted if they meet the puzzle criteria, otherwise rejected
6. **Code Blocks**: Final factory code should be wrapped in triple backticks with ``` markers


## Example

Here's a simple example that accepts robots with exactly one red tape (ending tape should be empty):

```
START start:
    NEXT entry

PULLER_RB entry:
    [R] end

END end
```

# Task 
Your task is to design a factory with code with following functionality:

The input tape carries only red/blue colors. Move the last symbol to the front.

