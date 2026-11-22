# Sokoban Solver

Push-optimal Sokoban solver for the make.fun sokoban-solver-01 competition. It splits each level into hallway + rooms, solves every room push-optimally with A* (Hungarian assignment heuristic + deadlock pruning), and stitches the rooms together with player walks.

Run locally from the kit dir: `mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node /path/to/main.ts`, then verify with `node ../verify.mjs practice/input out`.
