// Sokoban solver for the make.fun sokoban-solver-01 competition.
//
// Strategy: split the level into the hallway plus the rooms that open off
// it, solve each room push-optimally with A* (Hungarian assignment
// heuristic over BFS box distances + static deadlock pruning), then stitch
// the rooms together with player walks (walks never cost pushes).
// A bounded move-A* is kept as a fallback for any room (or level) the
// decomposition cannot handle, so output is always a legal move string.
//
// Deterministic by construction: fixed node budgets, fixed tie-breaking
// (f, h, insertion order), no clock, no randomness, no I/O outside the
// input and output directories.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const ROOM_NODE_BUDGET = 1_500_000;
const MOVE_NODE_BUDGET = 200_000;

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];

interface Level {
  w: number;
  h: number;
  floor: boolean[];
  goals: number[];
  goalSet: Set<number>;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n');
  while (rows.length > 0 && rows[rows.length - 1].length === 0) rows.pop();
  const h = rows.length;
  const w = Math.max(0, ...rows.map((r) => r.length));
  const floor = new Array<boolean>(w * h).fill(false);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ch = rows[y][x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      floor[i] = true;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, floor, goals, goalSet: new Set(goals), boxes, player };
}

// Neighbour of cell i in direction d (0=u, 1=d, 2=l, 3=r); -1 off board.
function step(lv: Level, i: number, d: number): number {
  const x = i % lv.w;
  const y = (i / lv.w) | 0;
  const nx = x + DX[d];
  const ny = y + DY[d];
  if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) return -1;
  return ny * lv.w + nx;
}

// Flood fill from `start` over `open` cells, treating `blocked` as walls.
function flood(start: number, open: Set<number>, blocked: Set<number>, lv: Level): Set<number> {
  const seen = new Set<number>([start]);
  const stack = [start];
  while (stack.length > 0) {
    const c = stack.pop()!;
    for (let d = 0; d < 4; d++) {
      const n = step(lv, c, d);
      if (n < 0 || !open.has(n) || blocked.has(n) || seen.has(n)) continue;
      seen.add(n);
      stack.push(n);
    }
  }
  return seen;
}

// Smallest cell index in the region the player can reach from `from`
// (boxes are obstacles). All player positions in one region are one state.
function canonical(from: number, boxes: Set<number>, open: Set<number>, lv: Level): number {
  const region = flood(from, open, boxes, lv);
  let best = Infinity;
  for (const c of region) if (c < best) best = c;
  return best;
}

// ---- binary heap ordered by (f, h, insertion order) -----------------------

interface HeapNode {
  f: number;
  h: number;
  g: number;
  seq: number;
  key: string;
}

function heapLess(a: HeapNode, b: HeapNode): boolean {
  if (a.f !== b.f) return a.f < b.f;
  if (a.h !== b.h) return a.h < b.h;
  return a.seq < b.seq;
}

class Heap<T extends HeapNode> {
  private a: T[] = [];
  get size(): number {
    return this.a.length;
  }
  push(node: T): void {
    this.a.push(node);
    let i = this.a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(this.a[i], this.a[p])) break;
      const t = this.a[i];
      this.a[i] = this.a[p];
      this.a[p] = t;
      i = p;
    }
  }
  pop(): T | undefined {
    const top = this.a[0];
    const last = this.a.pop();
    if (this.a.length > 0 && last !== undefined) {
      this.a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < this.a.length && heapLess(this.a[l], this.a[m])) m = l;
        if (r < this.a.length && heapLess(this.a[r], this.a[m])) m = r;
        if (m === i) break;
        const t = this.a[i];
        this.a[i] = this.a[m];
        this.a[m] = t;
        i = m;
      }
    }
    return top;
  }
}

// ---- rooms ----------------------------------------------------------------

interface Room {
  cells: number[]; // sub-board floor cells (component + entry cell)
  open: Set<number>;
  door: number;
  entry: number;
  boxes: number[]; // initial boxes (sorted)
  goals: number[];
  goalSet: Set<number>;
  dist: number[][]; // dist[from][to]: BFS box distance over open cells
  dead: Set<number>; // static dead squares (never goals)
  seals: boolean; // true when every interior neighbour of the door is a goal:
  // once solved, a box sits on each of them and the player can never leave
  // through the door, so this room must be solved last.
}

function buildRoom(lv: Level, comp: Set<number>, door: number, entry: number): Room {
  const open = new Set<number>(comp);
  open.add(entry);
  const cells = [...open];
  const boxes = lv.boxes.filter((b) => comp.has(b));
  const goals = lv.goals.filter((g) => comp.has(g));
  const goalSet = new Set(goals);

  // All-pairs BFS distances for the assignment heuristic.
  const dist: number[][] = new Array<number[]>(lv.w * lv.h);
  for (const c of cells) {
    const row = new Array<number>(lv.w * lv.h).fill(Infinity);
    row[c] = 0;
    const q = [c];
    for (let qi = 0; qi < q.length; qi++) {
      const cur = q[qi];
      for (let d = 0; d < 4; d++) {
        const n = step(lv, cur, d);
        if (n < 0 || !open.has(n) || row[n] !== Infinity) continue;
        row[n] = row[cur] + 1;
        q.push(n);
      }
    }
    dist[c] = row;
  }

  // Static dead squares: corners, plus wall runs with no goal in them.
  // A cell outside `open` counts as a wall for box movement.
  const dead = new Set<number>();
  const isWall = (c: number): boolean => c < 0 || !open.has(c);
  for (const c of cells) {
    if (goalSet.has(c)) continue;
    const wUp = isWall(step(lv, c, 0));
    const wDown = isWall(step(lv, c, 1));
    const wLeft = isWall(step(lv, c, 2));
    const wRight = isWall(step(lv, c, 3));
    if ((wUp && wLeft) || (wUp && wRight) || (wDown && wLeft) || (wDown && wRight)) dead.add(c);
  }
  // Wall runs: maximal runs of cells sharing a wall on one side, bounded
  // by walls at both ends along the run and containing no goal. A box in
  // such a run can only slide along it and can never reach a goal.
  const markRuns = (
    lines: number[][],
    beside: (c: number) => number,
    along: (c: number, s: number) => number,
  ): void => {
    for (const line of lines) {
      let run: number[] = [];
      const flush = (): void => {
        if (run.length === 0) return;
        const before = along(run[0], -1);
        const after = along(run[run.length - 1], 1);
        if (isWall(before) && isWall(after) && !run.some((c) => goalSet.has(c))) {
          for (const c of run) dead.add(c);
        }
        run = [];
      };
      for (const c of line) {
        if (open.has(c) && isWall(beside(c))) run.push(c);
        else flush();
      }
      flush();
    }
  };
  const hLines: number[][] = [];
  for (let y = 0; y < lv.h; y++) {
    const line: number[] = [];
    for (let x = 0; x < lv.w; x++) line.push(y * lv.w + x);
    hLines.push(line);
  }
  const vLines: number[][] = [];
  for (let x = 0; x < lv.w; x++) {
    const line: number[] = [];
    for (let y = 0; y < lv.h; y++) line.push(y * lv.w + x);
    vLines.push(line);
  }
  const alongH = (c: number, s: number): number => step(lv, c, s < 0 ? 2 : 3);
  const alongV = (c: number, s: number): number => step(lv, c, s < 0 ? 0 : 1);
  markRuns(hLines, (c) => step(lv, c, 0), alongH); // wall above
  markRuns(hLines, (c) => step(lv, c, 1), alongH); // wall below
  markRuns(vLines, (c) => step(lv, c, 2), alongV); // wall left
  markRuns(vLines, (c) => step(lv, c, 3), alongV); // wall right
  // A room seals when every interior neighbour of its door cell is a goal.
  const inward = [0, 1, 2, 3]
    .map((d) => step(lv, door, d))
    .filter((c) => c >= 0 && comp.has(c));
  const seals = inward.length > 0 && inward.every((c) => goalSet.has(c));
  return { cells, open, door, entry, boxes, goals, goalSet, dist, dead, seals };
}

// Split the level into rooms. Returns null when the shape does not fit
// (caller falls back to whole-level solving).
function decompose(lv: Level): Room[] | null {
  const hallRow = (lv.player / lv.w) | 0;
  for (const b of lv.boxes) if (((b / lv.w) | 0) === hallRow) return null;
  const rest = new Set<number>();
  for (let i = 0; i < lv.w * lv.h; i++) {
    if (lv.floor[i] && ((i / lv.w) | 0) !== hallRow) rest.add(i);
  }
  const seen = new Set<number>();
  const rooms: Room[] = [];
  for (const c of rest) {
    if (seen.has(c)) continue;
    const comp = flood(c, rest, new Set<number>(), lv);
    for (const x of comp) seen.add(x);
    const boxes = lv.boxes.filter((b) => comp.has(b));
    const goals = lv.goals.filter((g) => comp.has(g));
    if (boxes.length === 0 && goals.length === 0) continue; // empty pocket
    const doors = new Set<number>();
    let entry = -1;
    for (const cell of comp) {
      for (let d = 0; d < 4; d++) {
        const nb = step(lv, cell, d);
        if (nb >= 0 && lv.floor[nb] && ((nb / lv.w) | 0) === hallRow) {
          doors.add(cell);
          if (entry < 0) entry = nb;
        }
      }
    }
    if (doors.size !== 1) return null;
    if (boxes.length !== goals.length || boxes.length < 1 || boxes.length > 4) return null;
    rooms.push(buildRoom(lv, comp, [...doors][0], entry));
  }
  if (rooms.length === 0) return null;
  return rooms;
}

// True when some box off-goal is frozen: every push of it is permanently
// blocked because the destination cell, or the cell the player must stand
// on to push, is a wall or an already-frozen box. Fixpoint over the boxes;
// goal boxes are never frozen (they need not move) and never counted here.
function hasFrozenBox(lv: Level, room: Room, boxes: number[]): boolean {
  const isWall = (c: number): boolean => c < 0 || !room.open.has(c);
  const frozen = new Set<number>();
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of boxes) {
      if (frozen.has(b) || room.goalSet.has(b)) continue;
      let stuck = true;
      for (let d = 0; d < 4; d++) {
        const dest = step(lv, b, d);
        const behind = step(lv, b, d ^ 1);
        if (!isWall(dest) && !frozen.has(dest) && !isWall(behind) && !frozen.has(behind)) {
          stuck = false;
          break;
        }
      }
      if (stuck) {
        frozen.add(b);
        changed = true;
      }
    }
  }
  return frozen.size > 0;
}

// ---- per-room push-optimal A* ----------------------------------------------

interface PushStep {
  box: number;
  dir: number;
}

function solveRoom(lv: Level, room: Room, needExit: boolean): PushStep[] | null {
  const n = room.boxes.length;
  // Admissible heuristic: cheapest assignment of boxes to goals on BFS
  // box distances (brute-force permutations, n <= 4). Infinity when some
  // box cannot reach any goal, which also prunes that state.
  const heur = (boxes: number[]): number => {
    let best = Infinity;
    const perm = room.goals.map((_, i) => i);
    const rec = (k: number, sum: number): void => {
      if (sum >= best) return;
      if (k === n) {
        best = sum;
        return;
      }
      for (let i = k; i < n; i++) {
        const t = perm[k];
        perm[k] = perm[i];
        perm[i] = t;
        const d = room.dist[boxes[k]][room.goals[perm[k]]];
        if (d !== Infinity) rec(k + 1, sum + d);
        const t2 = perm[k];
        perm[k] = perm[i];
        perm[i] = t2;
      }
    };
    rec(0, 0);
    return best;
  };

  const keyOf = (boxes: number[], player: number): string => `${boxes.join(',')}|${player}`;
  const startBoxes = [...room.boxes];
  const startPlayer = canonical(room.entry, new Set(startBoxes), room.open, lv);
  const h0 = heur(startBoxes);
  if (h0 === Infinity) return null;

  interface Node extends HeapNode {
    boxes: number[];
    player: number;
  }
  const openHeap = new Heap<Node>();
  const best = new Map<string, number>();
  const parent = new Map<string, { prev: string; push: PushStep }>();
  let seq = 0;
  const startKey = keyOf(startBoxes, startPlayer);
  best.set(startKey, 0);
  openHeap.push({ f: h0, h: h0, g: 0, seq: seq++, key: startKey, boxes: startBoxes, player: startPlayer });

  let expansions = 0;
  while (openHeap.size > 0) {
    const cur = openHeap.pop()!;
    if (best.get(cur.key) !== cur.g) continue; // stale entry
    if (cur.boxes.every((b) => room.goalSet.has(b))) {
      // A room solved while the player still has to leave must end with
      // the player able to reach the entry cell; a solved state that
      // traps the player is not accepted, the search carries on through
      // it (pushing a box off its goal) instead.
      const canExit =
        !needExit || flood(cur.player, room.open, new Set(cur.boxes), lv).has(room.entry);
      if (canExit) {
        const pushes: PushStep[] = [];
        let k = cur.key;
        while (parent.has(k)) {
          const p = parent.get(k)!;
          pushes.push(p.push);
          k = p.prev;
        }
        pushes.reverse();
        return pushes;
      }
    }
    if (++expansions > ROOM_NODE_BUDGET) return null;
    const boxSet = new Set(cur.boxes);
    const region = flood(cur.player, room.open, boxSet, lv);
    for (let bi = 0; bi < n; bi++) {
      const b = cur.boxes[bi];
      for (let d = 0; d < 4; d++) {
        const dest = step(lv, b, d);
        if (dest < 0 || !room.open.has(dest) || boxSet.has(dest)) continue;
        const behind = step(lv, b, d ^ 1);
        if (behind < 0 || !region.has(behind)) continue;
        // A box pushed onto the entry cell can never be pushed back.
        if (dest === room.entry || room.dead.has(dest)) continue;
        const nb = [...cur.boxes];
        nb[bi] = dest;
        nb.sort((a, b2) => a - b2);
        const h = heur(nb);
        if (h === Infinity) continue;
        // Freeze deadlock: a box whose every push is permanently blocked
        // (destination or the cell the player must stand on is a wall or
        // an already-frozen box) can never move again, so a state holding
        // one off-goal is dead. Computed as a fixpoint over the boxes.
        if (hasFrozenBox(lv, room, nb)) continue;
        const np = canonical(b, new Set(nb), room.open, lv);
        const key = keyOf(nb, np);
        const g = cur.g + 1;
        if (best.has(key) && best.get(key)! <= g) continue;
        best.set(key, g);
        parent.set(key, { prev: cur.key, push: { box: b, dir: d } });
        openHeap.push({ f: g + h, h, g, seq: seq++, key, boxes: nb, player: np });
      }
    }
  }
  return null;
}

// ---- bounded move-A* fallback ----------------------------------------------
// State = player + boxes, cost = moves, heuristic = sum of Manhattan
// distances to the nearest goal. Used when a room (or the whole level)
// cannot be handled by the decomposition.
function solveMoves(
  lv: Level,
  open: Set<number>,
  boxes0: number[],
  goalSet: Set<number>,
  player0: number,
  budget: number,
  forbidden: Set<number> = new Set(),
  exitCell = -1, // when >= 0, a solved state must let the player reach it
): string | null {
  const manhattan = (a: number, b: number): number =>
    Math.abs((a % lv.w) - (b % lv.w)) + Math.abs(((a / lv.w) | 0) - ((b / lv.w) | 0));
  const heur = (boxes: number[]): number => {
    let s = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (const g of goalSet) m = Math.min(m, manhattan(b, g));
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  };
  const keyOf = (boxes: number[], player: number): string => `${boxes.join(',')}|${player}`;
  const startBoxes = [...boxes0].sort((a, b) => a - b);
  const h0 = heur(startBoxes);
  if (h0 === Infinity) return null;

  interface Node extends HeapNode {
    boxes: number[];
    player: number;
  }
  const openHeap = new Heap<Node>();
  const best = new Map<string, number>();
  const parent = new Map<string, { prev: string; move: string }>();
  let seq = 0;
  const startKey = keyOf(startBoxes, player0);
  best.set(startKey, 0);
  openHeap.push({ f: h0, h: h0, g: 0, seq: seq++, key: startKey, boxes: startBoxes, player: player0 });

  let expansions = 0;
  while (openHeap.size > 0) {
    const cur = openHeap.pop()!;
    if (best.get(cur.key) !== cur.g) continue;
    if (cur.boxes.every((b) => goalSet.has(b))) {
      const canExit =
        exitCell < 0 || flood(cur.player, open, new Set(cur.boxes), lv).has(exitCell);
      if (canExit) {
        let moves = '';
        let k = cur.key;
        while (parent.has(k)) {
          const p = parent.get(k)!;
          moves = p.move + moves;
          k = p.prev;
        }
        return moves;
      }
    }
    if (++expansions > budget) return null;
    const boxSet = new Set(cur.boxes);
    for (let d = 0; d < 4; d++) {
      const q = step(lv, cur.player, d);
      if (q < 0 || !open.has(q)) continue;
      let nb = cur.boxes;
      let np = q;
      let move = LETTER[d];
      if (boxSet.has(q)) {
        const b2 = step(lv, q, d);
        if (b2 < 0 || !open.has(b2) || boxSet.has(b2) || forbidden.has(b2)) continue;
        nb = [...cur.boxes];
        nb[nb.indexOf(q)] = b2;
        nb.sort((a, b3) => a - b3);
        move = LETTER[d].toUpperCase();
      }
      const h = heur(nb);
      if (h === Infinity) continue;
      const key = keyOf(nb, np);
      const g = cur.g + 1;
      if (best.has(key) && best.get(key)! <= g) continue;
      best.set(key, g);
      parent.set(key, { prev: cur.key, move });
      openHeap.push({ f: g + h, h, g, seq: seq++, key, boxes: nb, player: np });
    }
  }
  return null;
}

// ---- stitching ---------------------------------------------------------------

// Shortest player walk from `from` to `to` (boxes are obstacles), or null.
function bfsPath(lv: Level, from: number, to: number, boxSet: Set<number>): string | null {
  if (from === to) return '';
  const prev = new Map<number, { from: number; ch: string }>();
  const q = [from];
  prev.set(from, { from: -1, ch: '' });
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi];
    for (let d = 0; d < 4; d++) {
      const n = step(lv, c, d);
      if (n < 0 || !lv.floor[n] || boxSet.has(n) || prev.has(n)) continue;
      prev.set(n, { from: c, ch: LETTER[d] });
      if (n === to) {
        let s = '';
        let cur = n;
        while (cur !== from) {
          const p = prev.get(cur)!;
          s = p.ch + s;
          cur = p.from;
        }
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

// Apply a move string to (player, boxSet) on the full board; returns the
// new player cell. Assumes the moves are legal.
function applyMoves(lv: Level, player: number, boxSet: Set<number>, moves: string): number {
  for (const m of moves) {
    const d = 'udlr'.indexOf(m.toLowerCase());
    if (d < 0) continue;
    const q = step(lv, player, d);
    if (q < 0) continue;
    if (boxSet.has(q)) {
      boxSet.delete(q);
      boxSet.add(step(lv, q, d));
    }
    player = q;
  }
  return player;
}

function solveLevel(lv: Level): string {
  const rooms = decompose(lv);
  if (rooms === null) {
    const allFloor = new Set<number>();
    for (let i = 0; i < lv.w * lv.h; i++) if (lv.floor[i]) allFloor.add(i);
    return solveMoves(lv, allFloor, lv.boxes, lv.goalSet, lv.player, MOVE_NODE_BUDGET) ?? '';
  }
  // Rooms that seal themselves (a box ends on every interior neighbour of
  // the door) go last: the player cannot leave them once solved. The rest
  // go left to right by door column. Every room but the last must end with
  // the player able to walk out.
  rooms.sort((a, b) => {
    if (a.seals !== b.seals) return a.seals ? 1 : -1;
    return (a.door % lv.w) - (b.door % lv.w);
  });
  const boxSet = new Set<number>(lv.boxes);
  let player = lv.player;
  let out = '';
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri];
    const needExit = ri < rooms.length - 1;
    const walk = bfsPath(lv, player, room.entry, boxSet);
    if (walk === null) return out;
    out += walk;
    player = room.entry;
    const plan = solveRoom(lv, room, needExit);
    if (plan !== null) {
      for (const p of plan) {
        const behind = step(lv, p.box, p.dir ^ 1);
        const w2 = bfsPath(lv, player, behind, boxSet);
        if (w2 === null) return out;
        out += w2;
        out += LETTER[p.dir].toUpperCase();
        boxSet.delete(p.box);
        boxSet.add(step(lv, p.box, p.dir));
        player = p.box;
      }
    } else {
      const m = solveMoves(
        lv,
        room.open,
        room.boxes,
        room.goalSet,
        room.entry,
        MOVE_NODE_BUDGET,
        new Set([room.entry]),
        needExit ? room.entry : -1,
      );
      if (m !== null) {
        out += m;
        player = applyMoves(lv, player, boxSet, m);
      }
      // Room stays unsolved; keep the output a legal move string.
    }
  }
  return out;
}

// ---- main --------------------------------------------------------------------

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    const lv = parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
    moves = solveLevel(lv);
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
