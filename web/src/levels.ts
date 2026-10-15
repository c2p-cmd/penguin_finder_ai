/** One playable stage. `room` is a MagicaVoxel room in assets/rooms; null = the flat graveyard. */
export interface LevelSpec {
  name: string;
  tagline: string;
  room: string | null;
  ghosts: number; // at the start
  maxGhosts: number;
  speed: [number, number]; // ghost speed, in flat-graveyard units per second
  spawnEvery: number; // base seconds between new ghosts
  ambient: [number, number, number]; // moonlight tint (linear)
}

export const LEVELS: LevelSpec[] = [
  {
    name: 'Moonlit Graveyard', tagline: 'Warm up among the graves', room: null,
    ghosts: 4, maxGhosts: 12, speed: [80, 170], spawnEvery: 2.6, ambient: [0.075, 0.105, 0.2],
  },
  {
    name: 'The Overgrown Garden', tagline: 'Vines, a fountain and a hole in the hedge', room: 'garden',
    ghosts: 4, maxGhosts: 10, speed: [80, 160], spawnEvery: 2.8, ambient: [0.06, 0.1, 0.15],
  },
  {
    name: 'Dungeon Stairs', tagline: 'Torchlight on the old steps', room: 'stairs',
    ghosts: 5, maxGhosts: 11, speed: [90, 170], spawnEvery: 2.6, ambient: [0.07, 0.07, 0.16],
  },
  {
    name: 'Boss Room', tagline: 'Something big used to live here', room: 'boss',
    ghosts: 6, maxGhosts: 12, speed: [100, 180], spawnEvery: 2.4, ambient: [0.09, 0.07, 0.15],
  },
  {
    name: 'Trapped Room', tagline: 'The floor is lava. Mostly.', room: 'trapped',
    ghosts: 6, maxGhosts: 13, speed: [100, 190], spawnEvery: 2.2, ambient: [0.1, 0.06, 0.12],
  },
  {
    name: "King's Throne", tagline: 'The court is still in session', room: 'throne',
    ghosts: 7, maxGhosts: 14, speed: [110, 200], spawnEvery: 2.0, ambient: [0.1, 0.06, 0.13],
  },
  {
    name: 'Dragon Room', tagline: 'Last stop. Mind the river.', room: 'dragon',
    ghosts: 8, maxGhosts: 15, speed: [120, 215], spawnEvery: 1.8, ambient: [0.06, 0.08, 0.17],
  },
];
