export interface SceneLabels {
  agent: {
    x: number;
    y: number;
    health: number;
    health_status: string;
    alive: boolean;
    evacuated: boolean;
    location: string;
    smoke_level: string;
    fire_visible: boolean;
    fire_direction: string | null;
    last_action_feedback: string;
  };
  episode: {
    id: string;
    step: number;
    max_steps: number;
    template: string;
    difficulty: string;
    wind_dir: string;
    fire_spread_rate: number;
    humidity: number;
    fire_sources: number;
  };
  map: {
    width: number;
    height: number;
    exit_positions: [number, number][];
    door_registry: Record<string, [number, number]>;
  };
  surroundings: {
    visible_objects: {
      id: string;
      type: string;
      relative_pos: string;
      state: string;
    }[];
    blocked_exit_ids: string[];
    audible_signals: string[];
    available_actions: string[];
  };
}

export interface SceneGraph {
  channels: string[];
  channel_info: Record<string, string>;
  width: number;
  height: number;
  grid: number[][][]; // grid[y][x] = [cell_type, fire, smoke, is_agent, is_visible]
}

export interface SceneResponse {
  labels: SceneLabels;
  graph: SceneGraph;
}

export interface Observation {
  map_state: {
    cell_grid: number[];
    fire_grid: number[];
    smoke_grid: number[];
    agent_x: number;
    agent_y: number;
    visible_cells: [number, number][];
    door_registry: Record<string, [number, number]>;
    exit_positions: [number, number][];
    step_count: number;
    max_steps: number;
    grid_w: number;
    grid_h: number;
    template_name: string;
  };
  agent_health: number;
  location_label: string;
  smoke_level: string;
  wind_dir: string;
  fire_visible: boolean;
  fire_direction: string | null;
  last_action_feedback: string;
  narrative: string;
  reward?: number;
  done?: boolean;
  metadata?: {
    fire_sources: number;
    fire_spread_rate: number;
    humidity: number;
    difficulty: string;
  };
}

export interface Door {
  id: string;
  x: number;
  y: number;
  state: 'open' | 'closed' | 'failed';
}

export interface StaffMember {
  id: string;
  x: number;
  y: number;
  phase: number;
  mood: 'calm' | 'panicked';
}

export interface ApiReport {
  call_type: string;
  request: any;
  response: any;
  meta: string;
}
