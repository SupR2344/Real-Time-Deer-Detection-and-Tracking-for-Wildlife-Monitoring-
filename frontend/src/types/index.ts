// -*- coding: utf-8 -*-
// Typed TypeScript interfaces for DeerGuard AI

export type RiskLevel = 'SAFE' | 'CAUTION' | 'HIGH_RISK' | 'UNKNOWN';

export interface Track {
  track_id: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
  class_id: number;
  class_name: string;
  risk: RiskLevel;
  speed: number;
  position: [number, number];            // [cx, cy]
  predicted: [number, number];           // [px, py]
  trajectory: [number, number, number][]; // [[frame_id, x, y], ...]
  caption?: string | null;
}

export interface TracksResponse {
  frame_id: number;
  detection_count: number;
  deer_detected: boolean;
  tracks: Track[];
}

export interface RuntimeStatus {
  system_online: boolean;
  backend_connected: boolean;
  uptime: number;
  fps: number;
  frame_id: number;
  gpu: string;
  active_tracks: number;
  high_risk_count: number;
  road_zone_configured: boolean;
  road_zone_points: [number, number][];
  vlm_status: string;
  vlm_captions: { track_id: number; caption: string }[];
  source: {
    type: string | null;
    source: string | null;
    status?: string | null;
  };
  components: {
    camera: string;
    detection: string;
    tracking: string;
    prediction: string;
    collision_engine: string;
    vlm: string;
  };
  vlm_triggered: boolean;
}

export interface RoadZoneResponse {
  points: [number, number][];
  frame_width: number;
  frame_height: number;
  configured: boolean;
}

export interface SourceResponse {
  type: string | null;
  source: string | null;
  status: string;
}

export interface WebcamDevice {
  index: number;
  label: string;
  resolution: string;
  kind: 'integrated' | 'webcam';
}

export interface VideoFile {
  name: string;
  path: string;
}

export interface AvailableSourcesResponse {
  webcams: WebcamDevice[];
  videos: VideoFile[];
  supported_types: {
    id: string;
    name: string;
    default_val: string;
  }[];
}

export interface EventLog {
  time: string;
  message: string;
  level: 'info' | 'caution' | 'danger' | 'success';
}
