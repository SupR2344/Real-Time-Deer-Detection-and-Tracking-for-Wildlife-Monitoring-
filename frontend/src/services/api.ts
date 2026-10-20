// -*- coding: utf-8 -*-
// Single authoritative API client for DeerGuard AI backend

import {
  TracksResponse,
  RuntimeStatus,
  RoadZoneResponse,
  SourceResponse,
  AvailableSourcesResponse
} from '../types';

export const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');

export const api = {
  async getHealth(): Promise<{ status: string; uptime: number }> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return res.json();
  },

  async getTracks(): Promise<TracksResponse> {
    const res = await fetch(`${API_BASE}/tracks`);
    if (!res.ok) throw new Error('Failed to fetch tracks');
    return res.json();
  },

  async getRuntime(): Promise<RuntimeStatus> {
    const res = await fetch(`${API_BASE}/runtime`);
    if (!res.ok) throw new Error('Failed to fetch runtime status');
    return res.json();
  },

  async getRoadZone(): Promise<RoadZoneResponse> {
    const res = await fetch(`${API_BASE}/road-zone`);
    if (!res.ok) throw new Error('Failed to fetch road zone');
    return res.json();
  },

  async setRoadZone(points: [number, number][], frame_width: number, frame_height: number): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/road-zone`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ points, frame_width, frame_height }),
    });
    if (!res.ok) throw new Error('Failed to save road zone');
    return res.json();
  },

  async deleteRoadZone(): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/road-zone`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to clear road zone');
    return res.json();
  },

  async getSource(): Promise<SourceResponse> {
    const res = await fetch(`${API_BASE}/source`);
    if (!res.ok) throw new Error('Failed to get source');
    return res.json();
  },

  async getAvailableSources(): Promise<AvailableSourcesResponse> {
    const res = await fetch(`${API_BASE}/source/available`);
    if (!res.ok) throw new Error('Failed to get available sources');
    return res.json();
  },

  async configureSource(type: string, source: string): Promise<{ status: string; type: string; source: string }> {
    const res = await fetch(`${API_BASE}/source`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type, source }),
    });
    if (!res.ok) throw new Error('Failed to configure source');
    return res.json();
  },

  async stopSource(): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/source/stop`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to stop source');
    return res.json();
  },

  async resetSystem(): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/reset`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to reset system');
    return res.json();
  },
};
