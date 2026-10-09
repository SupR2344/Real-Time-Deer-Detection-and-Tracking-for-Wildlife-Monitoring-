import React, { useEffect, useState, useRef, useCallback, MouseEvent } from 'react';
import './App.css';
import { Track, RuntimeStatus, EventLog, WebcamDevice, VideoFile } from './types';
import { api } from './services/api';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function riskLabel(risk: string) {
  switch (risk) {
    case 'HIGH_RISK': return 'HIGH RISK';
    case 'CAUTION':   return 'CAUTION';
    case 'SAFE':      return 'SAFE';
    default:          return 'UNKNOWN';
  }
}

function formatTime(date: Date) {
  return date.toTimeString().slice(0, 8);
}

function formatDate(date: Date) {
  return date.toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' });
}

function getPosition(bbox: [number, number, number, number]) {
  const cx = Math.round((bbox[0] + bbox[2]) / 2);
  const cy = Math.round((bbox[1] + bbox[3]) / 2);
  return `(${cx}, ${cy})`;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

const RiskBadge = ({ risk }: { risk: string }) => (
  <span className={`risk-badge risk-${risk.toLowerCase().replace('_', '-')}`}>
    {riskLabel(risk)}
  </span>
);

const StatusDot = ({ status }: { status: string }) => {
  const active = status === 'ACTIVE' || status === 'ONLINE' || status === 'RUNNING' || status === 'STREAMING';
  const standby = status === 'STANDBY' || status === 'IDLE';
  return (
    <span className={`status-pill ${active ? 'pill-active' : standby ? 'pill-standby' : 'pill-inactive'}`}>
      {status}
    </span>
  );
};

const ComponentRow = ({ icon, label, status }: { icon: string; label: string; status: string }) => (
  <div className="component-row">
    <span className="component-icon">{icon}</span>
    <span className="component-label">{label}</span>
    <StatusDot status={status} />
  </div>
);

// ─── Trajectory Canvas (Uses Real Backend Points) ─────────────────────────────

function TrajCanvas({ track, frameWidth = 640, frameHeight = 480 }: { track: Track; frameWidth?: number; frameHeight?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const traj = track.trajectory || [];
    if (traj.length === 0) {
      ctx.fillStyle = '#6b8ab0';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Trajectory unavailable', W / 2, H / 2);
      return;
    }

    // Determine bounding box of trajectory to normalize or scale relative to frame
    const fw = frameWidth > 0 ? frameWidth : 640;
    const fh = frameHeight > 0 ? frameHeight : 480;

    const sx = (x: number) => (x / fw) * W;
    const sy = (y: number) => (y / fh) * H;

    // Draw past trajectory line
    if (traj.length >= 2) {
      ctx.strokeStyle = '#2ecc71';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx(traj[0][1]), sy(traj[0][2]));
      for (let i = 1; i < traj.length; i++) {
        ctx.lineTo(sx(traj[i][1]), sy(traj[i][2]));
      }
      ctx.stroke();

      // Draw past points
      ctx.fillStyle = '#2ecc71';
      for (const pt of traj) {
        ctx.beginPath();
        ctx.arc(sx(pt[1]), sy(pt[2]), 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Draw current position
    const curX = sx(track.position[0]);
    const curY = sy(track.position[1]);
    ctx.fillStyle = '#e74c3c';
    ctx.beginPath();
    ctx.arc(curX, curY, 4.5, 0, Math.PI * 2);
    ctx.fill();

    // Draw predicted line & future point
    if (track.predicted && (track.predicted[0] !== 0 || track.predicted[1] !== 0)) {
      const predX = sx(track.predicted[0]);
      const predY = sy(track.predicted[1]);

      ctx.strokeStyle = '#3498db';
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(curX, curY);
      ctx.lineTo(predX, predY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#3498db';
      ctx.beginPath();
      ctx.arc(predX, predY, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [track, frameWidth, frameHeight]);

  return <canvas ref={canvasRef} width={260} height={120} className="traj-canvas-el" />;
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [selectedSourceTab, setSelectedSourceTab] = useState<'integrated' | 'webcam' | 'rtsp' | 'video'>('video');
  const [availableWebcams, setAvailableWebcams] = useState<WebcamDevice[]>([]);
  const [availableVideos, setAvailableVideos] = useState<VideoFile[]>([]);
  const [rtspUrl, setRtspUrl] = useState('');
  const [webcamIndex, setWebcamIndex] = useState('1');
  const [videoPath, setVideoPath] = useState('13297686_3840_2160_60fps.mp4');
  const [isSwitchingSource, setIsSwitchingSource] = useState(false);

  const [isConfiguringZone, setIsConfiguringZone] = useState(false);
  const [zonePoints, setZonePoints] = useState<[number, number][]>([]);

  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [selectedTrack, setSelectedTrack] = useState<Track | null>(null);
  const [events, setEvents] = useState<EventLog[]>([]);
  const [now, setNow] = useState(new Date());
  const [backendOnline, setBackendOnline] = useState(false);
  const prevTracksRef = useRef<Track[]>([]);
  const streamRef = useRef<HTMLImageElement>(null);

  // Clock tick
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // Add event helper
  const addEvent = useCallback((msg: string, level: EventLog['level']) => {
    setEvents(prev => [{
      time: formatTime(new Date()),
      message: msg,
      level,
    }, ...prev].slice(0, 30));
  }, []);

  // Fetch available camera sources on mount
  useEffect(() => {
    let mounted = true;
    const loadSources = async () => {
      try {
        const data = await api.getAvailableSources();
        if (!mounted) return;
        setAvailableWebcams(data.webcams || []);
        setAvailableVideos(data.videos || []);
        const externalCamera = (data.webcams || []).find(cam => cam.kind === 'webcam');
        if (externalCamera) setWebcamIndex(String(externalCamera.index));
        if (data.videos && data.videos.length > 0) {
          setVideoPath(data.videos[0].path);
        }
      } catch (err) {
        console.warn('Could not query available sources:', err);
      }
    };
    loadSources();
    return () => { mounted = false; };
  }, []);

  // Poll /tracks every 1000ms
  useEffect(() => {
    let mounted = true;
    const pollTracks = async () => {
      try {
        const data = await api.getTracks();
        if (!mounted) return;
        const newTracks: Track[] = data.tracks || [];
        setTracks(newTracks);
        setBackendOnline(true);

        // Diff tracks to generate events
        const prev = prevTracksRef.current;
        const prevIds = new Set(prev.map(t => t.track_id));
        const newIds = new Set(newTracks.map(t => t.track_id));

        newTracks.forEach(t => {
          if (!prevIds.has(t.track_id)) {
            addEvent(`New track #${t.track_id} detected (${t.class_name})`, 'info');
          }
        });
        prev.forEach(p => {
          if (!newIds.has(p.track_id)) {
            addEvent(`Track #${p.track_id} lost`, 'info');
          }
        });
        newTracks.forEach(t => {
          const old = prev.find(p => p.track_id === t.track_id);
          if (old && old.risk !== t.risk) {
            if (t.risk === 'HIGH_RISK') addEvent(`Track #${t.track_id} entered HIGH RISK status!`, 'danger');
            else if (t.risk === 'CAUTION') addEvent(`Track #${t.track_id} status changed to CAUTION`, 'caution');
            else if (t.risk === 'SAFE') addEvent(`Track #${t.track_id} returned to SAFE status`, 'success');
          }
          if (t.caption && (!old || old.caption !== t.caption)) {
            addEvent(`VLM (BLIP) generated caption for track #${t.track_id}: "${t.caption}"`, 'success');
          }
        });

        prevTracksRef.current = newTracks;

        // Auto-select first track if none selected or keep updated
        setSelectedTrack(sel => {
          if (!sel && newTracks.length > 0) return newTracks[0];
          if (sel) {
            const updated = newTracks.find(t => t.track_id === sel.track_id);
            return updated ?? (newTracks[0] ?? null);
          }
          return null;
        });
      } catch {
        if (mounted) setBackendOnline(false);
      }
    };

    pollTracks();
    const iv = setInterval(pollTracks, 1000);
    return () => {
      mounted = false;
      clearInterval(iv);
    };
  }, [addEvent]);

  // Poll /runtime every 1500ms
  useEffect(() => {
    let mounted = true;
    const pollRuntime = async () => {
      try {
        const data = await api.getRuntime();
        if (!mounted) return;
        setStatus(data);
        if (data.source && data.source.source && data.source.type) {
          const st = data.source.type.toLowerCase();
          if (st.includes('integrated')) setSelectedSourceTab('integrated');
          else if (st.includes('webcam') || st.includes('usb')) setSelectedSourceTab('webcam');
          else if (st.includes('rtsp') || st.includes('ip')) setSelectedSourceTab('rtsp');
          else setSelectedSourceTab('video');
        }
      } catch {}
    };
    pollRuntime();
    const iv = setInterval(pollRuntime, 1500);
    return () => {
      mounted = false;
      clearInterval(iv);
    };
  }, []);

  // Initial event
  useEffect(() => {
    addEvent('DeerGuard AI dashboard connected', 'info');
  }, [addEvent]);

  const highRiskCount = status?.high_risk_count ?? tracks.filter(t => t.risk === 'HIGH_RISK').length;
  const fps = status?.fps ?? 0;
  const frameId = status?.frame_id ?? 0;
  const hasLiveFrame = backendOnline && frameId >= 0 && status?.components.camera === 'STREAMING';
  const deerDetected = (status?.detection_count ?? 0) > 0;

  const handleConnectSource = async (type: string, sourceValue: string) => {
    setIsSwitchingSource(true);
    try {
      addEvent(`Requesting source switch to [${type}]: ${sourceValue}`, 'info');
      await api.configureSource(type, sourceValue);
      addEvent(`Source [${type}] configured — connecting...`, 'info');
      // Promptly refresh status
      setTimeout(async () => {
        try {
          const rt = await api.getRuntime();
          setStatus(rt);
        } catch {}
      }, 800);
    } catch (e: any) {
      addEvent(`Failed to switch source: ${e.message || e}`, 'danger');
    } finally {
      setIsSwitchingSource(false);
    }
  };

  const handleStopSource = async () => {
    setIsSwitchingSource(true);
    try {
      await api.stopSource();
      addEvent('Camera source stopped by user', 'info');
      setTimeout(async () => {
        try {
          const rt = await api.getRuntime();
          setStatus(rt);
        } catch {}
      }, 500);
    } catch (e: any) {
      addEvent(`Failed to stop source: ${e.message || e}`, 'danger');
    } finally {
      setIsSwitchingSource(false);
    }
  };

  const handleReset = async () => {
    if (!window.confirm('Are you sure you want to reset system state and tracking history?')) return;
    try {
      await api.resetSystem();
      addEvent('System reset initiated', 'info');
      setTracks([]);
      setSelectedTrack(null);
      prevTracksRef.current = [];
    } catch (e) {
      addEvent(`Failed to reset: ${e}`, 'danger');
    }
  };

  const handleSaveRoadZone = async () => {
    if (zonePoints.length < 3) {
      alert('At least 3 points required to define a road zone polygon.');
      return;
    }
    try {
      const nw = streamRef.current?.naturalWidth || 1280;
      const nh = streamRef.current?.naturalHeight || 720;
      await api.setRoadZone(zonePoints, nw, nh);
      setIsConfiguringZone(false);
      addEvent(`Road zone polygon saved (${zonePoints.length} vertices)`, 'success');
    } catch (e) {
      alert(`Error saving road zone: ${e}`);
    }
  };

  const handleDeleteRoadZone = async () => {
    if (!window.confirm('Clear and delete configured road zone?')) return;
    try {
      await api.deleteRoadZone();
      setZonePoints([]);
      setIsConfiguringZone(false);
      addEvent('Road zone cleared', 'info');
    } catch (e) {
      alert(`Error clearing road zone: ${e}`);
    }
  };

  return (
    <div className="dg-root">
      {/* ── TOP HEADER ── */}
      <header className="dg-header">
        <div className="dg-header-brand">
          <div className="dg-logo">
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
              <circle cx="18" cy="18" r="18" fill="#1a2535" />
              <text x="18" y="24" textAnchor="middle" fontSize="18" fill="#e8c56a">🦌</text>
            </svg>
          </div>
          <div className="dg-brand-text">
            <span className="brand-deer">DEER</span><span className="brand-guard">GUARD</span> <span className="brand-ai">AI</span>
            <div className="brand-sub">AI Deer Detection • Tracking • Trajectory Prediction • Collision Warning System</div>
          </div>
        </div>

        <div className="dg-header-status">
          <div className={`status-indicator ${backendOnline ? 'online' : 'offline'}`}>
            <span className="indicator-dot" />
            {backendOnline ? 'SYSTEM ONLINE' : 'SYSTEM OFFLINE'}
          </div>
          <div className="header-chip">
            <span className="chip-icon">⚙</span> Backend {backendOnline ? 'Connected' : 'Disconnected'}
          </div>
          <div className="header-chip">
            <span className="chip-icon">🖥</span> GPU: {status?.gpu || 'Detecting...'}
          </div>
          <div className="header-clock">
            <span className="chip-icon">🕐</span>
            <div className="clock-block">
              <div className="clock-date">{formatDate(now)}</div>
              <div className="clock-time">{formatTime(now)}</div>
            </div>
          </div>
          <button className="reset-btn" onClick={handleReset}>↺ Reset System</button>
        </div>
      </header>

      {/* ── MAIN WORKSPACE ── */}
      <main className="dg-main">
        {/* LEFT COLUMN */}
        <div className="dg-left">
          {/* ── LIVE CAMERAS / SOURCE CONTROL ── */}
          <section className="panel source-panel">
            <div className="panel-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>📹 LIVE CAMERAS / SOURCE CONTROL</span>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Unified Backend MJPEG Pipeline Feed</span>
            </div>

            <div className="source-section">
              {/* 4 Source Category Selector Tabs */}
              <div className="source-tabs">
                <button
                  className={`source-tab-btn ${selectedSourceTab === 'integrated' ? 'active' : ''}`}
                  onClick={() => setSelectedSourceTab('integrated')}
                >
                  📷 Integrated Camera
                </button>
                <button
                  className={`source-tab-btn ${selectedSourceTab === 'webcam' ? 'active' : ''}`}
                  onClick={() => setSelectedSourceTab('webcam')}
                >
                  🔌 USB / Webcam
                </button>
                <button
                  className={`source-tab-btn ${selectedSourceTab === 'rtsp' ? 'active' : ''}`}
                  onClick={() => setSelectedSourceTab('rtsp')}
                >
                  🌐 IP Camera (RTSP)
                </button>
                <button
                  className={`source-tab-btn ${selectedSourceTab === 'video' ? 'active' : ''}`}
                  onClick={() => setSelectedSourceTab('video')}
                >
                  📁 Local Video File
                </button>
              </div>

              {/* Source Configuration Controls */}
              <div className="source-config-row">
                {selectedSourceTab === 'integrated' && (
                  <>
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '3px' }}>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                        {availableWebcams.some(w => w.index === 0)
                          ? `Hardware Detected: ${availableWebcams.find(w => w.index === 0)?.label}`
                          : 'OpenCV device index 0 (availability unverified)'}
                      </div>
                      <input
                        type="text"
                        value="0"
                        readOnly
                        className="source-input-field"
                        style={{ background: '#0a101d', color: '#8aa2c0', cursor: 'default' }}
                      />
                    </div>
                    <button
                      className="source-action-btn"
                      disabled={isSwitchingSource}
                      onClick={() => handleConnectSource('integrated', '0')}
                    >
                      {isSwitchingSource ? 'Connecting...' : 'Connect Integrated Camera'}
                    </button>
                  </>
                )}

                {selectedSourceTab === 'webcam' && (
                  <>
                    <div style={{ flex: 1, display: 'flex', gap: '8px', alignItems: 'center' }}>
                        {availableWebcams.some(cam => cam.kind === 'webcam') && (
                        <select
                          className="source-select-field"
                          style={{ flex: 1 }}
                          value={webcamIndex}
                          onChange={e => setWebcamIndex(e.target.value)}
                        >
                          {availableWebcams.filter(cam => cam.kind === 'webcam').map(cam => (
                            <option key={cam.index} value={String(cam.index)}>
                              {cam.label}
                            </option>
                          ))}
                        </select>
                      )}
                      <input
                        type="text"
                        placeholder="Device index (e.g. 0, 1, 2)"
                        value={webcamIndex}
                        onChange={e => setWebcamIndex(e.target.value)}
                        className="source-input-field"
                        style={{ maxWidth: '160px' }}
                      />
                    </div>
                    <button
                      className="source-action-btn"
                      disabled={isSwitchingSource}
                      onClick={() => handleConnectSource('webcam', webcamIndex)}
                    >
                      {isSwitchingSource ? 'Connecting...' : 'Connect Webcam'}
                    </button>
                  </>
                )}

                {selectedSourceTab === 'rtsp' && (
                  <>
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      <input
                        type="text"
                        placeholder="rtsp://camera-host:554/stream"
                        value={rtspUrl}
                        onChange={e => setRtspUrl(e.target.value)}
                        className="source-input-field"
                        autoComplete="off"
                      />
                      <span className="source-info-subtext">
                        🔒 Backend terminates RTSP stream directly; browser receives processed live MJPEG pipeline feed.
                      </span>
                    </div>
                    <button
                      className="source-action-btn"
                      disabled={isSwitchingSource || !rtspUrl.trim()}
                      onClick={() => handleConnectSource('rtsp', rtspUrl)}
                    >
                      {isSwitchingSource ? 'Connecting...' : 'Connect IP Camera'}
                    </button>
                  </>
                )}

                {selectedSourceTab === 'video' && (
                  <>
                    <div style={{ flex: 1, display: 'flex', gap: '8px', alignItems: 'center' }}>
                      {availableVideos.length > 0 && (
                        <select
                          className="source-select-field"
                          style={{ flex: 1 }}
                          value={videoPath}
                          onChange={e => setVideoPath(e.target.value)}
                        >
                          {availableVideos.map(v => (
                            <option key={v.path} value={v.path}>
                              {v.name}
                            </option>
                          ))}
                        </select>
                      )}
                      <input
                        type="text"
                        placeholder="Video path (e.g. 13297686_3840_2160_60fps.mp4)"
                        value={videoPath}
                        onChange={e => setVideoPath(e.target.value)}
                        className="source-input-field"
                      />
                    </div>
                    <button
                      className="source-action-btn"
                      disabled={isSwitchingSource}
                      onClick={() => handleConnectSource('video', videoPath)}
                    >
                      {isSwitchingSource ? 'Loading...' : 'Load & Run Video'}
                    </button>
                  </>
                )}

                <button
                  className="source-stop-btn"
                  disabled={isSwitchingSource || (status?.components.camera === 'DISCONNECTED')}
                  onClick={handleStopSource}
                >
                  Stop Source
                </button>
              </div>

              {/* Real-Time Live Status Strip */}
              <div className="source-status-strip">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Source:</span>
                  <strong style={{ color: '#fff' }}>
                    {status?.source?.type ? status.source.type.toUpperCase() : 'NONE'}
                  </strong>
                  <span style={{ color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                    {status?.source?.source ? `(${status.source.source})` : ''}
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                  {/* Connection Status Badge */}
                  <span className={`source-status-badge ${(status?.components.camera || 'DISCONNECTED').toLowerCase()}`}>
                    ● {status?.components.camera || 'DISCONNECTED'}
                  </span>

                  {/* Deer Detected Status */}
                  {hasLiveFrame && deerDetected ? (
                    <span style={{ background: 'rgba(231,76,60,0.2)', color: 'var(--red)', border: '1px solid var(--red)', padding: '2px 8px', borderRadius: '10px', fontSize: '10px', fontWeight: 700 }}>
                      🦌 DEER DETECTED ({status?.detection_count})
                    </span>
                  ) : hasLiveFrame ? (
                    <span style={{ background: 'rgba(46,204,113,0.15)', color: 'var(--green)', border: '1px solid var(--green)', padding: '2px 8px', borderRadius: '10px', fontSize: '10px', fontWeight: 600 }}>
                      NO DEER DETECTED
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-muted)' }}>WAITING FOR LIVE FRAME</span>
                  )}

                  {/* Highest Risk */}
                  <span style={{ color: 'var(--text-muted)' }}>
                    Risk: <RiskBadge risk={highRiskCount > 0 ? 'HIGH_RISK' : tracks.length > 0 ? (tracks[0]?.risk || 'SAFE') : 'SAFE'} />
                  </span>

                  {/* Telemetry FPS & Frame */}
                  <span style={{ color: 'var(--text-muted)' }}>
                    FPS: <strong style={{ color: '#fff' }}>{backendOnline && fps > 0 ? fps.toFixed(1) : '--'}</strong>
                  </span>
                  <span style={{ color: 'var(--text-muted)' }}>
                    Frame: <strong style={{ color: '#fff' }}>{backendOnline && frameId >= 0 ? frameId : '--'}</strong>
                  </span>
                </div>
              </div>
            </div>
          </section>

          {/* Video Stream & Road Zone Canvas */}
          <section className="panel camera-panel">
            <div className="camera-header">
              <span className="panel-label">📷 LIVE CAMERA FEED <span className="annotated-tag">(Real-Time AI Pipeline)</span></span>
              <div className="camera-meta">
                <span>FPS: {backendOnline && fps > 0 ? fps.toFixed(1) : '--'}</span>
                <span>Frame: {backendOnline && frameId >= 0 ? frameId : '--'}</span>
              </div>
            </div>

            <div className="camera-body" style={{ position: 'relative', overflow: 'hidden' }}>
              <img
                ref={streamRef}
                src="/api/video/stream"
                alt="Live annotated stream"
                className="camera-stream"
                onError={() => setBackendOnline(false)}
                style={{ display: 'block', width: '100%', height: 'auto', minHeight: '320px', objectFit: 'contain' }}
              />
              <div className="live-badge">● {status?.components.camera || 'DISCONNECTED'}</div>

              {/* Interactive Zone Editor Overlay */}
              {isConfiguringZone && (
                <div
                  style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', cursor: 'crosshair', zIndex: 10 }}
                  onClick={(e: MouseEvent<HTMLDivElement>) => {
                    if (!streamRef.current) return;
                    const rect = streamRef.current.getBoundingClientRect();
                    const nw = streamRef.current.naturalWidth || 1280;
                    const nh = streamRef.current.naturalHeight || 720;
                    const scaleX = nw / rect.width;
                    const scaleY = nh / rect.height;
                    const x = (e.clientX - rect.left) * scaleX;
                    const y = (e.clientY - rect.top) * scaleY;
                    setZonePoints(prev => [...prev, [Math.round(x), Math.round(y)]]);
                  }}
                >
                  <svg width="100%" height="100%" style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}>
                    {zonePoints.length > 0 && (
                      <polygon
                        points={zonePoints.map(p => {
                          if (!streamRef.current) return '0,0';
                          const rect = streamRef.current.getBoundingClientRect();
                          const nw = streamRef.current.naturalWidth || 1280;
                          const nh = streamRef.current.naturalHeight || 720;
                          const px = p[0] / (nw / rect.width);
                          const py = p[1] / (nh / rect.height);
                          return `${px},${py}`;
                        }).join(' ')}
                        fill="rgba(240, 200, 50, 0.35)"
                        stroke="#f0c832"
                        strokeWidth="3"
                      />
                    )}
                    {zonePoints.map((p, i) => {
                      if (!streamRef.current) return null;
                      const rect = streamRef.current.getBoundingClientRect();
                      const nw = streamRef.current.naturalWidth || 1280;
                      const nh = streamRef.current.naturalHeight || 720;
                      const px = p[0] / (nw / rect.width);
                      const py = p[1] / (nh / rect.height);
                      return (
                        <g key={i}>
                          <circle cx={px} cy={py} r={5} fill="#e74c3c" stroke="#fff" strokeWidth="1.5" />
                          <text x={px + 8} y={py + 5} fill="white" fontSize="13" fontWeight="bold" textShadow="1px 1px 2px black">{i + 1}</text>
                        </g>
                      );
                    })}
                  </svg>
                </div>
              )}

              {/* Road Zone Editor Controls */}
              {isConfiguringZone && (
                <div style={{ position: 'absolute', bottom: 12, left: 12, zIndex: 20, display: 'flex', gap: '8px', background: 'rgba(11,17,32,0.9)', padding: '8px 12px', borderRadius: '6px', border: '1px solid #1e304a' }}>
                  <span style={{ color: '#c5d8f0', alignSelf: 'center', marginRight: '6px', fontSize: '12px', fontWeight: 600 }}>Click video to add road polygon vertices</span>
                  <button onClick={() => setZonePoints(prev => prev.slice(0, -1))} style={{ padding: '4px 10px', background: '#1e304a', color: '#c5d8f0', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>Undo</button>
                  <button onClick={() => setZonePoints([])} style={{ padding: '4px 10px', background: '#1e304a', color: '#c5d8f0', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>Clear</button>
                  <button onClick={handleSaveRoadZone} style={{ padding: '4px 14px', background: '#2ecc71', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 600 }}>Save Polygon</button>
                  <button onClick={() => setIsConfiguringZone(false)} style={{ padding: '4px 10px', background: '#34495e', color: '#c5d8f0', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>Cancel</button>
                </div>
              )}
            </div>
          </section>

          {/* Telemetry Metrics Row */}
          <div className="stats-row">
            <div className="stat-card">
              <div className="stat-icon deer-icon">🦌</div>
              <div>
                <div className="stat-label">ACTIVE DEER</div>
                <div className="stat-value">{backendOnline ? tracks.length : '--'}</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon track-icon">👥</div>
              <div>
                <div className="stat-label">ACTIVE TRACKS</div>
                <div className="stat-value">{backendOnline ? tracks.length : '--'}</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon fps-icon">⏱</div>
              <div>
                <div className="stat-label">CURRENT FPS</div>
                <div className="stat-value">{backendOnline && fps > 0 ? fps.toFixed(1) : '--'}</div>
              </div>
            </div>
            <div className={`stat-card ${highRiskCount > 0 ? 'stat-danger' : ''}`}>
              <div className="stat-icon warn-icon">⚠</div>
              <div>
                <div className="stat-label">HIGH RISK EVENTS</div>
                <div className="stat-value">{backendOnline ? highRiskCount : '--'}</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon proc-icon">📊</div>
              <div>
                <div className="stat-label">PROCESSING STATUS</div>
                <StatusDot status={backendOnline ? (status?.processing_status ?? 'WAITING') : 'OFFLINE'} />
              </div>
            </div>
          </div>

          {/* Active Tracks Monitor Table */}
          <section className="panel track-panel">
            <div className="panel-title">☰ REAL-TIME TRACK MONITOR</div>
            <table className="track-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>CONFIDENCE</th>
                  <th>POSITION (x, y)</th>
                  <th>SPEED (px/f)</th>
                  <th>PREDICTION</th>
                  <th>COLLISION RISK</th>
                  <th>VLM CAPTION</th>
                </tr>
              </thead>
              <tbody>
                {tracks.length === 0 ? (
                  <tr><td colSpan={7} className="no-tracks">{backendOnline ? 'No active tracks detected in current frame' : 'Connecting to backend...'}</td></tr>
                ) : (
                  tracks.map(t => (
                    <tr
                      key={t.track_id}
                      className={`track-row ${selectedTrack?.track_id === t.track_id ? 'selected-row' : ''} ${t.risk === 'HIGH_RISK' ? 'row-high-risk' : ''}`}
                      onClick={() => setSelectedTrack(t)}
                    >
                      <td className="td-id">#{t.track_id}</td>
                      <td>{(t.confidence * 100).toFixed(0)}%</td>
                      <td>{getPosition(t.bbox)}</td>
                      <td>{(t.speed || 0).toFixed(1)}</td>
                      <td>{t.predicted ? `(${t.predicted[0]}, ${t.predicted[1]})` : '—'}</td>
                      <td><RiskBadge risk={t.risk} /></td>
                      <td style={{ maxWidth: '160px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {t.caption ? <span style={{ color: '#2ecc71' }}>{t.caption}</span> : <span className="vlm-dash">—</span>}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </section>

          {/* Event Logs */}
          <section className="panel events-panel">
            <div className="panel-title">🔔 SYSTEM EVENT LOG</div>
            <div className="events-list">
              {events.length === 0 ? (
                <div style={{ padding: '8px', color: '#6b8ab0' }}>EVENT HISTORY UNAVAILABLE</div>
              ) : (
                events.slice(0, 6).map((e, i) => (
                  <div key={i} className={`event-row event-${e.level}`}>
                    <span className={`event-dot dot-${e.level}`} />
                    <span className="event-time">{e.time}</span>
                    <span className="event-msg">{e.message}</span>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>

        {/* RIGHT COLUMN */}
        <div className="dg-right">
          {/* System Pipeline Status */}
          <section className="panel sys-status-panel">
            <div className="panel-title">⚙ SYSTEM PIPELINE HEALTH</div>
            {status ? (
              <>
                <ComponentRow icon="📷" label="Camera Source" status={status.components.camera} />
                <ComponentRow icon="🎯" label="Detection (YOLOv11)" status={status.components.detection} />
                <ComponentRow icon="🔄" label="Tracking (ByteTrack)" status={status.components.tracking} />
                <ComponentRow icon="📐" label="Prediction (Kalman)" status={status.components.prediction} />
                <ComponentRow icon="💥" label="Collision Engine" status={status.components.collision_engine} />
                <ComponentRow icon="🧠" label="VLM (BLIP)" status={status.components.vlm} />
              </>
            ) : (
              <div className="loading-text">Connecting to backend pipeline…</div>
            )}
          </section>

          {/* Current Risk Status Banner */}
          <section className={`panel risk-panel ${highRiskCount > 0 ? 'risk-panel-danger' : ''}`}>
            <div className="panel-title-risk">
              <span>⚠</span> CURRENT ROAD COLLISION RISK
            </div>
            {(() => {
              const worst = tracks.reduce<string>((acc, t) => {
                if (t.risk === 'HIGH_RISK') return 'HIGH_RISK';
                if (t.risk === 'CAUTION' && acc !== 'HIGH_RISK') return 'CAUTION';
                if (t.risk === 'SAFE' && acc === 'UNKNOWN') return 'SAFE';
                return acc;
              }, tracks.length === 0
                ? ((status?.detection_count ?? 0) > 0 || !status?.road_zone_configured ? 'UNKNOWN' : 'SAFE')
                : 'SAFE');
              return (
                <div className={`risk-badge-large risk-large-${worst.toLowerCase().replace('_', '-')}`}>
                  ⚠ {riskLabel(worst)}
                </div>
              );
            })()}
            <div className="risk-desc">
              {status?.road_zone_configured
                ? (highRiskCount > 0 ? `${highRiskCount} DEER IN HIGH RISK ZONE` : 'Road zone monitored — risk evaluation active')
                : 'Road-zone geometry unavailable'}
            </div>
            <div className="risk-detail">
              {status?.road_zone_configured
                ? 'Deterministic collision engine running on active Kalman trajectories.'
                : 'UNKNOWN returned because road-zone geometry is not configured.\nNEVER treated as SAFE.'}
            </div>
          </section>

          {/* VLM BLIP Status Card */}
          <section className="panel vlm-panel">
            <div className="panel-title">💬 VISION-LANGUAGE MODEL (BLIP)</div>
            <div className="vlm-status-row">
              <StatusDot status={status?.vlm_status ?? 'STANDBY'} />
            </div>
            <div className="vlm-sub">Lazy-loaded. Triggered exclusively for genuine HIGH_RISK events.</div>
            <div className={`vlm-result ${status?.vlm_triggered ? 'triggered' : ''}`}>
              <span>•</span>
              {status?.vlm_triggered
                ? 'BLIP ACTIVE — Caption generated'
                : 'BLIP NOT TRIGGERED — NO NATURAL HIGH_RISK EVENT'}
            </div>
            {status?.vlm_captions && status.vlm_captions.length > 0 && (
              <div style={{ marginTop: '8px', padding: '6px 8px', background: '#0a1420', borderRadius: '4px', border: '1px solid #1e304a' }}>
                {status.vlm_captions.map((c, i) => (
                  <div key={i} style={{ fontSize: '11px', color: '#2ecc71', marginBottom: '4px' }}>
                    <strong>Track #{c.track_id}:</strong> "{c.caption}"
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* Road Zone Status & Editor Trigger */}
          <section className="panel road-panel">
            <div className="panel-title">🗺 ROAD ZONE CONFIGURATION</div>
            <div className="road-preview">
              {status?.road_zone_configured && status.road_zone_points.length > 0 ? (
                <svg width="100%" height="80" style={{ background: '#0a1420' }}>
                  <polygon
                    points={status.road_zone_points.map(p => `${p[0] / 5},${p[1] / 6}`).join(' ')}
                    fill="rgba(240,200,50,0.3)"
                    stroke="#f0c832"
                    strokeWidth="1.5"
                  />
                </svg>
              ) : (
                <div className="road-grid">
                  {Array.from({ length: 30 }).map((_, i) => (
                    <div key={i} className="road-cell" />
                  ))}
                </div>
              )}
            </div>
            <div className={`road-status ${status?.road_zone_configured ? 'road-active' : 'road-missing'}`}>
              {status?.road_zone_configured ? `ROAD ZONE CONFIGURED (${status.road_zone_points.length} VERTICES)` : 'ROAD ZONE NOT CONFIGURED'}
            </div>
            <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
              <button
                onClick={() => {
                  setIsConfiguringZone(true);
                  setZonePoints([]);
                }}
                style={{ flex: 1, padding: '8px', background: '#f39c12', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 600 }}
              >
                Configure Road Zone
              </button>
              {status?.road_zone_configured && (
                <button
                  onClick={handleDeleteRoadZone}
                  style={{ padding: '8px 12px', background: 'rgba(231,76,60,0.2)', border: '1px solid #e74c3c', color: '#e74c3c', borderRadius: '4px', cursor: 'pointer' }}
                >
                  Clear
                </button>
              )}
            </div>
          </section>

          {/* Selected Track Inspection Panel */}
          {selectedTrack && (
            <section className="panel selected-panel">
              <div className="selected-header">
                <span className="panel-title">🦌 SELECTED TRACK: <span className="selected-id">#{selectedTrack.track_id}</span></span>
                <button className="close-btn" onClick={() => setSelectedTrack(null)}>✕</button>
              </div>
              <div className="selected-body">
                <div className="selected-info">
                  <div className="sel-row">
                    <span className="sel-label">Class</span>
                    <span className="sel-value">{selectedTrack.class_name}</span>
                  </div>
                  <div className="sel-row">
                    <span className="sel-label">Confidence</span>
                    <span className="sel-value">{(selectedTrack.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="sel-row">
                    <span className="sel-label">Position</span>
                    <span className="sel-value">{getPosition(selectedTrack.bbox)}</span>
                  </div>
                  <div className="sel-row">
                    <span className="sel-label">Speed</span>
                    <span className="sel-value">{(selectedTrack.speed || 0).toFixed(1)} px/f</span>
                  </div>
                  <div className="sel-row">
                    <span className="sel-label">Risk Level</span>
                    <RiskBadge risk={selectedTrack.risk} />
                  </div>
                  <div className="sel-row">
                    <span className="sel-label">Prediction</span>
                    <span className="sel-value">{selectedTrack.predicted ? `(${selectedTrack.predicted[0]}, ${selectedTrack.predicted[1]})` : '—'}</span>
                  </div>
                  {selectedTrack.caption && (
                    <div className="sel-row" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                      <span className="sel-label">BLIP Caption:</span>
                      <span className="sel-value sel-caption" style={{ color: '#2ecc71', marginTop: '2px' }}>"{selectedTrack.caption}"</span>
                    </div>
                  )}
                </div>
                <div className="trajectory-panel">
                  <div className="traj-title">Real Trajectory &amp; Kalman Horizon</div>
                  <div className="traj-legend">
                    <span className="traj-dot past" />Past Points
                    <span className="traj-dot curr" />Current
                    <span className="traj-dot pred" />Kalman Pred
                  </div>
                  <div className="traj-canvas">
                    <TrajCanvas track={selectedTrack} frameWidth={streamRef.current?.naturalWidth || 640} frameHeight={streamRef.current?.naturalHeight || 480} />
                  </div>
                </div>
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
