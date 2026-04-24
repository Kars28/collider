"use client";

import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import { Line2 } from "three-stdlib";
import * as THREE from "three";

// ─── constants ───────────────────────────────────────────────────────────────

const TRACK_LENGTH = 8;
const TRACK_DURATION = 2;
const FLASH_DURATION = 0.3;
const BEAM_SPEED = 12;
const BEAM_START_X = 15;
const API_URL = "http://localhost:8000";

// ─── filter config ───────────────────────────────────────────────────────────

type FilterType =
  | "any" | "z_boson" | "jpsi"
  | "upsilon_1s" | "upsilon_2s" | "upsilon_3s"
  | "phi" | "rho_omega" | "high_mass" | "low_mass";

const FILTERS: { value: FilterType; label: string; color: string }[] = [
  { value: "any",        label: "ANY EVENT",               color: "#00d4ff" },
  { value: "z_boson",    label: "Z BOSON (85-97 GeV)",     color: "#00d4ff" },
  { value: "jpsi",       label: "J/\u03C8 MESON (2.9-3.3 GeV)",  color: "#ff006e" },
  { value: "upsilon_1s", label: "\u03A5(1S) UPSILON (9.2-9.6)",  color: "#aaff00" },
  { value: "upsilon_2s", label: "\u03A5(2S) UPSILON (9.9-10.1)", color: "#aaff00" },
  { value: "upsilon_3s", label: "\u03A5(3S) UPSILON (10.3-10.4)",color: "#aaff00" },
  { value: "phi",        label: "\u03C6 PHI MESON (0.99-1.05)",  color: "#ffffff" },
  { value: "rho_omega",  label: "\u03C1/\u03C9 MESON (0.6-0.9)",       color: "#ffffff" },
  { value: "high_mass",  label: "HIGH MASS (>100 GeV)",    color: "#ff006e" },
  { value: "low_mass",   label: "LOW MASS (<2 GeV)",       color: "#ffffff" },
];

// ─── types ───────────────────────────────────────────────────────────────────

type CollisionStage = "idle" | "beams" | "impact" | "tracks" | "showing";

interface CernParticle {
  type: string;
  charge: number;
  color: string;
  momentum: { px: number; py: number; pz: number };
  energy: number;
  eta: number;
  phi: number;
  pt: number;
}

interface CernEvent {
  event_id: string;
  invariant_mass: number;
  energy_tev: string;
  event_type: string;
  resolved_type?: string;
  is_custom?: boolean;
  particles: CernParticle[];
  track_directions: { x: number; y: number; z: number }[];
}

interface BeamOption {
  id: string;
  label: string;
  symbol: string;
  mass_gev: number;
}

interface DecayOption {
  id: string;
  label: string;
  symbol?: string;
  description?: string;
  mass_gev?: number;
}

type SceneMode = "real" | "custom";

interface ParticleInfo {
  name: string;
  mass: string;
  symbol: string;
  what_is_it: string;
  what_happened: string;
  discovery: string;
  nobel_prize: string;
  experiment: string;
  did_you_know: string;
  color: string;
}

interface HudData {
  energy: string;
  eventType: string;
  invariantMass: string;
  eventId: string;
  particles: number | string;
  status: "connecting" | "ready" | "live";
  isCustom?: boolean;
}

type TrackData = {
  dir: [number, number, number];
  color: string;
  particle?: CernParticle;       // real muon data (first 2 tracks)
  particleSymbol?: string;       // from particle-info lookup
};

// ─── helpers ─────────────────────────────────────────────────────────────────

function randomDirection(): [number, number, number] {
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  return [
    Math.sin(phi) * Math.cos(theta),
    Math.sin(phi) * Math.sin(theta),
    Math.cos(phi),
  ];
}

async function fetchEvents(filter: FilterType = "any"): Promise<CernEvent[]> {
  const url = `${API_URL}/events/filter?event_type=${filter}&limit=10`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch events");
  return res.json();
}

async function fetchParticleInfo(): Promise<Record<string, ParticleInfo>> {
  const res = await fetch(`${API_URL}/particle-info`);
  if (!res.ok) throw new Error("Failed to fetch particle info");
  return res.json();
}

async function fetchCollisionOptions(): Promise<{ beams: BeamOption[]; decay_channels: DecayOption[] }> {
  const res = await fetch(`${API_URL}/collision/options`);
  if (!res.ok) throw new Error("Failed to fetch collision options");
  return res.json();
}

async function fetchCustomCollision(body: {
  beam1: string; beam2: string; energy_tev: number; decay_channel: string;
}): Promise<CernEvent> {
  const res = await fetch(`${API_URL}/collision/custom`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("Failed to generate custom collision");
  return res.json();
}

// Map event_type strings from API to filter keys for particle-info lookup
function eventTypeToFilterKey(eventType: string): string {
  const map: Record<string, string> = {
    "Z Boson Candidate": "z_boson",
    "J/\u03C8 Meson": "jpsi",
    "\u03A5(1S) Meson": "upsilon_1s",
    "\u03A5(2S) Meson": "upsilon_2s",
    "\u03A5(3S) Meson": "upsilon_3s",
    "\u03C6 Meson": "phi",
    "\u03C1/\u03C9 Meson": "rho_omega",
    "High Mass Dimuon": "high_mass",
    "Low Mass Dimuon": "low_mass",
    "Dimuon Event": "any",
  };
  return map[eventType] ?? "any";
}

// ─── CMS-style detector ───────────────────────────────────────────────────────

function CMSDetector() {
  return (
    <group>
      {/* Barrel aligned along X axis */}
      <mesh rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[3, 3, 8, 48, 5, true]} />
        <meshBasicMaterial color="#007766" wireframe />
      </mesh>
      {/* Endcap +X */}
      <mesh position={[4, 0, 0]} rotation={[0, Math.PI / 2, 0]}>
        <circleGeometry args={[3, 48]} />
        <meshBasicMaterial color="#007766" wireframe side={THREE.DoubleSide} />
      </mesh>
      {/* Endcap -X */}
      <mesh position={[-4, 0, 0]} rotation={[0, Math.PI / 2, 0]}>
        <circleGeometry args={[3, 48]} />
        <meshBasicMaterial color="#007766" wireframe side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

// ─── beam spheres ────────────────────────────────────────────────────────────

function BeamSpheres({ stage, onImpact }: { stage: CollisionStage; onImpact: () => void }) {
  const group1 = useRef<THREE.Group>(null);
  const group2 = useRef<THREE.Group>(null);
  const x1 = useRef(-BEAM_START_X);
  const x2 = useRef(BEAM_START_X);
  const prevStage = useRef<CollisionStage>("idle");

  useFrame((_, delta) => {
    if (stage === "beams" && prevStage.current !== "beams") {
      x1.current = -BEAM_START_X;
      x2.current = BEAM_START_X;
    }
    prevStage.current = stage;
    if (stage !== "beams") return;

    const speed1 = BEAM_SPEED * (1 + (BEAM_START_X - Math.abs(x1.current)) / BEAM_START_X * 2);
    x1.current += speed1 * delta;
    x2.current -= speed1 * delta;
    if (x1.current > 0) x1.current = 0;
    if (x2.current < 0) x2.current = 0;

    if (group1.current) group1.current.position.x = x1.current;
    if (group2.current) group2.current.position.x = x2.current;

    if (Math.abs(x1.current) < 0.5 && Math.abs(x2.current) < 0.5) onImpact();
  });

  if (stage !== "beams") return null;
  return (
    <>
      <group ref={group1} position={[-BEAM_START_X, 0, 0]}>
        <mesh>
          <sphereGeometry args={[0.3, 32, 32]} />
          <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={3} />
        </mesh>
        <pointLight intensity={40} distance={12} color="#88ccff" />
      </group>
      <group ref={group2} position={[BEAM_START_X, 0, 0]}>
        <mesh>
          <sphereGeometry args={[0.3, 32, 32]} />
          <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={3} />
        </mesh>
        <pointLight intensity={40} distance={12} color="#88ccff" />
      </group>
    </>
  );
}

// ─── collision flash ─────────────────────────────────────────────────────────

function CollisionFlash({ stage, onDone }: { stage: CollisionStage; onDone: () => void }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const elapsed = useRef(0);
  const prevStage = useRef<CollisionStage>("idle");

  useFrame((_, delta) => {
    if (stage === "impact" && prevStage.current !== "impact") elapsed.current = 0;
    prevStage.current = stage;
    if (stage !== "impact" || !meshRef.current) return;

    elapsed.current += delta;
    const t = elapsed.current / FLASH_DURATION;
    if (t < 1) {
      (meshRef.current.material as THREE.MeshBasicMaterial).opacity = 1 - t;
      meshRef.current.scale.setScalar(1 + t * 2);
    } else {
      onDone();
    }
  });

  if (stage !== "impact") return null;
  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.5, 24, 24]} />
      <meshBasicMaterial color="#ffffff" transparent opacity={1} />
    </mesh>
  );
}

// ─── particle tracks (no raycasting — hover handled in 2D) ─────────────────

function Track({ dir, color }: { dir: [number, number, number]; color: string }) {
  const lineRef = useRef<Line2>(null);
  const elapsed = useRef(0);

  useFrame((_, delta) => {
    if (!lineRef.current) return;
    elapsed.current = Math.min(elapsed.current + delta, TRACK_DURATION);
    const t = elapsed.current / TRACK_DURATION;
    lineRef.current.geometry.setPositions([
      0, 0, 0,
      dir[0] * TRACK_LENGTH * t,
      dir[1] * TRACK_LENGTH * t,
      dir[2] * TRACK_LENGTH * t,
    ]);
  });

  return (
    <Line
      ref={lineRef}
      points={[[0, 0, 0], [dir[0] * 0.001, dir[1] * 0.001, dir[2] * 0.001]]}
      color={color}
      lineWidth={1.5}
    />
  );
}

function ParticleTracks({
  stage, onDone, trackData,
}: {
  stage: CollisionStage;
  onDone: () => void;
  trackData: TrackData[];
}) {
  const elapsed = useRef(0);
  const prevStage = useRef<CollisionStage>("idle");
  const doneFired = useRef(false);

  useFrame((_, delta) => {
    if (stage === "tracks" && prevStage.current !== "tracks") {
      elapsed.current = 0;
      doneFired.current = false;
    }
    prevStage.current = stage;
    if (stage !== "tracks") return;
    elapsed.current += delta;
    if (elapsed.current >= TRACK_DURATION && !doneFired.current) {
      doneFired.current = true;
      onDone();
    }
  });

  if (stage !== "tracks" && stage !== "showing") return null;

  return (
    <>
      {trackData.map((track, i) => (
        <Track key={i} dir={track.dir} color={track.color} />
      ))}
    </>
  );
}

// ─── build track data from CERN event ────────────────────────────────────────

function buildTrackData(event: CernEvent, pInfo?: Record<string, ParticleInfo>): TrackData[] {
  const tracks: TrackData[] = [];
  const filterKey = event.resolved_type ?? eventTypeToFilterKey(event.event_type);
  const sym = pInfo?.[filterKey]?.symbol ?? "";

  for (let i = 0; i < event.track_directions.length; i++) {
    const d = event.track_directions[i];
    const p = event.particles[i];
    tracks.push({
      dir: [d.x, d.y, d.z],
      color: p?.color ?? "#ffffff",
      particle: p,
      particleSymbol: sym,
    });
  }

  const colors = event.particles.map((p) => p.color);
  while (tracks.length < 30) {
    tracks.push({
      dir: randomDirection(),
      color: colors[tracks.length % colors.length],
    });
  }
  return tracks;
}

// ─── expose camera to parent via ref ────────────────────────────────────────

function CameraExposer({ cameraRef }: { cameraRef: React.MutableRefObject<THREE.Camera | null> }) {
  const { camera } = useThree();
  useEffect(() => { cameraRef.current = camera; }, [camera, cameraRef]);
  return null;
}

// ─── collision controller ───────────────────────────────────────────────────

function CollisionController({
  stage, setStage, trackKey, trackData,
}: {
  stage: CollisionStage;
  setStage: (s: CollisionStage) => void;
  trackKey: number;
  trackData: TrackData[];
}) {
  const handleImpact = useCallback(() => setStage("impact"), [setStage]);
  const handleFlashDone = useCallback(() => setStage("tracks"), [setStage]);
  const handleTracksDone = useCallback(() => setStage("showing"), [setStage]);

  return (
    <>
      <BeamSpheres stage={stage} onImpact={handleImpact} />
      <CollisionFlash stage={stage} onDone={handleFlashDone} />
      <ParticleTracks
        key={trackKey}
        stage={stage}
        onDone={handleTracksDone}
        trackData={trackData}
      />
    </>
  );
}

// ─── shared HUD styles ───────────────────────────────────────────────────────

const HUD_PANEL: React.CSSProperties = {
  position: "absolute",
  background: "rgba(0, 0, 0, 0.75)",
  border: "1px solid rgba(0, 255, 204, 0.3)",
  borderRadius: "2px",
  padding: "0.75rem 1rem",
  fontFamily: "monospace",
  color: "#00ffcc",
  textTransform: "uppercase",
  letterSpacing: "0.12em",
  fontSize: "0.7rem",
  lineHeight: 1.7,
  pointerEvents: "none",
  userSelect: "none",
};

// ─── scene root ──────────────────────────────────────────────────────────────

export default function Home() {
  const [stage, setStage] = useState<CollisionStage>("idle");
  const [trackKey, setTrackKey] = useState(0);
  const [trackData, setTrackData] = useState<TrackData[]>([]);
  const [activeFilter, setActiveFilter] = useState<FilterType>("any");
  const [filterLoading, setFilterLoading] = useState(false);

  // Particle info from API
  const particleInfoRef = useRef<Record<string, ParticleInfo>>({});
  const [activeInfo, setActiveInfo] = useState<ParticleInfo | null>(null);
  const [showInfoPanel, setShowInfoPanel] = useState(false);

  // Tooltip for track hover
  const [tooltip, setTooltip] = useState<{ x: number; y: number; track: TrackData } | null>(null);

  // Event buffer
  const eventsRef = useRef<CernEvent[]>([]);
  const indexRef = useRef(0);
  const filterRef = useRef<FilterType>("any");

  // ─── custom collision mode ───────────────────────────────────────────────
  const [sceneMode, setSceneMode] = useState<SceneMode>("real");
  const [beamOptions, setBeamOptions] = useState<BeamOption[]>([]);
  const [decayOptions, setDecayOptions] = useState<DecayOption[]>([]);
  const [customBeam1, setCustomBeam1] = useState("proton");
  const [customBeam2, setCustomBeam2] = useState("proton");
  const [customEnergy, setCustomEnergy] = useState(13.6);
  const [customDecay, setCustomDecay] = useState("any");
  const [customLoading, setCustomLoading] = useState(false);

  const [hud, setHud] = useState<HudData>({
    energy: "-- TeV",
    eventType: "CONNECTING TO CERN DATA...",
    invariantMass: "-- GeV",
    eventId: "--",
    particles: "--",
    status: "connecting",
  });

  // Fetch events
  const loadEvents = useCallback(async (filter?: FilterType) => {
    const f = filter ?? filterRef.current;
    try {
      const data = await fetchEvents(f);
      eventsRef.current = data;
      indexRef.current = 0;
      setHud((prev) => ({
        ...prev,
        status: prev.status === "connecting" ? "ready" : prev.status,
        eventType: prev.status === "connecting" ? "READY" : prev.eventType,
      }));
    } catch (err) {
      console.error("Failed to fetch CERN events:", err);
    }
  }, []);

  // Fetch on mount
  useEffect(() => {
    loadEvents("any");
    fetchParticleInfo()
      .then((info) => { particleInfoRef.current = info; })
      .catch((err) => console.error("Failed to fetch particle info:", err));
    fetchCollisionOptions()
      .then((opts) => {
        setBeamOptions(opts.beams);
        setDecayOptions(opts.decay_channels);
      })
      .catch((err) => console.error("Failed to fetch collision options:", err));
  }, [loadEvents]);

  // Show info panel 0.5s after tracks appear
  useEffect(() => {
    if (stage === "showing" && activeInfo) {
      const timer = setTimeout(() => setShowInfoPanel(true), 500);
      return () => clearTimeout(timer);
    }
    if (stage === "beams") {
      setShowInfoPanel(false);
    }
  }, [stage, activeInfo]);

  const handleFilterChange = useCallback(async (filter: FilterType) => {
    setActiveFilter(filter);
    filterRef.current = filter;
    setFilterLoading(true);
    await loadEvents(filter);
    setFilterLoading(false);
  }, [loadEvents]);

  const canFire = (stage === "idle" || stage === "showing") && !filterLoading && !customLoading;
  const canFireReal = canFire && eventsRef.current.length > 0;

  // Fire using real data
  const handleFireReal = useCallback(() => {
    if (!canFireReal) return;

    const events = eventsRef.current;
    const idx = indexRef.current;
    const event = events[idx];
    const filterKey = event.resolved_type ?? eventTypeToFilterKey(event.event_type);
    const info = particleInfoRef.current[filterKey] ?? null;

    setHud({
      energy: event.energy_tev,
      eventType: event.event_type,
      invariantMass: event.invariant_mass.toFixed(3) + " GeV",
      eventId: event.event_id,
      particles: 2,
      status: "live",
      isCustom: false,
    });

    setActiveInfo(info);
    setShowInfoPanel(false);
    setTooltip(null);
    setTrackData(buildTrackData(event, particleInfoRef.current));
    setTrackKey((k) => k + 1);
    setStage("beams");

    const nextIdx = idx + 1;
    if (nextIdx >= events.length) {
      loadEvents();
    } else {
      indexRef.current = nextIdx;
    }
  }, [canFireReal, loadEvents]);

  // Fire using custom collision
  const handleFireCustom = useCallback(async () => {
    if (!canFire) return;
    setCustomLoading(true);
    try {
      const event = await fetchCustomCollision({
        beam1: customBeam1,
        beam2: customBeam2,
        energy_tev: customEnergy,
        decay_channel: customDecay,
      });
      const filterKey = event.resolved_type ?? "any";
      const info = particleInfoRef.current[filterKey] ?? null;

      setHud({
        energy: event.energy_tev,
        eventType: event.event_type,
        invariantMass: event.invariant_mass.toFixed(3) + " GeV",
        eventId: event.event_id,
        particles: event.particles.length,
        status: "live",
        isCustom: true,
      });

      setActiveInfo(info);
      setShowInfoPanel(false);
      setTooltip(null);
      setTrackData(buildTrackData(event, particleInfoRef.current));
      setTrackKey((k) => k + 1);
      setStage("beams");
    } catch (err) {
      console.error("Custom collision failed:", err);
    } finally {
      setCustomLoading(false);
    }
  }, [canFire, customBeam1, customBeam2, customEnergy, customDecay]);

  // Unified fire handler
  const handleFire = useCallback(() => {
    if (sceneMode === "custom") {
      handleFireCustom();
    } else {
      handleFireReal();
    }
  }, [sceneMode, handleFireCustom, handleFireReal]);

  // ─── 2D screen-space hover detection ─────────────────────────────────────
  const cameraRef = useRef<THREE.Camera | null>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef(tooltip);
  tooltipRef.current = tooltip;

  // Helper: distance from point P to line segment AB in 2D
  const pointToSegment2D = useCallback(
    (px: number, py: number, ax: number, ay: number, bx: number, by: number) => {
      const dx = bx - ax;
      const dy = by - ay;
      const lenSq = dx * dx + dy * dy;
      if (lenSq === 0) return Math.hypot(px - ax, py - ay);
      let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
      t = Math.max(0, Math.min(1, t));
      return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
    },
    [],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if ((stage !== "tracks" && stage !== "showing") || trackData.length === 0) {
        if (tooltipRef.current) setTooltip(null);
        return;
      }
      const cam = cameraRef.current;
      const el = canvasRef.current;
      if (!cam || !el) return;

      const rect = el.getBoundingClientRect();
      const mx = e.clientX;
      const my = e.clientY;
      const w = rect.width;
      const h = rect.height;

      // Project 3D point to screen pixels
      const project = (x: number, y: number, z: number) => {
        const v = new THREE.Vector3(x, y, z).project(cam);
        return {
          sx: (v.x * 0.5 + 0.5) * w + rect.left,
          sy: (-v.y * 0.5 + 0.5) * h + rect.top,
        };
      };

      let bestDist = Infinity;
      let bestIdx = -1;

      for (let i = 0; i < trackData.length; i++) {
        const d = trackData[i].dir;
        const a = project(0, 0, 0);
        const b = project(d[0] * TRACK_LENGTH, d[1] * TRACK_LENGTH, d[2] * TRACK_LENGTH);
        const dist = pointToSegment2D(mx, my, a.sx, a.sy, b.sx, b.sy);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = i;
        }
      }

      if (bestDist < 15 && bestIdx >= 0) {
        const track = trackData[bestIdx];
        setTooltip({ x: mx, y: my, track });
      } else {
        setTooltip(null);
      }
    },
    [stage, trackData, pointToSegment2D],
  );

  return (
    <div
      ref={canvasRef}
      style={{ width: "100vw", height: "100vh", background: "#000", position: "relative" }}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setTooltip(null)}
    >
      <Canvas gl={{ antialias: false }} camera={{ position: [0, 4, 14], fov: 60 }}>
        <CameraExposer cameraRef={cameraRef} />
        <ambientLight intensity={0.4} />
        <pointLight position={[5, 5, 5]} intensity={60} color="#ffffff" />

        <CMSDetector />
        <CollisionController
          stage={stage}
          setStage={setStage}
          trackKey={trackKey}
          trackData={trackData}
        />

        <OrbitControls enablePan={false} />

        <EffectComposer>
          <Bloom intensity={1.2} luminanceThreshold={0.4} luminanceSmoothing={0.9} mipmapBlur />
        </EffectComposer>
      </Canvas>

      {/* ── CSS animations ── */}
      <style>{`
        @keyframes hud-pulse{0%,100%{opacity:.35;box-shadow:0 0 4px #0f0}50%{opacity:1;box-shadow:0 0 10px #0f0}}
        @keyframes slide-in-right{from{transform:translateX(120%);opacity:0}to{transform:translateX(0);opacity:1}}
        input[type=range]{-webkit-appearance:none;appearance:none;background:rgba(0,255,204,0.15);height:4px;border-radius:2px;outline:none;}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#00ffcc;cursor:pointer;box-shadow:0 0 8px rgba(0,255,204,0.5);}
        select.cyber-select{-webkit-appearance:none;appearance:none;background:rgba(0,0,0,0.6);color:#00ffcc;border:1px solid rgba(0,255,204,0.3);border-radius:2px;padding:0.3rem 0.5rem;font-family:monospace;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.05em;cursor:pointer;width:100%;outline:none;}
        select.cyber-select:focus{border-color:#00ffcc;}
      `}</style>

      {/* ── HUD: top-left title ── */}
      <div style={{ ...HUD_PANEL, top: "1.25rem", left: "1.25rem" }}>
        <div style={{ fontSize: "0.85rem", fontWeight: "bold", marginBottom: "0.15rem" }}>
          CERN LHC SIMULATOR
        </div>
        <div style={{ color: "rgba(0,255,204,0.55)", fontSize: "0.6rem" }}>
          CMS Detector — Run 3
        </div>
      </div>

      {/* ── HUD: left panel with mode tabs ── */}
      <div
        style={{
          ...HUD_PANEL,
          top: "5.5rem",
          left: "1.25rem",
          pointerEvents: "auto",
          cursor: "default",
          maxHeight: "calc(100vh - 12rem)",
          overflowY: "auto",
          minWidth: "210px",
        }}
      >
        {/* Mode tabs */}
        <div style={{ display: "flex", gap: "0", marginBottom: "0.5rem", borderBottom: "1px solid rgba(0,255,204,0.2)" }}>
          {(["real", "custom"] as SceneMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setSceneMode(m)}
              style={{
                flex: 1,
                background: sceneMode === m ? "rgba(0,255,204,0.12)" : "transparent",
                color: sceneMode === m ? "#00ffcc" : "#556",
                border: "none",
                borderBottom: sceneMode === m ? "2px solid #00ffcc" : "2px solid transparent",
                fontFamily: "monospace",
                fontSize: "0.58rem",
                fontWeight: "bold",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                padding: "0.35rem 0.3rem",
                cursor: "pointer",
                transition: "all 0.15s",
              }}
            >
              {m === "real" ? "REAL DATA" : "CUSTOM"}
            </button>
          ))}
        </div>

        {/* ── Real data mode: filters ── */}
        {sceneMode === "real" && (
          <>
            <div style={{ marginBottom: "0.35rem", fontWeight: "bold", fontSize: "0.65rem", color: "rgba(0,255,204,0.5)" }}>
              EVENT FILTER
            </div>
            {filterLoading && (
              <div style={{ color: "#ffcc00", marginBottom: "0.25rem", fontSize: "0.6rem" }}>LOADING...</div>
            )}
            {FILTERS.map((f) => (
              <label
                key={f.value}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.4rem",
                  cursor: "pointer",
                  padding: "0.1rem 0",
                  color: activeFilter === f.value ? "#00ffcc" : "#778",
                  fontSize: "0.6rem",
                  transition: "color 0.15s",
                }}
              >
                <span
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: "50%",
                    border: `1.5px solid ${activeFilter === f.value ? f.color : "#445"}`,
                    background: activeFilter === f.value ? f.color : "transparent",
                    display: "inline-block",
                    flexShrink: 0,
                    boxShadow: activeFilter === f.value ? `0 0 6px ${f.color}80` : "none",
                    transition: "all 0.15s",
                  }}
                />
                <input
                  type="radio"
                  name="event-filter"
                  value={f.value}
                  checked={activeFilter === f.value}
                  onChange={() => handleFilterChange(f.value)}
                  style={{ display: "none" }}
                />
                {f.label}
              </label>
            ))}
          </>
        )}

        {/* ── Custom mode: collision builder ── */}
        {sceneMode === "custom" && (
          <>
            <div style={{ marginBottom: "0.35rem", fontWeight: "bold", fontSize: "0.65rem", color: "rgba(0,255,204,0.5)" }}>
              CUSTOM COLLISION BUILDER
            </div>
            <div style={{ borderTop: "1px solid rgba(0,255,204,0.15)", paddingTop: "0.4rem" }}>
              {/* Beam selectors */}
              <div style={{ display: "flex", gap: "0.4rem", marginBottom: "0.5rem" }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: "0.5rem", color: "#667", marginBottom: "0.15rem" }}>BEAM 1</div>
                  <select
                    className="cyber-select"
                    value={customBeam1}
                    onChange={(e) => setCustomBeam1(e.target.value)}
                  >
                    {beamOptions.map((b) => (
                      <option key={b.id} value={b.id}>{b.symbol} {b.label}</option>
                    ))}
                  </select>
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: "0.5rem", color: "#667", marginBottom: "0.15rem" }}>BEAM 2</div>
                  <select
                    className="cyber-select"
                    value={customBeam2}
                    onChange={(e) => setCustomBeam2(e.target.value)}
                  >
                    {beamOptions.map((b) => (
                      <option key={b.id} value={b.id}>{b.symbol} {b.label}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Energy slider */}
              <div style={{ marginBottom: "0.5rem" }}>
                <div style={{ fontSize: "0.5rem", color: "#667", marginBottom: "0.15rem" }}>ENERGY</div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                  <input
                    type="range"
                    min={1}
                    max={14}
                    step={0.1}
                    value={customEnergy}
                    onChange={(e) => setCustomEnergy(parseFloat(e.target.value))}
                    style={{ flex: 1 }}
                  />
                  <span style={{ fontSize: "0.6rem", color: "#00ffcc", minWidth: "50px", textAlign: "right" }}>
                    {customEnergy.toFixed(1)} TeV
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.45rem", color: "#445", marginTop: "0.1rem" }}>
                  <span>1 TeV</span>
                  <span>14 TeV</span>
                </div>
              </div>

              {/* Decay channel */}
              <div style={{ marginBottom: "0.6rem" }}>
                <div style={{ fontSize: "0.5rem", color: "#667", marginBottom: "0.15rem" }}>DECAY CHANNEL</div>
                <select
                  className="cyber-select"
                  value={customDecay}
                  onChange={(e) => setCustomDecay(e.target.value)}
                >
                  {decayOptions.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.symbol ? `${d.symbol} ` : ""}{d.label}{d.mass_gev ? ` (${d.mass_gev} GeV)` : ""}
                    </option>
                  ))}
                </select>
              </div>

              {/* Generate button */}
              <button
                onClick={handleFireCustom}
                disabled={!canFire || customLoading}
                style={{
                  width: "100%",
                  background: "rgba(0, 255, 204, 0.08)",
                  color: "#00ffcc",
                  border: "1px solid rgba(0,255,204,0.4)",
                  borderRadius: "2px",
                  padding: "0.45rem 0.5rem",
                  fontSize: "0.6rem",
                  fontFamily: "monospace",
                  fontWeight: "bold",
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                  cursor: canFire && !customLoading ? "pointer" : "not-allowed",
                  opacity: canFire && !customLoading ? 1 : 0.4,
                  transition: "all 0.15s",
                }}
              >
                {customLoading ? "GENERATING..." : "⚡ GENERATE COLLISION"}
              </button>
            </div>
          </>
        )}
      </div>

      {/* ── HUD: top-right event stats ── */}
      <div style={{ ...HUD_PANEL, top: "1.25rem", right: "1.25rem", minWidth: "240px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.3rem" }}>
          <span style={{ fontWeight: "bold", fontSize: "0.65rem", color: "rgba(0,255,204,0.5)" }}>EVENT DATA</span>
          {hud.isCustom && (
            <span style={{
              background: "rgba(0,255,204,0.12)",
              border: "1px solid rgba(0,255,204,0.3)",
              borderRadius: "2px",
              padding: "0.05rem 0.35rem",
              fontSize: "0.45rem",
              fontWeight: "bold",
              color: "#00ffcc",
              letterSpacing: "0.08em",
            }}>
              ⚡ AI PREDICTED
            </span>
          )}
        </div>
        {hud.status === "connecting" ? (
          <div style={{ color: "#ffcc00" }}>{hud.eventType}</div>
        ) : (
          <>
            <div><span style={{ color: "#aaa" }}>COLLISION ENERGY:</span> {hud.energy}</div>
            <div><span style={{ color: "#aaa" }}>EVENT TYPE:</span> {hud.eventType}</div>
            <div><span style={{ color: "#aaa" }}>INVARIANT MASS:</span> {hud.invariantMass}</div>
            <div><span style={{ color: "#aaa" }}>EVENT ID:</span> {hud.eventId}</div>
            <div><span style={{ color: "#aaa" }}>PARTICLES DETECTED:</span> {hud.particles}</div>
          </>
        )}
      </div>

      {/* ── HUD: physics explanation panel (slides in from right) ── */}
      {showInfoPanel && activeInfo && (
        <div
          style={{
            ...HUD_PANEL,
            top: "12rem",
            right: "1.25rem",
            maxWidth: "320px",
            maxHeight: "calc(100vh - 18rem)",
            overflowY: "auto",
            pointerEvents: "auto",
            animation: "slide-in-right 0.4s ease-out",
            lineHeight: 1.5,
            textTransform: "none",
            letterSpacing: "0.04em",
          }}
        >
          {/* Close button */}
          <button
            onClick={() => setShowInfoPanel(false)}
            style={{
              position: "absolute",
              top: "0.4rem",
              right: "0.5rem",
              background: "none",
              border: "none",
              color: "#556",
              cursor: "pointer",
              fontFamily: "monospace",
              fontSize: "0.85rem",
              padding: "0.15rem 0.3rem",
              lineHeight: 1,
            }}
          >
            ✕
          </button>

          <div style={{ fontWeight: "bold", fontSize: "0.65rem", color: "rgba(0,255,204,0.5)", textTransform: "uppercase", letterSpacing: "0.12em", marginBottom: "0.2rem" }}>
            PARTICLE ANALYSIS
          </div>
          <div style={{ borderBottom: "1px solid rgba(0,255,204,0.2)", marginBottom: "0.5rem" }} />

          <div style={{ fontSize: "0.95rem", fontWeight: "bold", color: activeInfo.color, marginBottom: "0.1rem" }}>
            {activeInfo.symbol} {activeInfo.name}
          </div>
          <div style={{ color: "#aaa", fontSize: "0.65rem", marginBottom: "0.6rem" }}>
            Mass: {activeInfo.mass}
          </div>

          <div style={{ fontWeight: "bold", fontSize: "0.6rem", color: "rgba(0,255,204,0.5)", textTransform: "uppercase", marginBottom: "0.15rem" }}>
            WHAT HAPPENED
          </div>
          <div style={{ color: "#ccc", fontSize: "0.62rem", marginBottom: "0.5rem" }}>
            {activeInfo.what_happened}
          </div>

          <div style={{ fontWeight: "bold", fontSize: "0.6rem", color: "rgba(0,255,204,0.5)", textTransform: "uppercase", marginBottom: "0.15rem" }}>
            DISCOVERY
          </div>
          <div style={{ color: "#ccc", fontSize: "0.62rem", marginBottom: "0.15rem" }}>
            {activeInfo.discovery}
          </div>
          {activeInfo.nobel_prize && (
            <div style={{ color: "#ffcc00", fontSize: "0.58rem", marginBottom: "0.5rem" }}>
              🏅 {activeInfo.nobel_prize}
            </div>
          )}

          <div style={{ fontWeight: "bold", fontSize: "0.6rem", color: "rgba(0,255,204,0.5)", textTransform: "uppercase", marginBottom: "0.15rem" }}>
            THIS EXPERIMENT
          </div>
          <div style={{ color: "#ccc", fontSize: "0.62rem", marginBottom: "0.5rem" }}>
            {activeInfo.experiment}
          </div>

          <div style={{ fontWeight: "bold", fontSize: "0.6rem", color: "rgba(0,255,204,0.5)", textTransform: "uppercase", marginBottom: "0.15rem" }}>
            DID YOU KNOW
          </div>
          <div style={{ color: "#ccc", fontSize: "0.62rem" }}>
            {activeInfo.did_you_know}
          </div>
        </div>
      )}

      {/* ── Track tooltip ── */}
      {tooltip && (
        <div
          style={{
            position: "fixed",
            left: tooltip.x + 14,
            top: tooltip.y - 10,
            background: "rgba(0,0,0,0.9)",
            border: "1px solid rgba(0,255,204,0.4)",
            borderRadius: "2px",
            padding: "0.4rem 0.6rem",
            fontFamily: "monospace",
            fontSize: "0.58rem",
            color: "#00ffcc",
            pointerEvents: "none",
            zIndex: 100,
            whiteSpace: "nowrap",
            lineHeight: 1.6,
          }}
        >
          {tooltip.track.particle ? (
            <>
              <div style={{ fontWeight: "bold", color: tooltip.track.color }}>
                {tooltip.track.particleSymbol ?? ""} {tooltip.track.particle.type} ({tooltip.track.particle.charge > 0 ? "+" : ""}{tooltip.track.particle.charge})
              </div>
              <div><span style={{ color: "#aaa" }}>Energy:</span> {tooltip.track.particle.energy.toFixed(2)} GeV</div>
              <div><span style={{ color: "#aaa" }}>pT:</span> {tooltip.track.particle.pt.toFixed(2)} GeV/c</div>
              <div><span style={{ color: "#aaa" }}>η:</span> {tooltip.track.particle.eta.toFixed(3)}</div>
              <div><span style={{ color: "#aaa" }}>φ:</span> {tooltip.track.particle.phi.toFixed(3)} rad</div>
            </>
          ) : (
            <div style={{ color: tooltip.track.color }}>Secondary particle track</div>
          )}
        </div>
      )}

      {/* ── HUD: bottom-left particle legend ── */}
      <div style={{ ...HUD_PANEL, bottom: "4rem", left: "1.25rem", maxWidth: "200px" }}>
        <div style={{ marginBottom: "0.2rem", fontWeight: "bold", fontSize: "0.6rem", color: "rgba(0,255,204,0.5)" }}>
          PARTICLE LEGEND
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.6rem" }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#00d4ff", display: "inline-block", flexShrink: 0 }} />
          <span>μ⁺ &nbsp;Positive Muon</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.6rem" }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#ff006e", display: "inline-block", flexShrink: 0 }} />
          <span>μ⁻ &nbsp;Negative Muon</span>
        </div>
      </div>

      {/* ── HUD: bottom-left data source ── */}
      <div style={{ ...HUD_PANEL, bottom: "1.25rem", left: "1.25rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <span
          style={{
            width: 6, height: 6, borderRadius: "50%",
            background: hud.status === "connecting" ? "#ffcc00" : "#0f0",
            display: "inline-block",
            animation: "hud-pulse 1.8s ease-in-out infinite",
          }}
        />
        <span style={{ color: "#aaa", fontSize: "0.6rem" }}>
          {hud.status === "connecting" ? "CONNECTING" : "LIVE"}
        </span>
        <span style={{ fontSize: "0.6rem" }}>DATA SOURCE: CERN OPEN DATA PORTAL</span>
      </div>

      {/* ── FIRE COLLISION button ── */}
      <button
        onClick={handleFire}
        disabled={sceneMode === "real" ? !canFireReal : (!canFire || customLoading)}
        style={{
          position: "absolute",
          bottom: "2rem",
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(0, 0, 0, 0.85)",
          color: "#00ffcc",
          border: "1px solid #00ffcc",
          borderRadius: "2px",
          padding: "0.75rem 2.5rem",
          fontSize: "0.8rem",
          fontFamily: "monospace",
          fontWeight: "bold",
          letterSpacing: "0.25em",
          textTransform: "uppercase",
          cursor: (sceneMode === "real" ? canFireReal : (canFire && !customLoading)) ? "pointer" : "not-allowed",
          opacity: (sceneMode === "real" ? canFireReal : (canFire && !customLoading)) ? 1 : 0.4,
          boxShadow: "0 0 16px rgba(0,255,204,0.35), inset 0 0 12px rgba(0,255,204,0.06)",
          transition: "box-shadow 0.15s ease, opacity 0.15s ease",
        }}
        onMouseEnter={(e) =>
          ((e.currentTarget as HTMLButtonElement).style.boxShadow =
            "0 0 28px rgba(0,255,204,0.65), inset 0 0 18px rgba(0,255,204,0.12)")
        }
        onMouseLeave={(e) =>
          ((e.currentTarget as HTMLButtonElement).style.boxShadow =
            "0 0 16px rgba(0,255,204,0.35), inset 0 0 12px rgba(0,255,204,0.06)")
        }
      >
        {sceneMode === "custom" ? "⚡ Fire Custom Collision" : "⬡ Fire Collision"}
      </button>
    </div>
  );
}
