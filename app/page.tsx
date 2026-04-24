"use client";

import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import { Line2 } from "three-stdlib";
import * as THREE from "three";

// ─── constants ───────────────────────────────────────────────────────────────

const TRACK_LENGTH = 8;           // scale factor for real direction vectors
const TRACK_DURATION = 2;         // seconds for tracks to fully extend
const FLASH_DURATION = 0.3;       // seconds for impact flash
const BEAM_SPEED = 12;            // units per second
const BEAM_START_X = 15;          // starting distance from center
const API_URL = "http://localhost:8000";

// ─── types ───────────────────────────────────────────────────────────────────

type CollisionStage = "idle" | "beams" | "impact" | "tracks" | "showing";

interface CernEvent {
  event_id: string;
  invariant_mass: number;
  energy_tev: string;
  event_type: string;
  particles: {
    type: string;
    charge: number;
    color: string;
    momentum: { px: number; py: number; pz: number };
    energy: number;
    eta: number;
    phi: number;
    pt: number;
  }[];
  track_directions: { x: number; y: number; z: number }[];
}

interface HudData {
  energy: string;
  eventType: string;
  invariantMass: string;
  eventId: string;
  particles: number | string;
  status: "connecting" | "ready" | "live";
}

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

async function fetchEvents(): Promise<CernEvent[]> {
  const res = await fetch(`${API_URL}/events`);
  if (!res.ok) throw new Error("Failed to fetch events");
  return res.json();
}

// ─── CMS-style detector ───────────────────────────────────────────────────────

function CMSDetector() {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((_, delta) => {
    if (groupRef.current) groupRef.current.rotation.y += delta * 0.25;
  });

  return (
    <group ref={groupRef}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[1.5, 1.5, 4, 48, 5, true]} />
        <meshBasicMaterial color="#00ffcc" wireframe />
      </mesh>
      <mesh position={[0, 0, 2]}>
        <circleGeometry args={[1.5, 48]} />
        <meshBasicMaterial color="#00ffcc" wireframe side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, 0, -2]}>
        <circleGeometry args={[1.5, 48]} />
        <meshBasicMaterial color="#00ffcc" wireframe side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

// ─── beam spheres ────────────────────────────────────────────────────────────

function BeamSpheres({
  stage,
  onImpact,
}: {
  stage: CollisionStage;
  onImpact: () => void;
}) {
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

    if (Math.abs(x1.current) < 0.5 && Math.abs(x2.current) < 0.5) {
      onImpact();
    }
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

function CollisionFlash({
  stage,
  onDone,
}: {
  stage: CollisionStage;
  onDone: () => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const elapsed = useRef(0);
  const prevStage = useRef<CollisionStage>("idle");

  useFrame((_, delta) => {
    if (stage === "impact" && prevStage.current !== "impact") {
      elapsed.current = 0;
    }
    prevStage.current = stage;

    if (stage !== "impact") return;
    if (!meshRef.current) return;

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

// ─── particle tracks ─────────────────────────────────────────────────────────

type TrackData = { dir: [number, number, number]; color: string };

function Track({ dir, color }: TrackData) {
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
  stage,
  onDone,
  trackData,
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

// ─── build track data from a CERN event ─────────────────────────────────────
// Uses real track_directions + particle colors, plus random filler tracks.

function buildTrackData(event: CernEvent): TrackData[] {
  const tracks: TrackData[] = [];

  // Real muon tracks from the event data
  for (let i = 0; i < event.track_directions.length; i++) {
    const d = event.track_directions[i];
    const color = event.particles[i]?.color ?? "#ffffff";
    tracks.push({ dir: [d.x, d.y, d.z], color });
  }

  // Fill remaining slots with random directions using particle colors
  const colors = event.particles.map((p) => p.color);
  while (tracks.length < 30) {
    tracks.push({
      dir: randomDirection(),
      color: colors[tracks.length % colors.length],
    });
  }

  return tracks;
}

// ─── collision controller ───────────────────────────────────────────────────

function CollisionController({
  stage,
  setStage,
  trackKey,
  trackData,
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

  // Event buffer — stores 10 events, cycles through them
  const eventsRef = useRef<CernEvent[]>([]);
  const indexRef = useRef(0);

  const [hud, setHud] = useState<HudData>({
    energy: "-- TeV",
    eventType: "CONNECTING TO CERN DATA...",
    invariantMass: "-- GeV",
    eventId: "--",
    particles: "--",
    status: "connecting",
  });

  // Fetch 10 events from API
  const loadEvents = useCallback(async () => {
    try {
      const data = await fetchEvents();
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
    loadEvents();
  }, [loadEvents]);

  const canFire = (stage === "idle" || stage === "showing") && eventsRef.current.length > 0;

  const handleFire = useCallback(() => {
    if (!canFire) return;

    const events = eventsRef.current;
    const idx = indexRef.current;
    const event = events[idx];

    // Update HUD with real event data
    setHud({
      energy: event.energy_tev,
      eventType: event.event_type,
      invariantMass: event.invariant_mass.toFixed(3) + " GeV",
      eventId: event.event_id,
      particles: 2,
      status: "live",
    });

    // Build track data from real directions
    setTrackData(buildTrackData(event));
    setTrackKey((k) => k + 1);
    setStage("beams");

    // Advance index — auto-refetch when we've used all 10
    const nextIdx = idx + 1;
    if (nextIdx >= events.length) {
      loadEvents(); // Fetch fresh batch
    } else {
      indexRef.current = nextIdx;
    }
  }, [canFire, loadEvents]);

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#000", position: "relative" }}>
      <Canvas gl={{ antialias: false }} camera={{ position: [0, 0, 7], fov: 60 }}>
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
          <Bloom
            intensity={2.5}
            luminanceThreshold={0.1}
            luminanceSmoothing={0.9}
            mipmapBlur
          />
        </EffectComposer>
      </Canvas>

      {/* ── pulsing dot keyframe ── */}
      <style>{`@keyframes hud-pulse{0%,100%{opacity:.35;box-shadow:0 0 4px #0f0}50%{opacity:1;box-shadow:0 0 10px #0f0}}`}</style>

      {/* ── HUD: top-left title ── */}
      <div style={{ ...HUD_PANEL, top: "1.25rem", left: "1.25rem" }}>
        <div style={{ fontSize: "0.85rem", fontWeight: "bold", marginBottom: "0.15rem" }}>
          CERN LHC SIMULATOR
        </div>
        <div style={{ color: "rgba(0,255,204,0.55)", fontSize: "0.6rem" }}>
          CMS Detector — Run 3
        </div>
      </div>

      {/* ── HUD: top-right event stats ── */}
      <div style={{ ...HUD_PANEL, top: "1.25rem", right: "1.25rem", minWidth: "240px" }}>
        <div style={{ marginBottom: "0.3rem", fontWeight: "bold", fontSize: "0.65rem", color: "rgba(0,255,204,0.5)" }}>
          EVENT DATA
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

      {/* ── HUD: bottom-left data source ── */}
      <div style={{ ...HUD_PANEL, bottom: "1.25rem", left: "1.25rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
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
          cursor: canFire ? "pointer" : "not-allowed",
          opacity: canFire ? 1 : 0.4,
          boxShadow:
            "0 0 16px rgba(0,255,204,0.35), inset 0 0 12px rgba(0,255,204,0.06)",
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
        ⬡ Fire Collision
      </button>
    </div>
  );
}
