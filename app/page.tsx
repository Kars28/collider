"use client";

import { useMemo, useRef, useState, useCallback } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import { Line2 } from "three-stdlib";
import * as THREE from "three";

// ─── constants ───────────────────────────────────────────────────────────────

const COLORS = ["#00d4ff", "#ff006e", "#aaff00", "#ffffff"] as const;
const TRACK_COUNT = 30;
const TRACK_LENGTH = 6;
const TRACK_DURATION = 2;       // seconds for tracks to fully extend
const FLASH_DURATION = 0.3;     // seconds for impact flash
const BEAM_SPEED = 12;          // units per second — spheres accelerate toward center
const BEAM_START_X = 15;        // starting distance from center

// HUD data pools
const ENERGIES   = ["7.0 TeV", "8.0 TeV", "13.0 TeV", "13.6 TeV"] as const;
const EVENT_TYPES = ["pp → H + X", "pp → tt\u0304", "pp → W± + jets", "pp → Z⁰ + X"] as const;

function randomEventStats() {
  return {
    energy:    ENERGIES[Math.floor(Math.random() * ENERGIES.length)],
    particles: Math.floor(Math.random() * (47 - 18 + 1)) + 18,
    eventType: EVENT_TYPES[Math.floor(Math.random() * EVENT_TYPES.length)],
    timestamp: new Date().toLocaleTimeString("en-GB", { hour12: false }),
  };
}

// ─── types ───────────────────────────────────────────────────────────────────

type CollisionStage = "idle" | "beams" | "impact" | "tracks" | "showing";

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

// ─── CMS-style detector ───────────────────────────────────────────────────────

function CMSDetector() {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((_, delta) => {
    if (groupRef.current) groupRef.current.rotation.y += delta * 0.25;
  });

  return (
    <group ref={groupRef}>
      {/* Barrel — open cylinder along Z */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[1.5, 1.5, 4, 48, 5, true]} />
        <meshBasicMaterial color="#00ffcc" wireframe />
      </mesh>
      {/* Front endcap */}
      <mesh position={[0, 0, 2]}>
        <circleGeometry args={[1.5, 48]} />
        <meshBasicMaterial color="#00ffcc" wireframe side={THREE.DoubleSide} />
      </mesh>
      {/* Back endcap */}
      <mesh position={[0, 0, -2]}>
        <circleGeometry args={[1.5, 48]} />
        <meshBasicMaterial color="#00ffcc" wireframe side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

// ─── beam spheres ────────────────────────────────────────────────────────────
// Only visible during 'beams' stage. Moves toward center each frame.

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
    // Reset positions when entering 'beams' stage
    if (stage === "beams" && prevStage.current !== "beams") {
      x1.current = -BEAM_START_X;
      x2.current = BEAM_START_X;
    }
    prevStage.current = stage;

    if (stage !== "beams") return;

    // Accelerate toward center — ease-in by using distance-dependent speed
    const speed1 = BEAM_SPEED * (1 + (BEAM_START_X - Math.abs(x1.current)) / BEAM_START_X * 2);
    const speed2 = speed1;

    x1.current += speed1 * delta;  // moving right (from -15 toward 0)
    x2.current -= speed2 * delta;  // moving left  (from +15 toward 0)

    // Clamp so they don't overshoot
    if (x1.current > 0) x1.current = 0;
    if (x2.current < 0) x2.current = 0;

    if (group1.current) group1.current.position.x = x1.current;
    if (group2.current) group2.current.position.x = x2.current;

    // Impact when both spheres reach within 0.5 of center
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
// Bright white sphere at center, fades out over FLASH_DURATION.

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

function ParticleTracks({ stage, onDone }: { stage: CollisionStage; onDone: () => void }) {
  const elapsed = useRef(0);
  const prevStage = useRef<CollisionStage>("idle");
  const doneFired = useRef(false);

  const tracks = useMemo<TrackData[]>(
    () =>
      Array.from({ length: TRACK_COUNT }, () => ({
        dir: randomDirection(),
        color: COLORS[Math.floor(Math.random() * COLORS.length)],
      })),
    []
  );

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

  // Render during both 'tracks' (animating) and 'showing' (persistent)
  if (stage !== "tracks" && stage !== "showing") return null;

  return (
    <>
      {tracks.map((track, i) => (
        <Track key={i} dir={track.dir} color={track.color} />
      ))}
    </>
  );
}

// ─── collision controller (useFrame-driven state machine) ───────────────────
// This component lives inside the Canvas and drives the state transitions.

function CollisionController({
  stage,
  setStage,
  trackKey,
}: {
  stage: CollisionStage;
  setStage: (s: CollisionStage) => void;
  trackKey: number;
}) {
  const handleImpact = useCallback(() => setStage("impact"), [setStage]);
  const handleFlashDone = useCallback(() => setStage("tracks"), [setStage]);
  const handleTracksDone = useCallback(() => setStage("showing"), [setStage]);

  return (
    <>
      <BeamSpheres stage={stage} onImpact={handleImpact} />
      <CollisionFlash stage={stage} onDone={handleFlashDone} />
      <ParticleTracks key={trackKey} stage={stage} onDone={handleTracksDone} />
    </>
  );
}

// ─── scene root ──────────────────────────────────────────────────────────────

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

export default function Home() {
  const [stage, setStage] = useState<CollisionStage>("idle");
  const [trackKey, setTrackKey] = useState(0);
  const [eventStats, setEventStats] = useState({
    energy:    "-- TeV",
    particles: "--" as number | string,
    eventType: "READY",
    timestamp: "--:--:--",
  });

  const canFire = stage === "idle" || stage === "showing";
  const handleFire = useCallback(() => {
    if (!canFire) return;
    setEventStats(randomEventStats());   // new random stats each collision
    setTrackKey((k) => k + 1);
    setStage("beams");
  }, [canFire]);

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#000", position: "relative" }}>
      <Canvas gl={{ antialias: false }} camera={{ position: [0, 0, 7], fov: 60 }}>
        <ambientLight intensity={0.4} />
        <pointLight position={[5, 5, 5]} intensity={60} color="#ffffff" />

        <CMSDetector />
        <CollisionController stage={stage} setStage={setStage} trackKey={trackKey} />

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
      <div style={{ ...HUD_PANEL, top: "1.25rem", right: "1.25rem", minWidth: "210px" }}>
        <div style={{ marginBottom: "0.3rem", fontWeight: "bold", fontSize: "0.65rem", color: "rgba(0,255,204,0.5)" }}>EVENT DATA</div>
        <div><span style={{ color: "#aaa" }}>COLLISION ENERGY:</span> {eventStats.energy}</div>
        <div><span style={{ color: "#aaa" }}>PARTICLES DETECTED:</span> {eventStats.particles}</div>
        <div><span style={{ color: "#aaa" }}>EVENT TYPE:</span> {eventStats.eventType}</div>
        <div><span style={{ color: "#aaa" }}>TIMESTAMP:</span> {eventStats.timestamp}</div>
      </div>

      {/* ── HUD: bottom-left data source ── */}
      <div style={{ ...HUD_PANEL, bottom: "1.25rem", left: "1.25rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: "#0f0",
            display: "inline-block",
            animation: "hud-pulse 1.8s ease-in-out infinite",
          }}
        />
        <span style={{ color: "#aaa", fontSize: "0.6rem" }}>LIVE</span>
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
