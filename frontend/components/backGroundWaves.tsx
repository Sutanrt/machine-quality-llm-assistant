"use client";

import { useEffect, useRef } from "react";

type LayerConfig = {
  pointSpacingX: number;
  pointSpacingY: number;
  amp: number;
  speed: number;
  size: number;
  trailHeight: number;
  colorNear: string;
  colorFar: string;
  opacity: number;
  blur: number;
};

export function BackgroundWaves() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    // Fullscreen canvas
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    // Layer definitions (front -> back)
    const layers: LayerConfig[] = [
      {
        pointSpacingX: 28,
        pointSpacingY: 60,
        amp: 26,
        speed: 0.0016,
        size: 2.4,
        trailHeight: 48,
        colorNear: "#4dfff0",
        colorFar: "#005bff",
        opacity: 0.9,
        blur: 0,
      },
      {
        pointSpacingX: 32,
        pointSpacingY: 60,
        amp: 22,
        speed: 0.0012,
        size: 2.0,
        trailHeight: 64,
        colorNear: "#3399ff",
        colorFar: "#0022aa",
        opacity: 0.6,
        blur: 1.5,
      },
      {
        pointSpacingX: 36,
        pointSpacingY: 60,
        amp: 18,
        speed: 0.0008,
        size: 1.6,
        trailHeight: 80,
        colorNear: "#4455ff",
        colorFar: "#220066",
        opacity: 0.35,
        blur: 3,
      },
    ];

    // helper: lerp color hex a->b by t
    function lerpColor(a: string, b: string, t: number) {
      const ah = a.replace("#", "");
      const bh = b.replace("#", "");
      const ar = parseInt(ah.substring(0, 2), 16);
      const ag = parseInt(ah.substring(2, 4), 16);
      const ab = parseInt(ah.substring(4, 6), 16);
      const br = parseInt(bh.substring(0, 2), 16);
      const bg = parseInt(bh.substring(2, 4), 16);
      const bb = parseInt(bh.substring(4, 6), 16);

      const rr = Math.round(ar + (br - ar) * t);
      const rg = Math.round(ag + (bg - ag) * t);
      const rb = Math.round(ab + (bb - ab) * t);

      return `rgba(${rr},${rg},${rb},1)`;
    }

    // glow circle behind a point (radial gradient)
    function drawGlowCircle(
      ctx: CanvasRenderingContext2D,
      x: number,
      y: number,
      baseColor: string,
      radius: number,
      alpha: number
    ) {
      const g = ctx.createRadialGradient(x, y, 0, x, y, radius);
      g.addColorStop(0, baseColor.replace("1)", `${alpha})`)); // inner bright
      g.addColorStop(1, baseColor.replace("1)", "0)")); // fade out
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    // curved trail like a falling neon droplet
    function drawTrail(
      ctx: CanvasRenderingContext2D,
      x: number,
      y: number,
      h: number,
      color: string,
      alphaBase: number
    ) {
      // We'll draw multiple thin segments with fading alpha & slight x offset for curve
      const segments = 12;
      for (let i = 0; i < segments; i++) {
        const p = i / segments; // 0 -> 1
        const yy = y + p * h;
        // curve sideways a bit using sin => feels 3D-ish
        const xx = x + Math.sin(p * 2.5) * 4; // small horizontal wobble
        const segAlpha = alphaBase * (1 - p); // fade
        const segH = (h / segments) * 1.2;

        const grad = ctx.createLinearGradient(xx, yy, xx, yy + segH);
        grad.addColorStop(0, color.replace("1)", `${segAlpha})`));
        grad.addColorStop(1, color.replace("1)", "0)"));

        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.2 - p * 0.8; // thinner at bottom
        ctx.beginPath();
        ctx.moveTo(xx, yy);
        ctx.lineTo(xx, yy + segH);
        ctx.stroke();
      }
    }

    let t = 0;
    let rafId: number;

    function draw() {
      const canvas = canvasRef.current;
      if (!canvas) return; // ⬅️ guard penting

      const ctx = canvas.getContext("2d");
      if (!ctx) return; // ⬅️ guard penting

      const w = canvas.width;
      const h = canvas.height;

      // background fill
      ctx.fillStyle = "rgb(3,4,12)"; // deep black/navy
      ctx.fillRect(0, 0, w, h);

      layers.forEach((layer, layerIndex) => {
        const {
          pointSpacingX,
          pointSpacingY,
          amp,
          speed,
          size,
          trailHeight,
          colorNear,
          colorFar,
          opacity,
          blur,
        } = layer;

        // apply blur to this layer render
        if (blur > 0) {
          ctx.filter = `blur(${blur}px)`;
        } else {
          ctx.filter = "none";
        }

        // rows of waves
        for (let rowY = pointSpacingY * 0.5; rowY < h; rowY += pointSpacingY) {
          for (
            let colX = pointSpacingX * 0.5;
            colX < w;
            colX += pointSpacingX
          ) {
            // noise-ish wave: base sin + tiny wobble
            const wave =
              Math.sin(colX * 0.015 + t * speed * 2 + rowY * 0.03) * amp +
              Math.cos(rowY * 0.05 + t * speed * 3 + colX * 0.02) * amp * 0.3;

            const py = rowY + wave;

            // depth factor for color & alpha
            // layerIndex 0 = front (depthFactor ~1), back = smaller
            const depthFactor = 1 - layerIndex * 0.33;

            // lerp color front/back
            const baseColor = lerpColor(colorNear, colorFar, layerIndex * 0.4);

            // add glow
            drawGlowCircle(
              ctx,
              colX,
              py,
              baseColor,
              size * 4,
              0.18 * opacity * depthFactor
            );

            // draw main dot
            ctx.fillStyle = baseColor.replace(
              "1)",
              `${0.8 * opacity * depthFactor})`
            );
            ctx.beginPath();
            ctx.arc(colX, py, size, 0, Math.PI * 2);
            ctx.fill();

            // some points drop a trail (not all, to avoid terlalu ramai)
            // pakai pseudo-random stabil: based on colX,rowY
            const hash =
              ((colX * 13.13 + rowY * 7.77 + layerIndex * 99.9) % 10 + 10) % 10;
            if (hash < 3) {
              drawTrail(
                ctx,
                colX,
                py,
                trailHeight * (0.5 + depthFactor),
                baseColor,
                0.4 * opacity * depthFactor
              );
            }
          }
        }
      });

      ctx.filter = "none";
      t += 1;
      rafId = requestAnimationFrame(draw);
    }

    draw();

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full block"
      style={{
        background:
          "radial-gradient(circle at 20% 20%, rgba(0,80,255,0.15) 0%, rgba(0,0,0,0) 60%)",
      }}
    />
  );
}
