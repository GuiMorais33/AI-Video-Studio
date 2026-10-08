"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type Click,
  type Health,
  type Job,
  type Project,
  engine,
  fileUrl,
  formatTime,
  fpsValue,
  overlayUrl,
} from "@/lib/engine";
import { WanPanel } from "./WanPanel";

type Mode = 1 | 0; // 1 = incluir, 0 = excluir

function JobProgress({ job, label }: { job: Job; label: string }) {
  return (
    <div className="job">
      <div className="job-head">
        <span>{job.message ?? label}</span>
        <span>{Math.round(job.progress * 100)}%</span>
      </div>
      <div className="bar">
        <div style={{ width: `${job.progress * 100}%` }} />
      </div>
    </div>
  );
}

function trackingStatus(p: Project): { text: string; tone: "muted" | "ok" | "warn" } {
  if (p.clicks.length === 0) return { text: "Clique na pessoa para começar.", tone: "muted" };
  if (p.tracked_revision === null) return { text: "Seleção pronta. Rastreie para cobrir o vídeo todo.", tone: "warn" };
  if (p.tracked_revision !== p.revision) return { text: "A seleção mudou desde o último rastreamento.", tone: "warn" };
  return { text: "Rastreamento atualizado.", tone: "ok" };
}

export default function ProjectPage() {
  const { id } = useParams<{ id: string }>();
  const [project, setProject] = useState<Project | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [frame, setFrame] = useState(0);
  const [mode, setMode] = useState<Mode>(1);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const originalRef = useRef<HTMLVideoElement>(null);
  const previewRef = useRef<HTMLVideoElement>(null);

  const refresh = useCallback(async () => {
    try {
      setProject(await engine.getProject(id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [id]);

  useEffect(() => {
    refresh();
    engine.health().then(setHealth, () => undefined);
  }, [refresh]);

  // Enquanto houver tarefa ativa, atualiza o progresso.
  const activeJob = project?.active_job ?? null;
  const activeJobId = activeJob?.id;
  // Geração no Kaggle leva dezenas de minutos: consulta mais espaçada.
  const remoteJobId = project?.remote_job?.id;
  useEffect(() => {
    if (!activeJobId && !remoteJobId) return;
    const timer = setInterval(refresh, activeJobId ? 700 : 5000);
    return () => clearInterval(timer);
  }, [activeJobId, remoteJobId, refresh]);

  const lastFrame = (project?.num_frames ?? 1) - 1;
  const goTo = useCallback((idx: number) => setFrame(Math.max(0, Math.min(lastFrame, idx))), [lastFrame]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement && e.target.type !== "range") return;
      const step = e.shiftKey ? 10 : 1;
      if (e.key === "ArrowRight") goTo(frame + step);
      else if (e.key === "ArrowLeft") goTo(frame - step);
      else return;
      e.preventDefault();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frame, goTo]);

  async function run(label: string, action: () => Promise<unknown>) {
    setBusy(label);
    setError(null);
    try {
      await action();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
      await refresh();
    }
  }

  function onStageClick(e: React.MouseEvent<SVGSVGElement>, label: Mode) {
    e.preventDefault();
    if (!project || busy || activeJob) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * project.width;
    const y = ((e.clientY - rect.top) / rect.height) * project.height;
    const firstUse = health?.tracker === "sam2" && !health.tracker_loaded;
    run(
      firstUse
        ? "Carregando o SAM 2 e o vídeo (na primeira vez pode levar alguns minutos em CPU)…"
        : "Calculando máscara…",
      async () => {
        await engine.addClick(project.id, { frame_idx: frame, x, y, label });
        if (firstUse) engine.health().then(setHealth, () => undefined);
      },
    );
  }

  if (!project) {
    return error ? <div className="banner error">{error}</div> : <p className="muted">Carregando projeto…</p>;
  }

  const fps = fpsValue(project.fps);
  const clicksHere = project.clicks.filter((c) => c.frame_idx === frame);
  const clickFrames = [...new Set(project.clicks.map((c) => c.frame_idx))].sort((a, b) => a - b);
  const status = trackingStatus(project);
  const locked = Boolean(busy || activeJob);
  const markerR = Math.max(project.width, project.height) * 0.009;
  const imageVersion = `${project.mask_version}-${activeJob ? Math.floor(activeJob.progress * 50) : "idle"}`;
  const exported = project.exports["preview.mp4"];
  const exportCurrent = exported && project.exported_mask_version === project.mask_version;
  const lastTrack = project.jobs.track;
  const lastExport = project.jobs.export;

  function playBoth() {
    for (const v of [originalRef.current, previewRef.current]) {
      if (!v) continue;
      v.currentTime = 0;
      void v.play();
    }
  }

  return (
    <div className="editor">
      <div className="editor-head">
        <div>
          <Link href="/" className="muted">
            ← Projetos
          </Link>
          <h1>{project.name}</h1>
          <p className="muted">
            {project.width}×{project.height} · {fps.toFixed(2)} fps · {project.num_frames} quadros ·{" "}
            {formatTime(project.duration)} {project.has_audio ? "· com áudio" : "· sem áudio"}
          </p>
        </div>
      </div>

      {health?.tracker === "demo" && (
        <div className="banner warn">Modo DEMONSTRAÇÃO: máscaras geométricas, sem IA.</div>
      )}
      {error && (
        <div className="banner error" onClick={() => setError(null)}>
          {error} <span className="muted">(clique para fechar)</span>
        </div>
      )}

      <div className="editor-body">
        <section className="viewer">
          <div
            className="stage"
            style={{
              aspectRatio: `${project.width} / ${project.height}`,
              // Cabe na largura disponível e em 72% da altura da tela, mantendo a proporção.
              width: `min(100%, calc(72vh * ${project.width / project.height}))`,
            }}
          >
            <img src={overlayUrl(project.id, frame, imageVersion)} alt={`Quadro ${frame}`} draggable={false} />
            <svg
              viewBox={`0 0 ${project.width} ${project.height}`}
              className={locked ? "locked" : mode === 1 ? "include" : "exclude"}
              onClick={(e) => onStageClick(e, e.shiftKey ? ((1 - mode) as Mode) : mode)}
              onContextMenu={(e) => onStageClick(e, 0)}
            >
              {clicksHere.map((c: Click) => (
                <circle
                  key={c.id}
                  cx={c.x}
                  cy={c.y}
                  r={markerR}
                  className={c.label === 1 ? "pos" : "neg"}
                  strokeWidth={markerR * 0.35}
                />
              ))}
            </svg>
            {busy && <div className="stage-busy">{busy}</div>}
          </div>

          <div className="timeline">
            <button onClick={() => goTo(frame - 1)} disabled={frame === 0} aria-label="Quadro anterior">
              ◀
            </button>
            <input
              type="range"
              min={0}
              max={lastFrame}
              value={frame}
              onChange={(e) => goTo(Number(e.target.value))}
            />
            <button onClick={() => goTo(frame + 1)} disabled={frame === lastFrame} aria-label="Próximo quadro">
              ▶
            </button>
            <span className="frame-label">
              Quadro {frame + 1}/{project.num_frames} · {formatTime(frame / fps)}
            </span>
          </div>
          <p className="muted small">
            Clique = {mode === 1 ? "incluir" : "excluir"} · Shift+clique = {mode === 1 ? "excluir" : "incluir"} ·
            botão direito = excluir · ← → navegam (Shift = 10 quadros)
          </p>
        </section>

        <aside className="panel">
          <div className="card">
            <h2>1. Selecionar a pessoa</h2>
            <div className="segmented">
              <button className={mode === 1 ? "active include" : ""} onClick={() => setMode(1)}>
                + Incluir
              </button>
              <button className={mode === 0 ? "active exclude" : ""} onClick={() => setMode(0)}>
                − Excluir
              </button>
            </div>
            <p className="muted small">
              Clique no corpo da pessoa. Se a máscara pegar algo a mais, use Excluir sobre a área errada. Corrija
              em qualquer quadro e rastreie de novo.
            </p>
            <div className="row">
              <button
                disabled={locked || project.clicks.length === 0}
                onClick={() => run("Desfazendo…", () => engine.undoClick(project.id))}
              >
                Desfazer
              </button>
              <button
                className="danger"
                disabled={locked || project.clicks.length === 0}
                onClick={() =>
                  confirm("Remover todos os cliques e máscaras?") &&
                  run("Limpando…", () => engine.clearClicks(project.id))
                }
              >
                Limpar tudo
              </button>
            </div>
            {clickFrames.length > 0 && (
              <div className="chips">
                <span className="muted small">Quadros com cliques:</span>
                {clickFrames.map((f) => (
                  <button key={f} className={`chip ${f === frame ? "active" : ""}`} onClick={() => goTo(f)}>
                    {f + 1}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="card">
            <h2>2. Rastrear no vídeo</h2>
            <p className={`status ${status.tone}`}>{status.text}</p>
            {activeJob?.kind === "track" ? (
              <JobProgress job={activeJob} label="Rastreando…" />
            ) : (
              <button
                className="primary"
                disabled={locked || project.clicks.length === 0}
                onClick={() => run("Iniciando rastreamento…", () => engine.track(project.id))}
              >
                Rastrear pessoa
              </button>
            )}
            {lastTrack?.status === "error" && <p className="status warn">Falhou: {lastTrack.message}</p>}
            {lastTrack?.status === "done" && !activeJob && <p className="muted small">{lastTrack.message}</p>}
          </div>

          <div className="card">
            <h2>3. Exportar prévia</h2>
            {activeJob?.kind === "export" ? (
              <JobProgress job={activeJob} label="Exportando…" />
            ) : (
              <button
                className="primary"
                disabled={locked || project.objects.length === 0}
                onClick={() => run("Iniciando exportação…", () => engine.exportProject(project.id))}
              >
                {exported ? "Exportar novamente" : "Exportar prévia"}
              </button>
            )}
            {lastExport?.status === "error" && <p className="status warn">Falhou: {lastExport.message}</p>}
            {exported && (
              <>
                <p className={`status ${exportCurrent ? "ok" : "warn"}`}>
                  {exportCurrent ? "Exportação atualizada." : "As máscaras mudaram depois da última exportação."}
                </p>
                <ul className="downloads">
                  <li>
                    <a href={fileUrl(project.id, "preview.mp4", project.exported_mask_version ?? 0)}>
                      preview.mp4
                    </a>{" "}
                    <span className="muted small">pessoa destacada + áudio</span>
                  </li>
                  <li>
                    <a href={fileUrl(project.id, "mask.mp4", project.exported_mask_version ?? 0)}>mask.mp4</a>{" "}
                    <span className="muted small">máscara preto e branco</span>
                  </li>
                  <li>
                    <a href={fileUrl(project.id, "masks.zip", project.exported_mask_version ?? 0)}>masks.zip</a>{" "}
                    <span className="muted small">PNG por quadro</span>
                  </li>
                </ul>
              </>
            )}
          </div>
        </aside>
      </div>

      <WanPanel project={project} locked={locked} run={run} />

      {exported && (
        <section className="compare card">
          <div className="compare-head">
            <h2>Antes / depois</h2>
            <button onClick={playBoth}>▶ Reproduzir juntos</button>
          </div>
          <div className="compare-grid">
            <figure>
              <video ref={originalRef} src={fileUrl(project.id, "clip.mp4")} controls playsInline />
              <figcaption>Original</figcaption>
            </figure>
            <figure>
              <video
                ref={previewRef}
                src={fileUrl(project.id, "preview.mp4", project.exported_mask_version ?? 0)}
                controls
                muted
                playsInline
              />
              <figcaption>Máscaras SAM 2</figcaption>
            </figure>
          </div>
        </section>
      )}
    </div>
  );
}
