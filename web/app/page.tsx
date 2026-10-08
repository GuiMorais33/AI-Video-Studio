"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import {
  EngineError,
  type Health,
  type ProjectSummary,
  engine,
  formatTime,
  frameUrl,
  uploadProject,
} from "@/lib/engine";

function EngineStatus({ health, error }: { health: Health | null; error: string | null }) {
  if (error) return <div className="banner error">{error}</div>;
  if (!health) return <div className="banner">Conectando ao motor…</div>;
  const problems: string[] = [];
  if (!health.ffmpeg) problems.push("FFmpeg não encontrado");
  if (health.tracker === "sam2" && !health.sam2.installed) problems.push("SAM 2 não instalado (rode scripts/setup.sh)");
  if (health.tracker === "sam2" && !health.sam2.checkpoint_present)
    problems.push(`checkpoint SAM 2.1 ${health.sam2.model} ausente (rode: studio models download)`);
  return (
    <>
      {health.tracker === "demo" && (
        <div className="banner warn">
          Modo DEMONSTRAÇÃO: as máscaras são círculos geométricos, sem IA. Use apenas para testar a interface.
        </div>
      )}
      {problems.length > 0 && <div className="banner error">{problems.join(" · ")}</div>}
      <div className="status-row">
        <span className="dot ok" /> Motor v{health.version} · rastreador{" "}
        {health.tracker === "sam2" ? `SAM 2.1 ${health.sam2.model}` : "demo"}
        {health.device && ` · ${health.device}`}
      </div>
    </>
  );
}

function UploadForm({ health, onCreated }: { health: Health | null; onCreated: (id: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [start, setStart] = useState("0");
  const [seconds, setSeconds] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const maxSeconds = health?.limits.max_seconds ?? 15;
  const defaultSeconds = health?.limits.default_seconds ?? 5;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    if (name.trim()) form.append("name", name.trim());
    form.append("start", start || "0");
    form.append("seconds", seconds || String(defaultSeconds));
    setError(null);
    setProgress(0);
    try {
      const project = await uploadProject(form, setProgress);
      onCreated(project.id);
    } catch (err) {
      setError(err instanceof EngineError ? err.message : String(err));
      setProgress(null);
    }
  }

  const uploading = progress !== null;
  return (
    <form className="card upload" onSubmit={submit}>
      <h2>Novo projeto</h2>
      <p className="muted">
        Use um vídeo seu ou de alguém que autorizou. O trecho é reduzido para no máximo {health?.limits.max_side ?? 854}px
        e {health?.limits.max_fps ?? 30} fps para caber no processamento em CPU.
      </p>
      <label className="file">
        <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} disabled={uploading} />
        <span>{file ? file.name : "Escolher vídeo…"}</span>
      </label>
      <div className="grid3">
        <label>
          Nome
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Transformação 001" />
        </label>
        <label>
          Início (s)
          <input type="number" min={0} step={0.1} value={start} onChange={(e) => setStart(e.target.value)} />
        </label>
        <label>
          Duração (s)
          <input
            type="number"
            min={0.5}
            max={maxSeconds}
            step={0.5}
            value={seconds}
            placeholder={String(defaultSeconds)}
            onChange={(e) => setSeconds(e.target.value)}
          />
        </label>
      </div>
      {error && <div className="banner error">{error}</div>}
      <button className="primary" disabled={!file || uploading}>
        {!uploading
          ? "Criar projeto"
          : progress < 1
            ? `Enviando… ${Math.round(progress * 100)}%`
            : "Preparando clipe e extraindo quadros…"}
      </button>
    </form>
  );
}

export default function Home() {
  const router = useRouter();
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [h, list] = await Promise.all([engine.health(), engine.listProjects()]);
      setHealth(h);
      setProjects(list);
      setHealthError(null);
    } catch (err) {
      setHealthError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function remove(p: ProjectSummary) {
    if (!confirm(`Excluir o projeto "${p.name}" e todos os arquivos dele?`)) return;
    try {
      await engine.deleteProject(p.id);
    } catch (err) {
      alert(err instanceof Error ? err.message : String(err));
    }
    refresh();
  }

  return (
    <div className="home">
      <EngineStatus health={health} error={healthError} />
      <UploadForm health={health} onCreated={(id) => router.push(`/projects/${id}`)} />
      <section>
        <h2>Projetos</h2>
        {projects === null && !healthError && <p className="muted">Carregando…</p>}
        {projects?.length === 0 && <p className="muted">Nenhum projeto ainda.</p>}
        <div className="project-grid">
          {projects?.map((p) => (
            <article key={p.id} className="card project">
              <Link href={`/projects/${p.id}`}>
                <img src={frameUrl(p.id, 0)} alt="" />
                <h3>{p.name}</h3>
              </Link>
              <p className="muted">
                {p.width}×{p.height} · {p.num_frames} quadros · {formatTime(p.duration)}
              </p>
              <p className="muted">
                {p.tracked_revision === null
                  ? "Ainda não rastreado"
                  : p.tracked_revision === p.revision
                    ? "Rastreado"
                    : "Rastreamento desatualizado"}
                {p.exported_mask_version !== null && " · exportado"}
              </p>
              <button className="link danger" onClick={() => remove(p)}>
                Excluir
              </button>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
