// Cliente da API local do motor (engine/studio/api.py).

export const ENGINE_URL = (process.env.NEXT_PUBLIC_ENGINE_URL ?? "http://127.0.0.1:8765").replace(/\/$/, "");

export type JobStatus = "queued" | "running" | "done" | "error";

export interface Job {
  id: string;
  project_id: string;
  kind: "track" | "export" | "wan_package" | "kaggle";
  status: JobStatus;
  progress: number;
  message: string | null;
  created_at: string;
  updated_at: string;
}

export interface Click {
  id: number;
  obj_id: number;
  frame_idx: number;
  x: number;
  y: number;
  label: 0 | 1;
  created_at: string;
}

export interface ProjectSummary {
  id: string;
  name: string;
  source_name: string;
  created_at: string;
  clip_start: number;
  width: number;
  height: number;
  fps: string;
  num_frames: number;
  duration: number;
  has_audio: number;
  revision: number;
  tracked_revision: number | null;
  mask_version: number;
  exported_mask_version: number | null;
}

export interface Project extends ProjectSummary {
  clicks: Click[];
  objects: number[];
  active_job: Job | null;
  remote_job: Job | null;
  jobs: Partial<Record<Job["kind"], Job>>;
  exports: Record<"preview.mp4" | "mask.mp4" | "masks.zip", boolean>;
  reference: boolean;
  wan: Record<WanFile, boolean>;
}

export type WanFile = "wan_package.zip" | "wan_input.mp4" | "wan_result.mp4" | "wan_final.mp4" | "comparativo.mp4";

export interface KaggleStatus {
  configured: boolean;
  source: string | null;
  installed: boolean;
  running: Job | null;
}

export interface WanPackageOptions {
  resolution: "480p" | "720p";
  fps: number | null;
  prompt: string;
}

export interface Health {
  status: string;
  version: string;
  tracker: "sam2" | "demo";
  tracker_loaded: boolean;
  device: string | null;
  sam2: { installed: boolean; model: string; checkpoint_present: boolean };
  ffmpeg: boolean;
  limits: { default_seconds: number; max_seconds: number; max_side: number; max_fps: number };
}

export class EngineError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${ENGINE_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new EngineError(`Motor fora do ar em ${ENGINE_URL}. Inicie com: bash scripts/start.sh`, 0);
  }
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* corpo não é JSON */
    }
    throw new EngineError(detail, resp.status);
  }
  return (resp.status === 204 ? undefined : await resp.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const engine = {
  health: () => request<Health>("/health"),
  listProjects: () => request<ProjectSummary[]>("/projects"),
  getProject: (id: string) => request<Project>(`/projects/${id}`),
  deleteProject: (id: string) => request<void>(`/projects/${id}`, { method: "DELETE" }),
  addClick: (id: string, click: { frame_idx: number; x: number; y: number; label: 0 | 1; obj_id?: number }) =>
    request<{ click: Click; mask_area: number; mask_version: number }>(`/projects/${id}/clicks`, json(click)),
  undoClick: (id: string) => request<{ removed: Click | null }>(`/projects/${id}/clicks/undo`, { method: "POST" }),
  clearClicks: (id: string) => request<{ ok: boolean }>(`/projects/${id}/clicks`, { method: "DELETE" }),
  track: (id: string) => request<Job>(`/projects/${id}/track`, { method: "POST" }),
  exportProject: (id: string) => request<Job>(`/projects/${id}/export`, { method: "POST" }),
  uploadReference: (id: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ width: number; height: number }>(`/projects/${id}/reference`, { method: "POST", body: form });
  },
  wanPackage: (id: string, options: WanPackageOptions) => request<Job>(`/projects/${id}/wan-package`, json(options)),
  kaggleStatus: () => request<KaggleStatus>("/kaggle"),
  saveKaggleToken: (token: string) =>
    request<{ username: string; quota: string | null }>("/kaggle/token", json({ token })),
  runOnKaggle: (id: string) => request<Job>(`/projects/${id}/kaggle`, { method: "POST" }),
  uploadWanResult: (id: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ message: string; project: Project }>(`/projects/${id}/wan-result`, { method: "POST", body: form });
  },
};

export const referenceUrl = (id: string, version: string | number) =>
  `${ENGINE_URL}/projects/${id}/reference?v=${version}`;

export const KAGGLE_GUIDE_URL = "https://github.com/GuiMorais33/AI-Video-Studio/blob/HEAD/docs/KAGGLE.md";

export const frameUrl = (id: string, idx: number) => `${ENGINE_URL}/projects/${id}/frames/${idx}`;
export const overlayUrl = (id: string, idx: number, version: string | number) =>
  `${ENGINE_URL}/projects/${id}/overlay/${idx}?v=${version}`;
export const fileUrl = (id: string, name: string, version?: string | number) =>
  `${ENGINE_URL}/projects/${id}/files/${name}${version === undefined ? "" : `?v=${version}`}`;

/** Envia o vídeo com progresso de upload (fetch não informa progresso de envio). */
export function uploadProject(
  form: FormData,
  onProgress: (fraction: number) => void,
): Promise<Project> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${ENGINE_URL}/projects`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onerror = () => reject(new EngineError(`Motor fora do ar em ${ENGINE_URL}.`, 0));
    xhr.onload = () => {
      let body: { detail?: unknown } = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* resposta vazia */
      }
      if (xhr.status === 201) resolve(body as Project);
      else reject(new EngineError(typeof body.detail === "string" ? body.detail : `Erro ${xhr.status}`, xhr.status));
    };
    xhr.send(form);
  });
}

export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}

export function fpsValue(fps: string): number {
  const [num, den] = fps.split("/").map(Number);
  return den ? num / den : num;
}
