"use client";

import { useEffect, useRef, useState } from "react";
import { KAGGLE_GUIDE_URL, type KaggleStatus, type Project, engine, fileUrl, referenceUrl } from "@/lib/engine";

interface Props {
  project: Project;
  locked: boolean;
  run: (label: string, action: () => Promise<unknown>) => Promise<void>;
}

/** Etapa 4: referência do personagem → pacote para o Kaggle → importar o resultado do Wan Animate. */
export function WanPanel({ project, locked, run }: Props) {
  const [refVersion, setRefVersion] = useState(0);
  const [resultVersion, setResultVersion] = useState(0);
  const [resolution, setResolution] = useState<"480p" | "720p">("480p");
  const [fps, setFps] = useState<string>("16");
  const [prompt, setPrompt] = useState("");
  const [kaggle, setKaggle] = useState<KaggleStatus | null>(null);
  const [kaggleUser, setKaggleUser] = useState<string | null>(null);
  const [token, setToken] = useState("");
  const refInput = useRef<HTMLInputElement>(null);
  const resultInput = useRef<HTMLInputElement>(null);

  const trackedCurrent = project.tracked_revision !== null && project.tracked_revision === project.revision;
  const activeJob = project.active_job?.kind === "wan_package" ? project.active_job : null;
  const lastJob = project.jobs.wan_package;
  const wan = project.wan;


  const remoteJob = project.remote_job;
  const lastKaggle = project.jobs.kaggle;
  // Muda quando o pacote é refeito, um resultado é importado ou o Kaggle devolve um vídeo.
  const version = `${project.mask_version}-${lastJob?.updated_at ?? ""}-${lastKaggle?.updated_at ?? ""}-${resultVersion}`;
  useEffect(() => {
    engine.kaggleStatus().then(setKaggle, () => undefined);
  }, [remoteJob?.id]);

  function saveToken() {
    run("Conectando ao Kaggle…", async () => {
      const account = await engine.saveKaggleToken(token);
      setKaggleUser(account.username);
      setToken("");
      setKaggle(await engine.kaggleStatus());
    });
  }

  const blockers: string[] = [];
  if (!project.reference) blockers.push("envie a imagem do personagem");
  if (!trackedCurrent) blockers.push("rastreie a pessoa (etapa 2)");

  function pickReference(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    run("Enviando a imagem do personagem…", async () => {
      await engine.uploadReference(project.id, file);
      setRefVersion((v) => v + 1);
    });
  }

  function pickResult(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    run("Importando o resultado do Kaggle…", async () => {
      await engine.uploadWanResult(project.id, file);
      setResultVersion((v) => v + 1);
    });
  }

  return (
    <section className="card wan">
      <h2>4. Personagem com Wan Animate (Kaggle)</h2>
      <p className="muted small">
        A troca do personagem roda numa GPU gratuita do Kaggle. Aqui você prepara o pacote, roda o notebook
        lá e traz o vídeo de volta. <a href={KAGGLE_GUIDE_URL} target="_blank" rel="noreferrer">Passo a passo</a>
      </p>

      <div className="wan-grid">
        <div>
          <h3>Personagem</h3>
          {project.reference ? (
            <img className="reference" src={referenceUrl(project.id, refVersion)} alt="Imagem de referência" />
          ) : (
            <p className="muted small">Imagem do personagem criada no ChatGPT: corpo inteiro, fundo simples.</p>
          )}
          <input ref={refInput} type="file" accept="image/*" hidden onChange={pickReference} />
          <button disabled={locked} onClick={() => refInput.current?.click()}>
            {project.reference ? "Trocar imagem" : "Enviar imagem"}
          </button>
        </div>

        <div>
          <h3>Pacote para o Kaggle</h3>
          <label>
            Resolução
            <select value={resolution} onChange={(e) => setResolution(e.target.value as "480p" | "720p")}>
              <option value="480p">480p (recomendado na GPU grátis)</option>
              <option value="720p">720p (mais lento)</option>
            </select>
          </label>
          <label>
            Quadros por segundo
            <select value={fps} onChange={(e) => setFps(e.target.value)}>
              <option value="16">16 fps (recomendado: 5 s cabem em 1 segmento)</option>
              <option value="original">Original do clipe (até ~2x mais lento)</option>
            </select>
          </label>
          <label>
            Descrição do personagem (opcional)
            <input value={prompt} maxLength={1000} onChange={(e) => setPrompt(e.target.value)}
              placeholder="ex.: a cartoon astronaut dancing" />
          </label>
          {activeJob ? (
            <div className="job">
              <div className="job-head">
                <span>{activeJob.message ?? "Gerando pacote…"}</span>
                <span>{Math.round(activeJob.progress * 100)}%</span>
              </div>
              <div className="bar"><div style={{ width: `${activeJob.progress * 100}%` }} /></div>
            </div>
          ) : (
            <button
              className="primary"
              disabled={locked || blockers.length > 0}
              onClick={() =>
                run("Iniciando o pacote…", () =>
                  engine.wanPackage(project.id, { resolution, fps: fps === "original" ? null : Number(fps), prompt }),
                )
              }
            >
              Gerar pacote para o Kaggle
            </button>
          )}
          {blockers.length > 0 && <p className="status warn">Antes: {blockers.join(" e ")}.</p>}
          {lastJob?.status === "error" && !activeJob && <p className="status warn">Falhou: {lastJob.message}</p>}
          {lastJob?.status === "done" && !activeJob && <p className="muted small">{lastJob.message}</p>}
          {wan["wan_package.zip"] && (
            <p>
              <a href={fileUrl(project.id, "wan_package.zip", version)}>Baixar wan_package.zip</a>
            </p>
          )}
        </div>

        <div>
          <h3>Gerar no Kaggle</h3>
          {kaggle && !kaggle.installed && (
            <p className="status warn">Pacote kaggle não instalado no motor. Rode: bash scripts/setup.sh</p>
          )}
          {kaggle && !kaggle.configured ? (
            <>
              <p className="muted small">
                Cole o token de API do Kaggle (kaggle.com → Settings → API → Generate New Token) ou, se o Kaggle
                baixar um arquivo kaggle.json, o conteúdo dele. Fica salvo só neste computador.
              </p>
              <input type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder="Token ou conteúdo do kaggle.json" />
              <button disabled={locked || token.trim().length < 20} onClick={saveToken}>
                Salvar token
              </button>
            </>
          ) : (
            kaggle && (
              <p className="muted small">
                Kaggle conectado{kaggleUser ? ` como ${kaggleUser}` : ""}. A geração roda numa GPU T4 grátis
                (12 a 40 min) e o resultado volta sozinho para cá.
              </p>
            )
          )}
          {remoteJob ? (
            <div className="job">
              <div className="job-head">
                <span>{remoteJob.message ?? "Enviando ao Kaggle…"}</span>
                <span>{Math.round(remoteJob.progress * 100)}%</span>
              </div>
              <div className="bar"><div style={{ width: `${remoteJob.progress * 100}%` }} /></div>
            </div>
          ) : (
            <button
              className="primary"
              disabled={locked || !kaggle?.configured || !wan["wan_package.zip"] || Boolean(kaggle?.running)}
              onClick={() => run("Enviando ao Kaggle…", () => engine.runOnKaggle(project.id))}
            >
              Gerar no Kaggle
            </button>
          )}
          {!remoteJob && kaggle?.running && kaggle.running.project_id !== project.id && (
            <p className="status warn">Outro projeto está usando o Kaggle agora (um por vez).</p>
          )}
          {lastKaggle?.status === "error" && !remoteJob && <p className="status warn">Falhou: {lastKaggle.message}</p>}
          {lastKaggle?.status === "done" && !remoteJob && <p className="muted small">{lastKaggle.message}</p>}
        </div>

        <div>
          <h3>Resultado</h3>
          <p className="muted small">
            Chega aqui sozinho quando a geração no Kaggle termina. No modo manual, rode o notebook no Kaggle e importe o
            resultado.mp4.
          </p>
          <input ref={resultInput} type="file" accept="video/*" hidden onChange={pickResult} />
          <button disabled={locked || !wan["wan_input.mp4"]} onClick={() => resultInput.current?.click()}>
            Importar resultado
          </button>
          {wan["wan_final.mp4"] && (
            <ul className="downloads">
              <li>
                <a href={fileUrl(project.id, "wan_final.mp4", version)}>wan_final.mp4</a>{" "}
                <span className="muted small">personagem + áudio original</span>
              </li>
              <li>
                <a href={fileUrl(project.id, "comparativo.mp4", version)}>comparativo.mp4</a>{" "}
                <span className="muted small">antes | depois</span>
              </li>
            </ul>
          )}
        </div>
      </div>

      {wan["comparativo.mp4"] && (
        <video className="wan-compare" src={fileUrl(project.id, "comparativo.mp4", version)} controls playsInline />
      )}
    </section>
  );
}
