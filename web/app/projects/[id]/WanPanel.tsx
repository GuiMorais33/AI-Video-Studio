"use client";

import { useEffect, useRef, useState } from "react";
import {
  KAGGLE_GUIDE_URL,
  MAX_FACE_PHOTOS,
  type FaceSwapper,
  type KaggleAccount,
  type KaggleStatus,
  type Project,
  engine,
  faceUrl,
  fileUrl,
  fpsValue,
  referenceUrl,
} from "@/lib/engine";

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
  const [faceOnReference, setFaceOnReference] = useState(true);
  const [kaggle, setKaggle] = useState<KaggleStatus | null>(null);
  const [account, setAccount] = useState<KaggleAccount | null>(null);
  const [token, setToken] = useState("");
  const refInput = useRef<HTMLInputElement>(null);
  const facesInput = useRef<HTMLInputElement>(null);
  const resultInput = useRef<HTMLInputElement>(null);

  const trackedCurrent = project.tracked_revision !== null && project.tracked_revision === project.revision;
  const activeJob = project.active_job?.kind === "wan_package" ? project.active_job : null;
  const lastJob = project.jobs.wan_package;
  const wan = project.wan;


  const remoteJob = project.remote_job;
  const lastKaggle = project.jobs.kaggle;
  const faceJob = project.active_job?.kind === "face_refine" ? project.active_job : null;
  const lastFaceJob = project.jobs.face_refine;
  const faceRefine = project.face_refine;
  const refined = faceRefine.result;
  const compareJob = project.active_job?.kind === "face_compare" ? project.active_job : null;
  const lastCompareJob = project.jobs.face_compare;
  const compare = faceRefine.compare;
  const faceSettings = faceRefine.settings;
  const finalInfo = project.wan_final_info;
  // Muda quando o pacote é refeito, um resultado é importado, o Kaggle devolve um vídeo ou o refino
  // do rosto termina (o wan_final.mp4 e o comparativo.mp4 são refeitos).
  const version = [
    project.mask_version,
    lastJob?.updated_at ?? "",
    // A tarefa do Kaggle muda a cada consulta de progresso: só conta quando termina.
    lastKaggle && lastKaggle.status !== "queued" && lastKaggle.status !== "running" ? lastKaggle.updated_at : "",
    refined?.concluido_em ?? "",
    resultVersion,
  ].join("-");
  useEffect(() => {
    engine.kaggleStatus().then(setKaggle, () => undefined);
  }, [remoteJob?.id]);
  // Usuário e cota de GPU: ao abrir e depois de cada geração no Kaggle.
  useEffect(() => {
    if (kaggle?.configured) engine.kaggleAccount().then(setAccount, () => undefined);
  }, [kaggle?.configured, lastKaggle?.updated_at]);

  function saveToken() {
    run("Conectando ao Kaggle…", async () => {
      setAccount(await engine.saveKaggleToken(token));
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

  function pickFaces(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    if (!files.length) return;
    run(files.length > 1 ? `Enviando ${files.length} fotos do rosto…` : "Enviando a foto do rosto…", () =>
      engine.uploadFaces(project.id, files),
    );
  }

  function removeFace(name: string) {
    run("Removendo a foto…", () => engine.deleteFace(project.id, name));
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
        lá e traz o vídeo de volta; o rosto é refinado depois, aqui neste computador.{" "}
        <a href={KAGGLE_GUIDE_URL} target="_blank" rel="noreferrer">Passo a passo</a>
      </p>

      <div className="wan-grid">
        <div>
          <h3>Personagem</h3>
          {project.reference ? (
            <img className="reference" src={referenceUrl(project.id, refVersion)} alt="Imagem de referência" />
          ) : (
            <p className="muted small">
              Foto de corpo inteiro (sua ou do personagem criado no ChatGPT), de frente e com fundo simples. Corpo,
              roupa e cabelo vêm dela.
            </p>
          )}
          <input ref={refInput} type="file" accept="image/*" hidden onChange={pickReference} />
          <button disabled={locked} onClick={() => refInput.current?.click()}>
            {project.reference ? "Trocar imagem" : "Enviar imagem"}
          </button>

          <h3>Fotos do rosto</h3>
          <p className="muted small">
            3 a 6 fotos nítidas do rosto: de frente, meio perfil, sorrindo, com boa luz. O rosto do vídeo final fica
            igual a elas. Sem fotos aqui, o rosto vem da imagem acima.
          </p>
          <p className="muted small">
            As fotos ficam neste computador: o refino do rosto roda aqui depois que o vídeo volta do Kaggle. Mudou as
            fotos? Basta usar &quot;Refazer o refino do rosto&quot;, sem gerar no Kaggle de novo.
          </p>
          {project.faces.length > 0 && (
            <div className="faces">
              {project.faces.map((name) => (
                <figure key={name}>
                  <img src={faceUrl(project.id, name)} alt={`Foto do rosto ${name}`} />
                  <button className="link" disabled={locked} onClick={() => removeFace(name)}>
                    remover
                  </button>
                </figure>
              ))}
            </div>
          )}
          <input ref={facesInput} type="file" accept="image/*" multiple hidden onChange={pickFaces} />
          <button disabled={locked || project.faces.length >= MAX_FACE_PHOTOS} onClick={() => facesInput.current?.click()}>
            {project.faces.length ? "Adicionar fotos do rosto" : "Enviar fotos do rosto"}
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
            Descrição do personagem (opcional, em inglês)
            <input value={prompt} maxLength={1000} onChange={(e) => setPrompt(e.target.value)}
              placeholder="ex.: a cartoon astronaut in a white suit and orange helmet, dancing" />
          </label>
          <p className="muted small">
            Descrever roupa e materiais deixa o resultado mais fiel à imagem. Gere o pacote de novo depois de trocar a
            imagem do personagem.
          </p>
          <label className="check">
            <input
              type="checkbox"
              checked={faceOnReference}
              disabled={project.faces.length === 0 || !faceRefine.installed}
              onChange={(e) => setFaceOnReference(e.target.checked)}
            />
            Pôr o rosto das suas fotos na imagem do personagem
          </label>
          <p className="muted small">
            {project.faces.length === 0
              ? "Envie as fotos do rosto para usar isto."
              : !faceRefine.installed
                ? "Precisa do refino do rosto instalado."
                : "Feito aqui no seu computador (~20 s): o Wan já gera o seu rosto, inclusive de perfil, onde a troca de rosto não entra. O Kaggle recebe só a imagem pronta. Com fundo liso, a imagem também é recortada em volta da pessoa, para o rosto ocupar mais pixels."}
          </p>
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
                  engine.wanPackage(project.id, {
                    resolution,
                    fps: fps === "original" ? null : Number(fps),
                    prompt,
                    rosto_na_referencia: faceOnReference && project.faces.length > 0 && faceRefine.installed,
                  }),
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
          {wan["referencia_wan.png"] && (
            <figure className="reference-sent">
              <img src={fileUrl(project.id, "referencia_wan.png", version)} alt="Imagem do personagem que vai para o Kaggle" />
              <figcaption className="muted small">Imagem do personagem que vai para o Kaggle</figcaption>
            </figure>
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
                Kaggle conectado{account ? ` como ${account.username}` : ""}. A geração roda numa GPU T4 grátis
                (cerca de 35 min para 5 s de vídeo) e o resultado volta sozinho para cá. Depois, o rosto é refinado
                neste computador, na CPU (cerca de 10 a 20 min para 5 s de vídeo).
                {account?.quota && (
                  <>
                    {" "}
                    GPU nesta semana: {formatHours(account.quota.gpu_hours_used)} de{" "}
                    {formatHours(account.quota.gpu_hours_total)} usadas
                    {account.quota.refresh ? ` (renova em ${formatDate(account.quota.refresh)})` : ""}.
                  </>
                )}
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
              disabled={
                locked || !kaggle?.configured || !wan["wan_package.zip"] || project.wan_package_stale || Boolean(kaggle?.running)
              }
              onClick={() => run("Enviando ao Kaggle…", () => engine.runOnKaggle(project.id))}
            >
              Gerar no Kaggle
            </button>
          )}
          {!remoteJob && project.wan_package_stale && (
            <p className="status warn">
              A imagem do personagem ou as fotos do rosto mudaram: gere o pacote de novo antes de enviar.
            </p>
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
                <span className="muted small">
                  {refined?.ok ? "rosto refinado, " : ""}tamanho dos Reels + áudio original
                  {finalInfo?.fundo === "original" ? ", sobre o seu vídeo em resolução cheia" : ""}
                  {finalInfo?.fps ? `, ${Math.round(fpsValue(finalInfo.fps))} fps` : ""}
                </span>
              </li>
              <li>
                <a href={fileUrl(project.id, "comparativo.mp4", version)}>comparativo.mp4</a>{" "}
                <span className="muted small">antes | depois</span>
              </li>
            </ul>
          )}

          {finalInfo?.sem_interpolacao && finalInfo.sem_interpolacao !== "interpolação desligada" && (
            <p className="status warn">
              O vídeo final ficou com o movimento do Wan ({Math.round(fpsValue(finalInfo.fps ?? "16"))} fps), sem os
              quadros intermediários: {finalInfo.sem_interpolacao}.
            </p>
          )}
          {finalInfo?.fundo === "reduzido" && (
            <p className="status warn">
              O fundo do vídeo final saiu ampliado do vídeo do Wan, não do seu vídeo original: {finalInfo.motivo}.
            </p>
          )}

          <h3>Refino do rosto</h3>
          {faceRefine.installed ? (
            <p className="muted small">
              Troca o rosto do vídeo pelo das fotos, aqui na CPU, assim que o vídeo chega (cerca de 10 a 20 min para 5
              s de vídeo; o primeiro refino também baixa ~1,7 GB de modelos).
            </p>
          ) : (
            <p className="status warn">
              {faceRefine.message ?? "Refino do rosto não instalado neste computador."} Sem ele, o vídeo final fica com o
              rosto gerado pelo Wan.
            </p>
          )}
          {faceJob ? (
            <div className="job">
              <div className="job-head">
                <span>{faceJob.message ?? "Refinando o rosto…"}</span>
                <span>{Math.round(faceJob.progress * 100)}%</span>
              </div>
              <div className="bar"><div style={{ width: `${faceJob.progress * 100}%` }} /></div>
            </div>
          ) : (
            <>
              {lastFaceJob?.status === "error" && <p className="status warn">Falhou: {lastFaceJob.message}</p>}
              {refined?.ok && (
                <p className="status ok">
                  Rosto refinado com {refined.fotos} foto(s) em {formatDuration(refined.segundos)}
                  {refined.trocador ? ` (${settingsLabel(refined.trocador, refined.restaurador_forca ?? 80)})` : ""}.
                  {refined.avisos?.map((aviso) => ` Aviso: ${aviso}.`).join("")}
                </p>
              )}
              {refined && !refined.ok && (
                <p className="status warn">
                  Refino do rosto não aplicado: {refined.motivo}. O wan_final.mp4 está sem o refino.
                  {wan["facefusion_rosto.log"] && (
                    <>
                      {" "}
                      <a href={fileUrl(project.id, "facefusion_rosto.log", version)}>Log do FaceFusion</a>
                    </>
                  )}
                </p>
              )}
              {!refined && wan["wan_result.mp4"] && faceRefine.installed && lastFaceJob?.status !== "error" && (
                <p className="muted small">Este vídeo ainda não passou pelo refino do rosto.</p>
              )}
              <button
                disabled={locked || !faceRefine.installed || !wan["wan_result.mp4"] || Boolean(remoteJob)}
                onClick={() => run("Iniciando o refino do rosto…", () => engine.faceRefine(project.id))}
              >
                {refined || lastFaceJob ? "Refazer o refino do rosto" : "Refinar o rosto"}
              </button>
            </>
          )}

          <h4>Comparar ajustes do rosto</h4>
          <p className="muted small">
            Ajuste atual: {settingsLabel(faceSettings.trocador, faceSettings.restaurador)}. A comparação testa os três
            trocadores e a força do restaurador num trecho de ~1,5 s com o rosto de frente, lado a lado e ampliado no
            rosto (cerca de 10 min na CPU; a primeira vez baixa ~0,8 GB de modelos).
          </p>
          {compareJob ? (
            <div className="job">
              <div className="job-head">
                <span>{compareJob.message ?? "Comparando os ajustes…"}</span>
                <span>{Math.round(compareJob.progress * 100)}%</span>
              </div>
              <div className="bar"><div style={{ width: `${compareJob.progress * 100}%` }} /></div>
            </div>
          ) : (
            <>
              {lastCompareJob?.status === "error" && (
                <p className="status warn">
                  Comparação falhou: {lastCompareJob.message}
                  {wan["facefusion_comparar.log"] && (
                    <>
                      {" "}
                      <a href={fileUrl(project.id, "facefusion_comparar.log", lastCompareJob.updated_at)}>Log</a>
                    </>
                  )}
                </p>
              )}
              <button
                disabled={locked || !faceRefine.installed || !wan["wan_result.mp4"] || Boolean(remoteJob)}
                onClick={() => run("Iniciando a comparação…", () => engine.faceCompare(project.id))}
              >
                {compare ? "Comparar de novo" : "Comparar ajustes do rosto"}
              </button>
            </>
          )}
          {compare && wan["comparar_rosto.mp4"] && (
            <div className="face-compare">
              <video
                src={fileUrl(project.id, "comparar_rosto.mp4", compare.concluido_em)}
                autoPlay
                loop
                muted
                playsInline
                controls
              />
              <p className="muted small">
                Qual ficou mais natural e mais parecido com você? Escolha e depois clique em &quot;Refazer o refino do
                rosto&quot;.{" "}
                <a href={fileUrl(project.id, "comparar_rosto.png", compare.concluido_em)} target="_blank" rel="noreferrer">
                  Ver um quadro parado
                </a>
              </p>
              <div className="choices">
                {compare.variantes.map((variant) => {
                  if (variant.trocador === null || variant.restaurador === null) return null;
                  const { trocador, restaurador } = variant;
                  const chosen = trocador === faceSettings.trocador && restaurador === faceSettings.restaurador;
                  return (
                    <button
                      key={variant.id}
                      className={chosen ? "primary" : undefined}
                      disabled={locked || chosen}
                      onClick={() =>
                        run("Salvando o ajuste do rosto…", () =>
                          engine.saveFaceSettings(project.id, { trocador, restaurador }),
                        )
                      }
                    >
                      {settingsLabel(trocador, restaurador)}
                      {chosen ? " (escolhido)" : ""}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {wan["comparativo.mp4"] && (
        <video className="wan-compare" src={fileUrl(project.id, "comparativo.mp4", version)} controls playsInline />
      )}
    </section>
  );
}

const SWAPPER_NAMES: Record<FaceSwapper, string> = {
  hyperswap_1a_256: "hyperswap 1a",
  hyperswap_1b_256: "hyperswap 1b",
  hyperswap_1c_256: "hyperswap 1c",
};

function settingsLabel(swapper: FaceSwapper, enhancer: number): string {
  return `${SWAPPER_NAMES[swapper]}, ${enhancer > 0 ? `restaurador ${enhancer}%` : "sem restaurador"}`;
}

function formatDuration(seconds: number): string {
  return seconds < 90 ? `${Math.round(seconds)} s` : `${Math.round(seconds / 60)} min`;
}

function formatHours(hours: number): string {
  return `${hours.toLocaleString("pt-BR", { maximumFractionDigits: 1 })} h`;
}

function formatDate(isoDate: string): string {
  const [year, month, day] = isoDate.split("-");
  return `${day}/${month}/${year}`;
}
