# AI Video Studio — notas para o Claude Code

- O usuário fala português (PT-BR); responda, comente e escreva mensagens de interface em PT-BR.
- Orçamento R$ 0: nunca adicione API paga, serviço com cobrança automática ou dependência que exija
  cartão.
- Uma fase só avança quando o critério em `docs/ROADMAP.md` for verificado com vídeo real. Atualize
  a coluna Status ao concluir algo.
- Hardware alvo: Windows + WSL2 (Ubuntu), provavelmente sem GPU NVIDIA. Tudo pesado precisa ter
  caminho em CPU ou rodar remotamente (Kaggle).
- Uso declarado pelo usuário (10/2026): não é para anúncios pagos, e a meta é o rosto final ficar
  igual às fotos enviadas. Modelos com licença não comercial ou de pesquisa são aceitos; registre a
  licença de cada um em `docs/LICENCAS.md`. Filtros de segurança das ferramentas (ex.: o filtro de
  conteúdo adulto do FaceFusion) não são desligados.

## Comandos

```bash
bash scripts/setup.sh                              # instala tudo (WSL/Linux)
bash scripts/start.sh                              # motor :8765 + painel :3000
cd engine && .venv/bin/python -m pytest -q         # testes do motor
cd web && npm run typecheck && npm run build       # checagens do painel
STUDIO_TRACKER=demo engine/.venv/bin/studio serve  # motor sem IA, para mexer na interface
pwsh -NoProfile -File scripts/tests/instalar-windows.test.ps1  # instalador do Windows com WSL simulado
bash scripts/tests/setup.test.sh && bash scripts/tests/torch-choice.test.sh  # setup.sh com pip/GPU/disco simulados
```

- Scripts `.ps1`:
  - só ASCII (o PowerShell 5.1 lê UTF-8 sem BOM como ANSI);
  - lembre que nomes de variáveis não diferenciam maiúsculas (`$distro` == `$Distro`);
  - comandos nativos que precisam de console (sudo, progresso) são chamados direto, fora de
    funções e sem capturar a saída.

## Arquitetura

- `engine/studio/session.py`:
  - As máscaras em disco são a fonte da verdade.
  - O estado do SAM 2 vive só na memória. Ao reconstruir, os quadros com cliques viram condição
    via `add_new_mask` com a máscara do disco.
  - Desfazer restaura o PNG salvo em `history/<click_id>.png`.
- Tarefas longas (rastrear, exportar, refino do rosto) rodam em um único worker (`jobs.py`), com
  progresso no SQLite. Cliques retornam 409 enquanto há tarefa ativa.
- Encerrar e cair:
  - `studio serve` trava `data/studio.lock` (um motor por pasta de dados; um segundo sai com
    código 3). Ao subir, o motor marca como interrompidas as tarefas que ficaram rodando.
  - O motor roda sem log de acesso. SIGHUP (janela fechada) e SIGTERM esperam até 10 s a tarefa
    parar e gravar "interrompido".
  - `start.sh` põe motor e painel cada um no seu `setsid` e grava o motor em
    `data/logs/motor.log`.
  - O lançador do Windows (gerado pelo `instalar-windows.ps1`):
    - segura a suspensão e desliga a Edição Rápida;
    - não sobe um segundo estúdio;
    - não some sem mostrar o código de saída.
- Memória (a causa medida das quedas no refino):
  - O SAM 2 guarda o clipe inteiro (~2 GB em 5 s a 30 fps) e o FaceFusion chega a 5,4 GB.
  - Antes do refino, o motor solta o vídeo do SAM 2 (`session.release()`) e confere a RAM livre
    mais o swap (`facefusion_min_memory_gb`).
  - O FaceFusion e o medidor de rostos saem com `setpriv --pdeathsig KILL` e `oom_score_adj`
    1000: morrem junto com o motor e, se faltar memória, são eles os mortos, não o motor.
- O painel (`web/`) fala direto com o motor via CORS (`NEXT_PUBLIC_ENGINE_URL`).
- Fase 3:
  - `remote/kaggle/aivs_wan.py` é a fonte do notebook. Depois de editá-lo, rode
    `python remote/kaggle/build_notebook.py` (há um teste que falha se o `.ipynb` estiver
    desatualizado).
  - O motor envia o pacote ao Kaggle por `engine/studio/kaggle_remote.py`, com tarefas `kaggle`
    numa fila separada.
  - No Kaggle: só execução em lote, privada, sem túnel e sem interface.
  - O notebook entrega `resultado.mp4` composto sobre o vídeo original, no tamanho da geração.
  - **Nada de FaceFusion nem outra ferramenta de troca de rosto no Kaggle.** O Kaggle bloqueia
    ("FaceFusion and similar are blocked", 08/2026), cancela a sessão sem erro e pode banir a
    conta. Não contorne o bloqueio. Há um teste que falha se o notebook citar o FaceFusion no
    código.
  - O refino do rosto roda no motor (`engine/studio/face_refine.py`): FaceFusion 3.9.1 na CPU,
    instalado pelo `setup.sh` em `engine/.vendor/facefusion`, tarefa `face_refine` disparada
    sozinha depois de importar o resultado. Usa as fotos de `faces/` (ou a referência). O
    FaceFusion troca todos os rostos da área em blocos (`--face-selector-mode many`), mas só volta
    para o vídeo o rosto que fica na maior parte dentro da silhueta da pessoa rastreada (SAM 2 +
    4 px). Com "one", o rosto de quem estava colado nela era trocado em alguns quadros e piscava.
    De perfil o FaceFusion estraga a troca: `facefusion_probe.py` (roda com o Python do FaceFusion)
    mede o ângulo do rosto em cada quadro, e só entram os quadros de frente até três quartos, em
    trechos de 3+ quadros com rampa de entrada e saída.
    Grava `wan_final.mp4` em 1080x1920 com o áudio.
  - O FaceFusion local pede Ubuntu 24.04 (Python 3.11+, FFmpeg 5.1+). O `setup.sh` grava
    `.venv/.aivs-ok` só quando a instalação termina; sem ele, o motor não o considera instalado.
